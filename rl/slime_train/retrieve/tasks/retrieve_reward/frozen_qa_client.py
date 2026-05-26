"""HTTP client helpers for frozen-model QA calls used by Retrieve reward.

Supports multi-endpoint load balancing via FROZEN_MODEL_ENDPOINTS env var.
Format: "url|model_name,url|model_name,..."
Falls back to single FROZEN_MODEL_URL + FROZEN_MODEL_NAME if FROZEN_MODEL_ENDPOINTS is not set.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

_aiohttp_sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}
_aiohttp_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


# =============================================================================
# Multi-endpoint load balancing
# =============================================================================

@dataclass
class Endpoint:
    """A single frozen-model endpoint."""
    url: str
    model_name: str
    # Weight for weighted random selection (higher = more traffic)
    weight: float = 1.0
    # Track consecutive failures for circuit-breaker logic
    consecutive_failures: int = 0
    max_failures: int = 5  # Mark as unhealthy after this many consecutive failures
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < self.max_failures

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1


class EndpointPool:
    """Pool of endpoints with load-balancing selection."""

    def __init__(self) -> None:
        self.endpoints: list[Endpoint] = []
        self._rr_counter = itertools.count()
        self._lock = threading.Lock()

    def select(self) -> Endpoint:
        """Select an endpoint using weighted round-robin with health awareness."""
        healthy = [ep for ep in self.endpoints if ep.is_healthy]
        if not healthy:
            # All endpoints unhealthy; reset and try all
            logger.warning("所有冻结模型端点不健康，重置状态并重试")
            for ep in self.endpoints:
                ep.consecutive_failures = 0
            healthy = self.endpoints

        if not healthy:
            raise RuntimeError("没有可用的冻结模型端点")

        # Weighted random selection
        total_weight = sum(ep.weight for ep in healthy)
        r = random.random() * total_weight
        cumulative = 0.0
        for ep in healthy:
            cumulative += ep.weight
            if r <= cumulative:
                return ep
        return healthy[-1]  # fallback


def _parse_endpoints() -> EndpointPool:
    """Parse FROZEN_MODEL_ENDPOINTS or fall back to single endpoint."""
    pool = EndpointPool()

    endpoints_str = os.environ.get("FROZEN_MODEL_ENDPOINTS", "")
    if endpoints_str:
        # Weights are proportional to output throughput (tok/s)
        # CVM1 glm-5.1: 338.8, CVM2 glm-5.1: 212.0, DeepSeek-V4-Pro: 309.1, Kimi-K2.6: 709.7
        # Normalize so they sum to reasonable values
        for entry in endpoints_str.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "|" in entry:
                url, model_name = entry.rsplit("|", 1)
            else:
                url = entry
                model_name = os.environ.get("FROZEN_MODEL_NAME", "default")
            pool.endpoints.append(Endpoint(url=url.strip(), model_name=model_name.strip()))

    if not pool.endpoints:
        # Fallback to single endpoint
        url = os.environ.get("FROZEN_MODEL_URL", "http://localhost:30000/v1/chat/completions")
        model_name = os.environ.get("FROZEN_MODEL_NAME", "default")
        pool.endpoints.append(Endpoint(url=url, model_name=model_name))

    # Assign weights based on known throughput characteristics
    _THROUGHPUT_WEIGHTS = {
        "http://150.158.142.94:30010/v1/chat/completions": 3.4,   # 338.8 tok/s
        "http://124.221.221.186:30000/v1/chat/completions": 2.1,  # 212.0 tok/s
        "http://123.207.200.23:30000/v1/chat/completions": 3.1,   # 309.1 tok/s
        "http://220.154.132.76:30000/v1/chat/completions": 7.1,   # 709.7 tok/s
    }
    for ep in pool.endpoints:
        ep.weight = _THROUGHPUT_WEIGHTS.get(ep.url, 1.0)

    logger.info(
        "冻结模型端点池初始化完成: %d 个端点 [%s]",
        len(pool.endpoints),
        ", ".join(f"{ep.url}({ep.model_name}, w={ep.weight:.1f})" for ep in pool.endpoints),
    )
    return pool


# Lazily initialized singleton
_endpoint_pool: EndpointPool | None = None
_pool_lock = threading.Lock()


def get_endpoint_pool() -> EndpointPool:
    global _endpoint_pool
    if _endpoint_pool is None:
        with _pool_lock:
            if _endpoint_pool is None:
                _endpoint_pool = _parse_endpoints()
    return _endpoint_pool


# =============================================================================
# Session / semaphore management
# =============================================================================

def get_aiohttp_session() -> aiohttp.ClientSession:
    """Return an aiohttp session bound to the current event loop.

    Slime can evaluate rewards from async workers and/or thread pools. A single
    process-global ``ClientSession`` is not safe across multiple event loops, so
    cache one session per running loop instead.
    """
    loop = asyncio.get_running_loop()
    session = _aiohttp_sessions.get(loop)
    if session is None or session.closed:
        timeout = aiohttp.ClientTimeout(total=int(os.environ.get("FROZEN_MODEL_TIMEOUT", "30")))
        connector = aiohttp.TCPConnector(limit=int(os.environ.get("FROZEN_MODEL_CONN_LIMIT", "128")))
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        _aiohttp_sessions[loop] = session
    return session


def get_frozen_model_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _aiohttp_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(int(os.environ.get("FROZEN_MODEL_MAX_CONCURRENCY", "32")))
        _aiohttp_semaphores[loop] = sem
    return sem


async def close_aiohttp_sessions() -> None:
    """Close cached frozen-model HTTP sessions for clean local smoke exits."""
    sessions = list(_aiohttp_sessions.values())
    _aiohttp_sessions.clear()
    _aiohttp_semaphores.clear()
    for session in sessions:
        if not session.closed:
            await session.close()


# =============================================================================
# Core API call functions (with load balancing)
# =============================================================================

async def call_frozen_model_async(messages: list[dict[str, str]]) -> str:
    """Call the frozen OpenAI-compatible chat-completions endpoint asynchronously.

    Uses multi-endpoint load balancing if FROZEN_MODEL_ENDPOINTS is configured.
    On failure, retries with a different endpoint (up to number of available endpoints).
    """
    pool = get_endpoint_pool()
    max_retries = min(len(pool.endpoints), 3)
    tried: set[str] = set()

    for attempt in range(max_retries):
        ep = pool.select()
        # Avoid retrying the same endpoint consecutively
        while ep.url in tried and len(tried) < len(pool.endpoints):
            ep = pool.select()
        tried.add(ep.url)

        payload = {
            "model": ep.model_name,
            "messages": messages,
            "max_tokens": int(os.environ.get("FROZEN_MODEL_MAX_TOKENS", "256")),
            "temperature": float(os.environ.get("FROZEN_MODEL_TEMPERATURE", "0.0")),
        }

        try:
            session = get_aiohttp_session()
            sem = get_frozen_model_semaphore()
            async with sem:
                async with session.post(ep.url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            "冻结模型返回 %s (endpoint=%s, model=%s): %s",
                            resp.status, ep.url, ep.model_name, error_text,
                        )
                        ep.record_failure()
                        continue
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        ep.record_success()
                        return _extract_message_text(choices[0].get("message", {}))
                    ep.record_success()
                    return ""
        except asyncio.TimeoutError:
            logger.warning("冻结模型调用超时 (endpoint=%s, attempt=%d/%d)", ep.url, attempt + 1, max_retries)
            ep.record_failure()
        except Exception as e:
            logger.warning("冻结模型调用异常 (endpoint=%s, attempt=%d/%d): %s", ep.url, attempt + 1, max_retries, e)
            ep.record_failure()

    logger.error("冻结模型调用失败: 已尝试 %d 个端点均失败", max_retries)
    return ""


def _extract_message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()

    # Some OpenAI-compatible servers expose reasoning-only fields or return
    # content=None when no final answer was produced. Treat that as empty rather
    # than raising and spamming rollout logs.
    for key in ("reasoning_content", "reasoning", "text"):
        value = message.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def call_frozen_model(messages: list[dict[str, str]]) -> str:
    """Synchronous frozen-model call with load balancing. Retained for local/manual testing."""
    import urllib.request

    pool = get_endpoint_pool()
    max_retries = min(len(pool.endpoints), 3)
    timeout = int(os.environ.get("FROZEN_MODEL_TIMEOUT", "30"))
    tried: set[str] = set()

    for attempt in range(max_retries):
        ep = pool.select()
        while ep.url in tried and len(tried) < len(pool.endpoints):
            ep = pool.select()
        tried.add(ep.url)

        payload = json.dumps({
            "model": ep.model_name,
            "messages": messages,
            "max_tokens": int(os.environ.get("FROZEN_MODEL_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("FROZEN_MODEL_TEMPERATURE", "0.0")),
        }).encode("utf-8")

        req = urllib.request.Request(
            ep.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    ep.record_success()
                    return _extract_message_text(choices[0].get("message", {}))
                ep.record_success()
                return ""
        except Exception as e:
            logger.warning("冻结模型调用失败 (endpoint=%s, attempt=%d/%d): %s", ep.url, attempt + 1, max_retries, e)
            ep.record_failure()

    logger.error("冻结模型同步调用失败: 已尝试 %d 个端点均失败", max_retries)
    return ""


# Backward-compatible private aliases.
_get_aiohttp_session = get_aiohttp_session
_close_aiohttp_sessions = close_aiohttp_sessions
_call_frozen_model_async = call_frozen_model_async
_call_frozen_model = call_frozen_model
