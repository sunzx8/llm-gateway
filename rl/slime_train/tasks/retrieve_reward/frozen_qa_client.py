"""HTTP client helpers for frozen-model QA calls used by Retrieve reward."""

from __future__ import annotations

import asyncio
import json
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

_aiohttp_sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}
_aiohttp_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


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


async def call_frozen_model_async(messages: list[dict[str, str]]) -> str:
    """Call the frozen OpenAI-compatible chat-completions endpoint asynchronously."""
    url = os.environ.get("FROZEN_MODEL_URL", "http://localhost:30000/v1/chat/completions")
    model_name = os.environ.get("FROZEN_MODEL_NAME", "default")

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": int(os.environ.get("FROZEN_MODEL_MAX_TOKENS", "256")),
        "temperature": float(os.environ.get("FROZEN_MODEL_TEMPERATURE", "0.0")),
    }

    try:
        session = get_aiohttp_session()
        sem = get_frozen_model_semaphore()
        async with sem:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.warning("冻结模型返回 %s: %s", resp.status, await resp.text())
                    return ""
                data = await resp.json()
                choices = data.get("choices", [])
                if choices:
                    return _extract_message_text(choices[0].get("message", {}))
                return ""
    except asyncio.TimeoutError:
        logger.warning("冻结模型调用超时")
        return ""
    except Exception as e:
        logger.warning("冻结模型调用异常: %s", e)
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
    """Synchronous frozen-model call retained for local/manual testing."""
    import urllib.request

    url = os.environ.get("FROZEN_MODEL_URL", "http://localhost:30000/v1/chat/completions")
    model_name = os.environ.get("FROZEN_MODEL_NAME", "default")
    timeout = int(os.environ.get("FROZEN_MODEL_TIMEOUT", "30"))

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": int(os.environ.get("FROZEN_MODEL_MAX_TOKENS", "8192")),
        "temperature": float(os.environ.get("FROZEN_MODEL_TEMPERATURE", "0.0")),
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return _extract_message_text(choices[0].get("message", {}))
            return ""
    except Exception as e:
        logger.warning("冻结模型调用失败: %s", e)
        return ""


# Backward-compatible private aliases.
_get_aiohttp_session = get_aiohttp_session
_close_aiohttp_sessions = close_aiohttp_sessions
_call_frozen_model_async = call_frozen_model_async
_call_frozen_model = call_frozen_model
