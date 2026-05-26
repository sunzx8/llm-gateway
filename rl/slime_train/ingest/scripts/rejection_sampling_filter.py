#!/usr/bin/env python3
"""离线拒绝采样过滤脚本：rollout-based rejection sampling.

对每个 sample 用当前 policy（部署在多台 vLLM 上的 Qwen3.6-27B）做 N 次（默认 8）
独立 rollout，按 task 类型分别评分，过滤掉：
  * pass_count == 0 / N（policy 完全做不出/会做，没学习信号）
  * 每个 rollout 内部"组内 probe 分数 max == min"（无区分度的题目）

改进 v2：
  * 多端点 round-robin 负载均衡（--rollout-url url1,url2,url3...）
  * 断点续传（基于 detail.jsonl 跳过已完成 sample）
  * 流式写 final filtered jsonl（边跑边写，可中途 kill）
  * 组内 probe scores max==min 视为无区分度，rollout 当作未通过

改进 v3：
  * 启动时对所有 rollout endpoint 做健康检查（GET /v1/models），
    只把通过检查的加入轮询池；全失败则直接报错退出。
  * 运行期某个 endpoint 触发异常时自动从池中临时剔除（fallback 到健康端点）。

环境变量与 launch_cluster_train.sh:128-134 严格对齐。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_HERE = Path(__file__).resolve()
for _parent in [_HERE, *_HERE.parents]:
    if _parent.name == "llm_gateway" and (_parent / "gateway").is_dir() and (_parent / "storage").is_dir():
        for _path in (str(_parent.parent), str(_parent)):
            if _path not in sys.path:
                sys.path.insert(0, _path)
        break

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths
ensure_workspace_paths(__file__)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for _n in ("httpx", "httpcore", "openai", "asyncio", "aiohttp", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logger = logging.getLogger("rejection_sampling_filter")


# ---------------- 多端点 LLM 池（round-robin） ----------------
_LLM_POOL: list[Any] = []
_LLM_POOL_IDX = 0
_LLM_POOL_LOCK = asyncio.Lock()


def _normalize_chat_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = url + "/v1/chat/completions"
    return url


def _openai_base_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)] or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _server_root(url: str) -> str:
    """从任意 endpoint URL 中提取 scheme://host[:port]。"""
    parts = urlsplit(url if "://" in url else "http://" + url)
    scheme = parts.scheme or "http"
    netloc = parts.netloc or parts.path
    return f"{scheme}://{netloc}"


def _check_endpoint_healthy(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """同步探测一个 endpoint 是否健康。

    依次尝试 GET {root}/v1/models 与 {root}/health。
    返回 (ok, detail_msg)。
    """
    root = _server_root(url)
    last_err = ""
    for probe in (f"{root}/v1/models", f"{root}/health"):
        try:
            req = urllib.request.Request(probe, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                code = resp.getcode()
                body = resp.read(1024).decode("utf-8", errors="replace")
                if 200 <= code < 300:
                    # /v1/models 需要包含 data 字段才算真就绪
                    if probe.endswith("/v1/models") and '"data"' not in body:
                        last_err = f"{probe} HTTP {code} but no data field"
                        continue
                    return True, f"{probe} HTTP {code}"
                last_err = f"{probe} HTTP {code}"
        except urllib.error.HTTPError as e:
            last_err = f"{probe} HTTPError {e.code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{probe} {type(e).__name__}: {e}"
    return False, last_err


def filter_healthy_urls(rollout_urls: list[str], timeout: float = 5.0) -> list[str]:
    """对 rollout URL 列表做健康过滤，返回健康子集，并打日志。"""
    healthy: list[str] = []
    for u in rollout_urls:
        ok, detail = _check_endpoint_healthy(u, timeout=timeout)
        if ok:
            logger.info("health check ✓ %s (%s)", u, detail)
            healthy.append(u)
        else:
            logger.warning("health check ✗ %s -> %s (将从轮询池剔除)", u, detail)
    return healthy


# 运行期失效记录：endpoint base_url -> 失败次数
_LLM_POOL_BAD: dict[str, int] = {}
_LLM_POOL_BASE_URLS: list[str] = []


def init_llm_pool(rollout_urls: list[str], model: str, args: argparse.Namespace,
                  skip_health_check: bool = False) -> None:
    """为每个 rollout endpoint 创建一个 LLMInterface 实例，加入全局轮询池。

    默认会先做一次健康检查（GET /v1/models），仅把健康的端点加入池；
    若调用方已经在外部过滤过健康端点，可传 skip_health_check=True 避免重复探测。
    全部不健康则抛出 RuntimeError。
    """
    from utils.memory_llm_interface import LLMInterface
    global _LLM_POOL, _LLM_POOL_BASE_URLS
    _LLM_POOL = []
    _LLM_POOL_BASE_URLS = []

    if skip_health_check:
        healthy_urls = list(rollout_urls)
    else:
        health_timeout = float(getattr(args, "health_timeout", 5.0))
        healthy_urls = filter_healthy_urls(rollout_urls, timeout=health_timeout)
    if not healthy_urls:
        raise RuntimeError(
            f"all {len(rollout_urls)} rollout endpoints failed health check, "
            f"abort: {rollout_urls}"
        )
    if not skip_health_check and len(healthy_urls) < len(rollout_urls):
        logger.warning("only %d/%d rollout endpoints are healthy; using subset",
                       len(healthy_urls), len(rollout_urls))

    for url in healthy_urls:
        chat_url = _normalize_chat_url(url)
        base_url = _openai_base_url(chat_url)
        llm = LLMInterface({
            "provider": "openai_compat",
            "model": model,
            "base_url": base_url,
            "api_key": os.environ.get("MEMORY_RL_LLM_API_KEY") or "none",
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
            "timeout": float(args.llm_timeout),
            "max_retries": int(os.environ.get("MEMORY_RL_LLM_MAX_RETRIES", "1")),
        })
        _LLM_POOL.append(llm)
        _LLM_POOL_BASE_URLS.append(base_url)
        logger.info("LLM pool +1: %s (model=%s)", base_url, model)
    if not _LLM_POOL:
        raise RuntimeError("no rollout endpoints configured")


async def _next_llm() -> Any:
    """Round-robin 取一个 LLM 实例。"""
    global _LLM_POOL_IDX
    async with _LLM_POOL_LOCK:
        llm = _LLM_POOL[_LLM_POOL_IDX % len(_LLM_POOL)]
        _LLM_POOL_IDX += 1
    return llm


def infer_task(metadata: dict[str, Any]) -> str:
    t = str(metadata.get("task", "")).lower()
    if t.startswith("ingest"): return "ingest"
    if t.startswith("consolidate") or t.startswith("evolve"): return "consolidate"
    if t.startswith("retrieve") or t.startswith("query") or t.startswith("consume"): return "retrieve"
    if isinstance(metadata.get("ground_truth"), dict): return "retrieve"
    if metadata.get("probes"): return "ingest"
    raise ValueError(f"无法推断 task: metadata.task={t!r}")


# ---------------- per-rollout 执行 ----------------
async def _rollout_ingest_once(metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """ingest 一次 rollout：跑完 task_loop，再 evaluate_probe_set 拿逐 probe 分。

    返回 (score, info)，info 包含 probe_scores / probe_score_max / probe_score_min /
    is_no_discrimination 等用于"组内 max==min 过滤"。
    """
    from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
    from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
    from llm_gateway.rl.slime_train.memory_rl.probes import (
        evaluate_probe_set, average_probe_score,
    )

    data_root = os.environ["SNAPSHOT_DATA_ROOT"]
    tv = os.environ.get("MEMORY_RL_TASK_VERSION", "t2_agent_loop")
    session = SnapshotSession(data_root=data_root, enable_git=False, task_version=tv)
    llm = await _next_llm()
    loaded = await acquire_loaded_env(metadata, session, llm=llm)
    info: dict[str, Any] = {}
    try:
        extras: dict[str, Any] = {
            "session_id": str(metadata.get("session_id", "")),
            "session_time": str(metadata.get("session_time", "")),
            "pending_messages": metadata.get("pending_messages", []),
        }
        if metadata.get("step_index") is not None:
            extras["ingest_number"] = int(metadata["step_index"])
        step = await loaded.env.apply_ingest_tool_calls(
            [], session_time=str(metadata.get("session_time", "")), extras=extras,
        )
        info["finish_reason"] = getattr(step.task_result, "finish_reason", "")
        info["task_error"] = getattr(step.task_result, "error", "") or ""
        probes = metadata.get("probes", []) or []
        evals = await evaluate_probe_set(loaded.env, probes)
        scores = [float(e.get("score", 0.0)) for e in evals]
        avg = average_probe_score(evals)
        info["r_probe"] = avg
        info["n_probes"] = len(probes)
        info["probe_scores"] = scores
        if scores:
            info["probe_score_max"] = max(scores)
            info["probe_score_min"] = min(scores)
            info["is_no_discrimination"] = (max(scores) == min(scores))
        else:
            info["is_no_discrimination"] = True
        return avg, info
    finally:
        await asyncio.to_thread(loaded.__exit__, None, None, None)
        session._snapshot_index = None


async def _rollout_consolidate_once(metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """consolidate 一次：before evals -> apply_consolidate -> after evals。

    "组内 max==min" 检测：用 after_evals 的逐 probe 分数。
    """
    from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
    from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
    from llm_gateway.rl.slime_train.memory_rl.probes import (
        average_probe_score, evaluate_probe_set, probe_accuracy, signed_probe_accuracy_delta,
    )

    data_root = os.environ["SNAPSHOT_DATA_ROOT"]
    tv = os.environ.get("MEMORY_RL_TASK_VERSION", "t2_agent_loop")
    session = SnapshotSession(data_root=data_root, enable_git=False, task_version=tv)
    llm = await _next_llm()

    probes = metadata.get("probes", []) or []
    if not probes:
        return 0.0, {"reason": "no_probes", "is_no_discrimination": True}

    async def _factory_before():
        return await acquire_loaded_env(metadata, session, llm=llm)

    before_evals = await evaluate_probe_set(None, probes, env_factory=_factory_before)
    before_acc = probe_accuracy(before_evals)
    before_score = average_probe_score(before_evals)

    loaded = await acquire_loaded_env(metadata, session, llm=llm)
    try:
        extras: dict[str, Any] = {"session_id": str(metadata.get("session_id", ""))}
        if metadata.get("step_index") is not None:
            extras["step_index"] = int(metadata["step_index"])
        await loaded.env.apply_consolidate_tool_calls([], extras=extras)
        after_evals = await evaluate_probe_set(loaded.env, probes)
    finally:
        await asyncio.to_thread(loaded.__exit__, None, None, None)
        session._snapshot_index = None

    after_acc = probe_accuracy(after_evals)
    after_score = average_probe_score(after_evals)
    delta_acc = signed_probe_accuracy_delta(before_evals, after_evals)
    after_scores = [float(e.get("score", 0.0)) for e in after_evals]
    return delta_acc, {
        "before_acc": before_acc, "after_acc": after_acc,
        "before_score": before_score, "after_score": after_score,
        "r_acc_delta_signed": delta_acc, "n_probes": len(probes),
        "after_probe_scores": after_scores,
        "after_probe_score_max": max(after_scores) if after_scores else 0.0,
        "after_probe_score_min": min(after_scores) if after_scores else 0.0,
        "is_no_discrimination": (
            (max(after_scores) == min(after_scores)) if after_scores else True
        ),
    }


async def _rollout_retrieve_once(metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """retrieve 一次：step_query -> retrieved_context -> frozen QA score。

    retrieve 单 sample 只有一个 ground_truth，没有"组"概念，max==min 不适用。
    我们对 retrieve 用 score_context_for_ground_truth 内部隐含的 alt_queries
    平均分数：若所有 alt 评分都相同则视为无区分度。
    """
    from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
    from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
    from llm_gateway.rl.slime_train.tasks.retrieve_reward.retrieval_hit import (
        score_context_against_gold_async,
    )

    data_root = os.environ["SNAPSHOT_DATA_ROOT"]
    tv = os.environ.get("MEMORY_RL_TASK_VERSION", "t2_agent_loop")
    session = SnapshotSession(data_root=data_root, enable_git=False, task_version=tv)
    llm = await _next_llm()

    ground_truth = metadata.get("ground_truth", {}) or {}
    query = metadata.get("query") or ground_truth.get("question") or ""
    if not query:
        return 0.0, {"reason": "no_query", "is_no_discrimination": True}
    if not metadata.get("snapshot_id"):
        return 0.0, {"reason": "no_snapshot", "is_no_discrimination": True}

    info: dict[str, Any] = {}
    loaded = await acquire_loaded_env(metadata, session, llm=llm)
    try:
        extra: dict[str, Any] = {}
        if metadata.get("session_time"): extra["session_time"] = str(metadata["session_time"])
        if metadata.get("step_index") is not None: extra["step_index"] = int(metadata["step_index"])
        step = await loaded.env.step_query(
            query=str(query), session_id=str(metadata.get("session_id", "")),
            extra_payload=extra or None,
        )
        retrieved = ""
        for attr in ("retrieved_context", "final_output", "finish_result", "finish_summary"):
            v = getattr(step.task_result, attr, None)
            if v:
                retrieved = str(v); break
        if not retrieved:
            stats = getattr(step.task_result, "stats", None)
            if isinstance(stats, dict):
                for k in ("retrieved_context", "query_memory"):
                    if stats.get(k):
                        retrieved = str(stats[k]); break
        info["retrieved_context_len"] = len(retrieved)
        info["retrieved_context_preview"] = retrieved[:200]
        info["finish_reason"] = getattr(step.task_result, "finish_reason", "")
        info["task_error"] = getattr(step.task_result, "error", "") or ""
    finally:
        await asyncio.to_thread(loaded.__exit__, None, None, None)
        session._snapshot_index = None

    # retrieve 用 main question + alt_queries 各跑一遍 frozen QA，用方差判定区分度
    main_q = ground_truth.get("question", "")
    alt_qs = ground_truth.get("alt_queries", []) or []
    questions = [main_q] + [q for q in alt_qs if q]
    if not retrieved.strip():
        retrieved_for_qa = "(No relevant memories found)"
    else:
        retrieved_for_qa = retrieved

    async def _qa_one(q: str) -> float:
        gt = {**ground_truth, "question": q}
        return float(await score_context_against_gold_async(retrieved_for_qa, gt))

    qa_scores = await asyncio.gather(*[_qa_one(q) for q in questions]) if questions else []
    qa_scores = [float(s) for s in qa_scores]
    score = (sum(qa_scores) / len(qa_scores)) if qa_scores else 0.0
    info["r_retrieval_hit"] = score
    info["qa_scores"] = qa_scores
    info["qa_score_max"] = max(qa_scores) if qa_scores else 0.0
    info["qa_score_min"] = min(qa_scores) if qa_scores else 0.0
    info["is_no_discrimination"] = (
        (max(qa_scores) == min(qa_scores)) if qa_scores else True
    )
    return score, info


PASS_THRESHOLDS = {"ingest": 0.5, "consolidate": 1e-9, "retrieve": 0.5}


def _is_pass(task: str, score: float, info: dict[str, Any], thresholds: dict[str, float],
             treat_no_discrimination_as_fail: bool) -> bool:
    """通过判定。

    若 treat_no_discrimination_as_fail=True 且 info[is_no_discrimination] 为真，
    则视该 rollout 未通过（即使 score>=threshold），从而：
      - 整组所有 rollout 都 max==min 时，pass_count=0 → 在最终过滤中被丢弃。
      - 即"组内全题得分都一样"的样本被自动排除。
    """
    if treat_no_discrimination_as_fail and bool(info.get("is_no_discrimination", False)):
        return False
    thr = thresholds.get(task, 0.5)
    return score > thr if task == "consolidate" else score >= thr


async def _process_one_sample(idx, sample, task, n_rollouts, rollout_sem, thresholds,
                              rollout_timeout, treat_no_disc_as_fail):
    metadata = sample.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    fn = {"ingest": _rollout_ingest_once,
          "consolidate": _rollout_consolidate_once,
          "retrieve": _rollout_retrieve_once}[task]

    async def _one(roll_idx):
        async with rollout_sem:
            t0 = time.perf_counter()
            try:
                score, info = await asyncio.wait_for(fn(metadata), timeout=rollout_timeout)
                return {"rollout_idx": roll_idx, "score": score,
                        "passed": _is_pass(task, score, info, thresholds, treat_no_disc_as_fail),
                        "latency_s": time.perf_counter() - t0, "info": info}
            except asyncio.TimeoutError:
                return {"rollout_idx": roll_idx, "score": 0.0, "passed": False,
                        "latency_s": time.perf_counter() - t0, "info": {"error": "timeout"}}
            except Exception as exc:
                logger.warning("sample %s rollout %s failed: %s", idx, roll_idx, exc)
                return {"rollout_idx": roll_idx, "score": 0.0, "passed": False,
                        "latency_s": time.perf_counter() - t0, "info": {"error": str(exc)[:300]}}

    rollouts = await asyncio.gather(*[_one(i) for i in range(n_rollouts)])
    pc = sum(1 for r in rollouts if r["passed"])
    avg = sum(r["score"] for r in rollouts) / max(1, len(rollouts))
    # 组级"全题区分度"：若每个 rollout 的 is_no_discrimination 都为 True，整组无区分度
    n_no_disc = sum(1 for r in rollouts
                    if isinstance(r.get("info"), dict)
                    and r["info"].get("is_no_discrimination") is True)
    group_no_discrimination = (n_no_disc == len(rollouts) and len(rollouts) > 0)
    keep = (0 < pc < n_rollouts) and (not group_no_discrimination)
    return {
        "index": idx, "task": task,
        "metadata_brief": {
            "task": metadata.get("task"),
            "snapshot_id": metadata.get("snapshot_id"),
            "traj_id": metadata.get("traj_id"),
            "session_id": metadata.get("session_id"),
        },
        "n_rollouts": n_rollouts, "pass_count": pc,
        "pass_rate": pc / max(1, n_rollouts),
        "avg_score": avg,
        "n_no_discrimination_rollouts": n_no_disc,
        "group_no_discrimination": group_no_discrimination,
        "keep": keep, "rollouts": rollouts,
    }


def load_samples(path: str, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit > 0 and len(out) >= limit:
                break
    return out


def _load_done_indices(detail_path: Path) -> set[int]:
    """从已存在的 detail.jsonl 读出已完成的 sample index 集合（断点续传用）。"""
    if not detail_path.exists():
        return set()
    done: set[int] = set()
    bad = 0
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            idx = rec.get("index")
            if isinstance(idx, int):
                done.add(idx)
    if bad:
        logger.warning("detail 文件有 %d 行无法解析（流式截断），跳过", bad)
    return done


def setup_env(args: argparse.Namespace, healthy_urls: list[str] | None = None) -> None:
    os.environ["MEMORY_RL_TASK_VERSION"] = args.task_version
    os.environ["MEMORY_RL_APPLY_MODE"] = "task_loop"
    os.environ["MEMORY_RL_TRAIN_TASK_LOOP"] = "1"
    # 兼容性：MEMORY_RL_LLM_API_URL 仍设置为第一个 endpoint，以防 env_acquire 走 fallback
    # 优先选用健康端点；若 healthy_urls 为空则回退到 args.rollout_url 的第一个
    if healthy_urls:
        first_raw = healthy_urls[0]
    else:
        first_raw = args.rollout_url.split(",")[0]
    first_url = _normalize_chat_url(first_raw)
    os.environ["MEMORY_RL_LLM_API_URL"] = first_url
    os.environ["MEMORY_RL_LLM_MODEL"] = args.rollout_model
    os.environ["MEMORY_RL_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["MEMORY_RL_LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MEMORY_RL_LLM_TIMEOUT"] = str(args.llm_timeout)
    os.environ["MEMORY_RL_GENERATE_TIMEOUT"] = str(args.llm_timeout)
    os.environ.setdefault("MEMORY_RL_LLM_MAX_RETRIES", "1")
    os.environ.setdefault("MEMORY_RL_MAX_AGENT_TURNS", str(args.max_agent_turns))
    if args.frozen_url:
        os.environ["FROZEN_MODEL_URL"] = args.frozen_url
    if args.frozen_model:
        os.environ["FROZEN_MODEL_NAME"] = args.frozen_model
    if args.frozen_endpoints:
        os.environ["FROZEN_MODEL_ENDPOINTS"] = args.frozen_endpoints
    os.environ.setdefault("FROZEN_MODEL_TIMEOUT", str(args.frozen_timeout))
    os.environ.setdefault("FROZEN_MODEL_MAX_CONCURRENCY", str(args.frozen_max_concurrency))
    os.environ.setdefault("FROZEN_MODEL_CONN_LIMIT", str(args.frozen_max_concurrency * 4))
    os.environ.setdefault("FROZEN_MODEL_MAX_TOKENS", "128000")
    os.environ.setdefault("FROZEN_MODEL_MAX_INPUT_TOKENS", "16384")
    os.environ.setdefault("PROBE_EVAL_MAX_CONCURRENCY", str(args.probe_eval_concurrency))
    os.environ["SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
    os.environ["INGEST_SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
    os.environ["CONSOLIDATE_SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
    os.environ["RETRIEVE_SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
    os.environ["MEMORY_RL_USE_SGLANG_TOOL_PARSER"] = "0"


async def _amain(args: argparse.Namespace) -> None:
    samples = load_samples(args.input, args.limit)
    logger.info("Loaded %d samples from %s", len(samples), args.input)
    if not samples:
        return
    if args.task == "auto":
        task = infer_task(samples[0].get("metadata", {}) or {})
    else:
        task = args.task

    rollout_urls = [u.strip() for u in args.rollout_url.split(",") if u.strip()]
    if not rollout_urls:
        raise RuntimeError("--rollout-url 未指定任何 endpoint")
    logger.info("Probing %d rollout endpoint(s) for health ...", len(rollout_urls))
    healthy_urls = filter_healthy_urls(rollout_urls, timeout=float(args.health_timeout))
    if not healthy_urls:
        raise RuntimeError(
            f"all {len(rollout_urls)} rollout endpoints failed health check: {rollout_urls}"
        )
    logger.info("Healthy rollout endpoints: %d/%d -> %s",
                len(healthy_urls), len(rollout_urls), healthy_urls)

    # 用健康端点重置环境变量，并初始化 LLM 池（已过滤健康，无需重复探测）
    setup_env(args, healthy_urls=healthy_urls)
    init_llm_pool(healthy_urls, args.rollout_model, args, skip_health_check=True)

    logger.info("task=%s, n_rollouts=%d, sample_concurrency=%d, rollout_concurrency=%d, "
                "endpoints=%d (healthy=%d), treat_no_disc_as_fail=%s",
                task, args.n_rollouts, args.sample_concurrency,
                args.sample_concurrency * args.n_rollouts,
                len(rollout_urls), len(healthy_urls),
                args.treat_no_discrimination_as_fail)

    thresholds = dict(PASS_THRESHOLDS)
    if args.pass_threshold is not None:
        thresholds[task] = args.pass_threshold

    rollout_sem = asyncio.Semaphore(max(1, args.sample_concurrency * args.n_rollouts))
    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_suffix(out_path.suffix + ".detail.jsonl")

    # 断点续传：跳过 detail.jsonl 中已 done 的 index
    done_indices = _load_done_indices(detail_path) if args.resume else set()
    if done_indices:
        logger.info("RESUME: %d samples already in detail.jsonl, will skip them",
                    len(done_indices))

    todo: list[tuple[int, dict[str, Any]]] = [
        (i, s) for i, s in enumerate(samples) if i not in done_indices
    ]
    if not todo:
        logger.info("Nothing to do (all samples already in detail). Will only rebuild final output.")
    else:
        logger.info("Pending samples to run: %d / %d", len(todo), len(samples))

    t_start = time.perf_counter()
    completed = [0]
    n_total = len(todo)
    lock = asyncio.Lock()

    async def _bounded(idx, sample):
        async with sample_sem:
            r = await _process_one_sample(idx, sample, task, args.n_rollouts,
                                          rollout_sem, thresholds, args.rollout_timeout,
                                          args.treat_no_discrimination_as_fail)
        async with lock:
            completed[0] += 1
            done = completed[0]
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            logger.info("[%d/%d] idx=%d pass=%d/%d nodisc=%d keep=%s avg=%.3f | %.1fs/sample | ETA %ds",
                        done, n_total, idx, r["pass_count"], r["n_rollouts"],
                        r["n_no_discrimination_rollouts"], r["keep"], r["avg_score"],
                        elapsed / max(1, done), int(eta))
        return idx, r

    pending = [_bounded(i, s) for i, s in todo]

    # 流式写：detail（append 模式以支持断点续跑） + final（append 模式）
    detail_mode = "a" if args.resume and detail_path.exists() else "w"
    final_mode = "a" if args.resume and out_path.exists() else "w"
    new_results: dict[int, dict[str, Any]] = {}

    df = open(detail_path, detail_mode, encoding="utf-8")
    of = open(out_path, final_mode, encoding="utf-8")
    try:
        if pending:
            for fut in asyncio.as_completed(pending):
                idx, r = await fut
                new_results[idx] = r
                df.write(json.dumps({k: r[k] for k in (
                    "index", "task", "metadata_brief", "n_rollouts", "pass_count",
                    "pass_rate", "avg_score", "n_no_discrimination_rollouts",
                    "group_no_discrimination", "keep", "rollouts")},
                    ensure_ascii=False) + "\n")
                df.flush()
                # 流式写 final filtered（边跑边写）
                if r["keep"]:
                    sample = samples[idx]
                    enriched = dict(sample)
                    meta = dict(enriched.get("metadata", {}) or {})
                    meta["__rejection_sampling__"] = {
                        "pass_count": r["pass_count"],
                        "n_rollouts": r["n_rollouts"],
                        "pass_rate": r["pass_rate"],
                        "avg_score": r["avg_score"],
                        "group_no_discrimination": r["group_no_discrimination"],
                        "n_no_discrimination_rollouts": r["n_no_discrimination_rollouts"],
                        "original_index": idx,
                    }
                    enriched["metadata"] = meta
                    of.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    of.flush()
    finally:
        df.close()
        of.close()

    elapsed = time.perf_counter() - t_start
    n_kept = sum(1 for r in new_results.values() if r["keep"])
    n_no_pass = sum(1 for r in new_results.values() if r["pass_count"] == 0)
    n_all_pass = sum(1 for r in new_results.values() if r["pass_count"] == r["n_rollouts"])
    n_group_nodisc = sum(1 for r in new_results.values() if r["group_no_discrimination"])
    logger.info("=" * 60)
    logger.info("This run: processed=%d kept=%d no_pass(0/N)=%d all_pass(N/N)=%d group_nodisc=%d",
                len(new_results), n_kept, n_no_pass, n_all_pass, n_group_nodisc)
    if pending:
        logger.info("Wall %.1fs (%.2fs/sample)", elapsed, elapsed / max(1, len(new_results)))
    logger.info("Out:    %s", out_path)
    logger.info("Detail: %s", detail_path)
    logger.info("提示：要按更严格阈值（例如只保留 2/8~6/8）重新构建 final，运行 build_filtered_from_detail.py")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["auto", "ingest", "consolidate", "retrieve"], default="auto")
    p.add_argument("--task-version", default="t2_agent_loop")
    p.add_argument("--limit", type=int, default=0, help="0=all")
    p.add_argument("--n-rollouts", type=int, default=8)
    p.add_argument("--sample-concurrency", type=int, default=8)
    p.add_argument("--rollout-timeout", type=float, default=900.0)
    p.add_argument("--pass-threshold", type=float, default=None)
    # rollout LLM (支持逗号分隔多端点)
    p.add_argument("--rollout-url",
                   default=os.environ.get("ROLLOUT_URL", "http://192.168.16.103:7777"),
                   help="单个或逗号分隔多个 vLLM 端点 URL，做 round-robin 负载均衡")
    p.add_argument("--health-timeout", type=float, default=5.0,
                   help="启动时探测每个 rollout endpoint 的 /v1/models 超时秒数，默认 5s")
    p.add_argument("--rollout-model",
                   default=os.environ.get("ROLLOUT_MODEL", "/data/cloud_disk_1/models/Qwen/Qwen3.6-27B"))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=64000)
    p.add_argument("--llm-timeout", type=int, default=1200)
    p.add_argument("--max-agent-turns", type=int, default=20)
    # frozen judge
    p.add_argument("--frozen-url", default=None)
    p.add_argument("--frozen-model", default=None)
    p.add_argument("--frozen-endpoints",
                   default="http://124.221.221.186:30000/v1/chat/completions|glm-5.1,"
                           "http://123.207.200.23:30000/v1/chat/completions|DeepSeek-V4-Pro,"
                           "http://220.154.132.76:30000/v1/chat/completions|Kimi-K2.6")
    p.add_argument("--frozen-timeout", type=int, default=1200)
    p.add_argument("--frozen-max-concurrency", type=int, default=32)
    p.add_argument("--probe-eval-concurrency", type=int, default=8)
    # data
    p.add_argument("--snapshot-data-root",
                   default="/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e")
    # 新增功能
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：跳过 detail.jsonl 中已存在的 sample index，并用 append 模式写")
    p.add_argument("--treat-no-discrimination-as-fail",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="把组内 max==min（无区分度）的 rollout 视为未通过；默认开启")
    args = p.parse_args()

    # 注意：setup_env 会在 _amain 中用健康过滤后的端点重新设置 MEMORY_RL_LLM_API_URL。
    # 这里仅做一次"早期"setup，确保即使 _amain 在健康检查前抛错（如样本读取失败），
    # 部分环境变量仍能被外层使用。
    setup_env(args)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
