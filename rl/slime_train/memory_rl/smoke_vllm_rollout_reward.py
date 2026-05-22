#!/usr/bin/env python3
"""Run rollout + reward smoke tests without Ray/Slime workers.

Supports both single-turn tool-plan rollout and an agentic loop. In agentic
mode, task execution is weakly coupled through an injected tool executor
function/context, so changing task semantics does not require changing the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp

_HERE = Path(__file__).resolve()
for _parent in [_HERE, *_HERE.parents]:
    if _parent.name == "llm_gateway" and (_parent / "gateway").is_dir() and (_parent / "storage").is_dir():
        for _path in (str(_parent.parent), str(_parent)):
            if _path not in sys.path:
                sys.path.insert(0, _path)
        break

from llm_gateway.rl.slime_train.memory_rl.agentic_rollout import extract_tool_calls_from_message, run_agentic_rollout
from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.slime_train.memory_rl.tool_executors import EnvToolExecutor
from llm_gateway.rl.slime_train.memory_rl.tool_schemas import get_tool_schemas


def _default_reward_path(task: str) -> str:
    if task == "ingest":
        return "llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func"
    if task == "consolidate":
        return "llm_gateway.rl.slime_train.tasks.consolidate_reward.reward.reward_func"
    if task == "retrieve":
        return "llm_gateway.rl.slime_train.tasks.retrieve_reward.reward.reward_func"
    if task == "mixed":
        return "llm_gateway.rl.slime_train.tasks.mixed_reward.reward.reward_func"
    raise ValueError(f"unsupported task: {task}")


def _prepare_imports(task: str) -> None:
    ensure_workspace_paths(__file__)


def _import_object(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _load_samples(path: str, limit: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit > 0 and len(samples) >= limit:
                break
    return samples


async def _call_rollout_message(session: aiohttp.ClientSession, messages: list[dict[str, Any]], args) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    api_key = args.api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.native_tools:
        tools = get_tool_schemas(args.task)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
    async with session.post(args.api_url, json=payload, headers=headers) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"rollout http {resp.status}: {text[:500]}")
        data = json.loads(text)
        choices = data.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}
        return choices[0].get("message", {}) or {"role": "assistant", "content": ""}


async def _call_rollout(session: aiohttp.ClientSession, sample: dict[str, Any], args) -> str:
    message = await _call_rollout_message(session, sample["prompt"], args)
    calls = extract_tool_calls_from_message(message)
    if calls:
        return json.dumps({"tool_calls": calls}, ensure_ascii=False)
    return message.get("content", "") or ""


@asynccontextmanager
async def _build_tool_executor(args, metadata: dict[str, Any]):
    if args.tool_executor_path:
        factory = _import_object(args.tool_executor_path)
        executor = factory(task=args.task, metadata=metadata, args=args)
        if hasattr(executor, "__aenter__"):
            async with executor as active:
                yield active
        else:
            yield executor
        return

    async with EnvToolExecutor(args.task, metadata, data_root=args.snapshot_data_root) as executor:
        yield executor


async def _run_agentic_response(
    session: aiohttp.ClientSession,
    sample: dict[str, Any],
    args,
) -> dict[str, Any]:
    async with _build_tool_executor(args, sample.get("metadata", {})) as executor:
        return await run_agentic_rollout(
            sample["prompt"],
            call_model=lambda messages: _call_rollout_message(session, messages, args),
            execute_tool_calls=executor,
            max_turns=args.max_agent_turns,
            use_native_tool_messages=args.native_tools,
        )


def _task_from_sample(sample: dict[str, Any], fallback: str) -> str:
    if fallback != "mixed":
        return fallback
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), dict) else {}
    task = str(metadata.get("task", "")).lower()
    if task.startswith("ingest"):
        return "ingest"
    if task.startswith("consolidate") or task.startswith("evolve"):
        return "consolidate"
    if task.startswith("retrieve") or task.startswith("query") or task.startswith("consume"):
        return "retrieve"
    if isinstance(metadata.get("ground_truth"), dict):
        return "retrieve"
    return "ingest" if metadata.get("probes") else "retrieve"


def _args_for_sample(args, sample: dict[str, Any]):
    if args.task != "mixed":
        return args
    copied = SimpleNamespace(**vars(args))
    copied.task = _task_from_sample(sample, args.task)
    return copied


async def _run_one(index: int, sample: dict[str, Any], session: aiohttp.ClientSession, reward_func, args) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        runtime_args = _args_for_sample(args, sample)
        agentic_trace = None
        stop_reason = None
        tool_call_count = None
        if args.agentic:
            rollout = await _run_agentic_response(session, sample, runtime_args)
            response = rollout["response"]
            agentic_trace = rollout["trace"]
            stop_reason = rollout.get("stop_reason")
            tool_call_count = len(rollout.get("tool_calls") or [])
        else:
            response = await _call_rollout(session, sample, runtime_args)

        reward_sample = SimpleNamespace(
            prompt=sample.get("prompt"),
            response=response,
            metadata=sample.get("metadata", {}),
        )
        reward = None
        if not args.skip_reward:
            reward = await reward_func(args, reward_sample)
        latency = time.perf_counter() - start
        return {
            "index": index,
            "reward": reward,
            "latency_s": latency,
            "response_len": len(response),
            "response": response,
            "agentic_trace": agentic_trace,
            "stop_reason": stop_reason,
            "tool_call_count": tool_call_count,
            "metadata": sample.get("metadata", {}),
        }
    except Exception as exc:
        latency = time.perf_counter() - start
        return {
            "index": index,
            "reward": 0.0,
            "latency_s": latency,
            "error": str(exc),
            "metadata": sample.get("metadata", {}),
        }


async def _run(args) -> list[dict[str, Any]]:
    _prepare_imports(args.task)
    reward_func = _import_object(args.reward_path or _default_reward_path(args.task))
    close_reward_sessions = _import_object(
        "llm_gateway.rl.slime_train.tasks.retrieve_reward.frozen_qa_client.close_aiohttp_sessions"
    )
    samples = _load_samples(args.data, args.limit)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def guarded(i: int, sample: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    return await _run_one(i, sample, session, reward_func, args)

            return await asyncio.gather(*[guarded(i, sample) for i, sample in enumerate(samples)])
    finally:
        await close_reward_sessions()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test vLLM rollout + async reward without Ray")
    parser.add_argument("--task", choices=["ingest", "consolidate", "retrieve", "mixed"], required=True)
    parser.add_argument("--data", required=True, help="Slime-format JSONL data")
    parser.add_argument("--snapshot-data-root", default=None, help="Dataset root containing snapshots/")
    parser.add_argument("--api-url", default=os.environ.get("VLLM_API_URL") or os.environ.get("ROLLOUT_MODEL_URL") or "http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL") or os.environ.get("ROLLOUT_MODEL_NAME") or "default")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--frozen-model-url", default=None, help="OpenAI-compatible endpoint used by reward QA; defaults to --api-url")
    parser.add_argument("--frozen-model-name", default=None, help="Reward QA model name; defaults to --model")
    parser.add_argument("--reward-path", default=None, help="Override custom reward path")
    parser.add_argument("--task-version", default=os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2"), help="MemoryEnv task_version: atomic_code_t2 or t2_agent_loop")
    parser.add_argument("--apply-mode", choices=["tool_calls", "task_loop"], default=os.environ.get("MEMORY_RL_APPLY_MODE", "tool_calls"), help="How reward applies ingest/consolidate outputs")
    parser.add_argument("--tool-executor-path", default=None, help="Factory path: factory(task, metadata, args) -> async executor")
    parser.add_argument("--agentic", action="store_true", help="Run multi-turn tool-observation loop instead of single-turn plan")
    parser.set_defaults(native_tools=True)
    parser.add_argument("--native-tools", dest="native_tools", action="store_true", help="Send T3 OpenAI tool schemas and consume message.tool_calls (default)")
    parser.add_argument("--no-native-tools", dest="native_tools", action="store_false", help="Do not send OpenAI tools; expect JSON tool calls in message content")
    parser.add_argument("--max-agent-turns", type=int, default=4)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-reward", action="store_true")
    args = parser.parse_args()

    os.environ["MEMORY_RL_TASK_VERSION"] = args.task_version
    os.environ["MEMORY_RL_APPLY_MODE"] = args.apply_mode
    os.environ["MEMORY_RL_LLM_API_URL"] = args.api_url
    os.environ["MEMORY_RL_LLM_MODEL"] = args.model
    os.environ["MEMORY_RL_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["MEMORY_RL_LLM_MAX_TOKENS"] = str(args.max_tokens)
    if args.api_key:
        os.environ["MEMORY_RL_LLM_API_KEY"] = args.api_key
    if args.snapshot_data_root:
        os.environ["SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
        if args.task == "mixed":
            for _task_name in ("ingest", "consolidate", "retrieve"):
                os.environ[f"{_task_name.upper()}_SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
        else:
            os.environ[f"{args.task.upper()}_SNAPSHOT_DATA_ROOT"] = args.snapshot_data_root
    if args.frozen_model_url:
        os.environ["FROZEN_MODEL_URL"] = args.frozen_model_url
    else:
        os.environ.setdefault("FROZEN_MODEL_URL", args.api_url)
    if args.frozen_model_name:
        os.environ["FROZEN_MODEL_NAME"] = args.frozen_model_name
    else:
        os.environ.setdefault("FROZEN_MODEL_NAME", args.model)

    start = time.perf_counter()
    results = asyncio.run(_run(args))
    elapsed = time.perf_counter() - start

    rewards = [r["reward"] for r in results if isinstance(r.get("reward"), (int, float))]
    errors = [r for r in results if r.get("error")]
    summary = {
        "count": len(results),
        "errors": len(errors),
        "avg_reward": sum(rewards) / len(rewards) if rewards else None,
        "elapsed_s": elapsed,
        "throughput_samples_per_s": len(results) / elapsed if elapsed > 0 else 0.0,
        "agentic": args.agentic,
        "native_tools": args.native_tools,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for result in results:
        line = {
            "index": result.get("index"),
            "reward": result.get("reward"),
            "latency_s": result.get("latency_s"),
            "response_len": result.get("response_len"),
            "turns": len(result.get("agentic_trace") or []),
            "tool_calls": result.get("tool_call_count"),
            "stop_reason": result.get("stop_reason"),
            "error": result.get("error"),
        }
        print(json.dumps(line, ensure_ascii=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
