from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp

_HERE = Path(__file__).resolve()
for _parent in [_HERE, *_HERE.parents]:
    if _parent.name == "llm_gateway" and (_parent / "gateway").is_dir() and (_parent / "storage").is_dir():
        for _path in (str(_parent.parent), str(_parent)):
            if _path not in sys.path:
                sys.path.insert(0, _path)
        break

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths
from llm_gateway.rl.slime_train.memory_rl.response_parser import parse_tool_calls_response

ensure_workspace_paths(__file__)

from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.memory_rl.tool_executors import EnvToolExecutor
from utils.memory_llm_interface import LLMResponse, ToolCall

TOKENIZER = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _build_tool_executor(task: str, metadata: dict[str, Any], args):
    executor_path = os.environ.get("MEMORY_RL_TOOL_EXECUTOR_PATH") or getattr(args, "memory_rl_tool_executor_path", None)
    if executor_path:
        factory = _import_object(str(executor_path))
        executor = factory(task=task, metadata=metadata, args=args)
        if hasattr(executor, "__aenter__"):
            async with executor as active:
                yield active
        else:
            yield executor
        return

    async with EnvToolExecutor(task, metadata) as executor:
        yield executor


def _import_object(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


async def custom_generate(args, sample, sampling_params: dict, evaluation: bool = False) -> Any:
    """Slime ``--custom-generate-function-path`` entrypoint for agentic rollout.

    It keeps slime's default rollout pipeline but replaces the single generation
    step with a weakly-coupled model/tool loop:

    model generate -> parse tool calls -> execute tools -> append observations -> next turn
    """
    tokenizer = _get_tokenizer(args)
    prompt = getattr(sample, "prompt", "")
    metadata = getattr(sample, "metadata", {}) if isinstance(getattr(sample, "metadata", {}), dict) else {}
    metadata.setdefault("_memory_rl_split", "eval" if evaluation else "train")
    metadata["_memory_rl_evaluation"] = bool(evaluation)
    sample.metadata = metadata
    task = _task_from_metadata(metadata)
    max_turns = int(os.environ.get("MEMORY_RL_MAX_AGENT_TURNS", getattr(args, "memory_rl_max_agent_turns", 4)))

    # Resolve tools: metadata["tools"] may be a list of tool-name strings OR full OpenAI dicts.
    # Always resolve to full OpenAI tool schema dicts via get_tool_schemas(task).
    from llm_gateway.rl.slime_train.memory_rl.tool_schemas import get_tool_schemas
    raw_tools = metadata.get("tools") if isinstance(metadata.get("tools"), list) else None
    if raw_tools and all(isinstance(t, str) for t in raw_tools):
        # metadata only has tool names — fetch full schemas from registry
        tools = get_tool_schemas(task) or None
    elif raw_tools and all(isinstance(t, dict) for t in raw_tools):
        tools = raw_tools
    else:
        tools = get_tool_schemas(task) or None

    # Initialize tool schemas for type-aware parameter parsing
    from llm_gateway.rl.slime_train.memory_rl.response_parser import set_tool_schemas
    set_tool_schemas(tools)

    if _use_native_task_loop_rollout(task, metadata):
        return await _custom_generate_task_loop(args, sample, sampling_params, tokenizer, task, metadata)

    prompt_text = _render_prompt(prompt, tokenizer, tools=tools)

    prompt_token_ids = _encode(tokenizer, prompt_text)
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    all_calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    current_text = prompt_text
    final_text = ""
    final_finish_type = "stop"

    if task not in {"ingest", "consolidate"} or not metadata.get("snapshot_id"):
        output = await _post_generate(args, current_text, sampling_params)
        final_text = str(output.get("text", ""))
        final_finish_type = _finish_type(output)
        ids = _encode(tokenizer, final_text)
        response_token_ids.extend(ids)
        loss_mask.extend([1] * len(ids))
        _set_sample_output(sample, prompt_token_ids, response_token_ids, loss_mask, final_text, final_finish_type)
        return sample

    use_server_parser = _use_sglang_tool_parser() and tools

    async with _build_tool_executor(task, metadata, args) as executor:
        for turn in range(max_turns):
            if use_server_parser:
                # --- SGLang /generate + /parse_function_call path ---
                # Ensure skip_special_tokens=False so tool-call markers are preserved
                sp = dict(sampling_params)
                sp["skip_special_tokens"] = False
                output = await _post_generate(args, current_text, sp)
                assistant_text = str(output.get("text", ""))
                final_text = assistant_text
                final_finish_type = _finish_type(output)

                assistant_ids = _encode(tokenizer, assistant_text)
                response_token_ids.extend(assistant_ids)
                loss_mask.extend([1] * len(assistant_ids))
                current_text += assistant_text

                # Parse tool calls via /parse_function_call endpoint;
                # 若 router 不支持 (典型: slime 自带 sglang router 返回 404)，
                # _post_parse_function_call 会返回 None，此时降级到本地正则。
                parse_result = await _post_parse_function_call(args, assistant_text, tools)
                if parse_result is None:
                    calls, _ = parse_tool_calls_response(assistant_text)
                else:
                    calls = _parse_function_call_response(parse_result)
            else:
                # --- Local regex parsing path (original) ---
                output = await _post_generate(args, current_text, sampling_params)
                assistant_text = str(output.get("text", ""))
                final_text = assistant_text
                final_finish_type = _finish_type(output)

                assistant_ids = _encode(tokenizer, assistant_text)
                response_token_ids.extend(assistant_ids)
                loss_mask.extend([1] * len(assistant_ids))
                current_text += assistant_text

                calls, _ = parse_tool_calls_response(assistant_text)

            if not calls:
                trace.append({"turn": turn, "content": assistant_text, "tool_calls": [], "observations": []})
                break

            observations = await executor(calls)
            all_calls.extend(calls)
            trace.append({
                "turn": turn,
                "content": assistant_text,
                "tool_calls": calls,
                "observations": observations,
            })

            if any(call.get("tool") == "finish" for call in calls):
                break

            # Format observations in Qwen3.6 chat_template format:
            # <|im_end|>\n<|im_start|>user\n<tool_response>...\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n
            obs_content = _format_observations_for_prompt(observations)
            observation_text = f"<|im_end|>\n<|im_start|>user\n{obs_content}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            observation_ids = _encode(tokenizer, observation_text)
            response_token_ids.extend(observation_ids)
            loss_mask.extend([0] * len(observation_ids))
            current_text += observation_text

    stop_reason = "finish" if any(call.get("tool") == "finish" for call in all_calls) else "no_tool_calls" if not all_calls else "max_turns"
    response_payload = {
        "tool_calls": all_calls,
        "summary": _last_finish_summary(all_calls) or final_text[:1000],
        "agentic_trace": trace,
        "stop_reason": stop_reason,
    }
    sample_response = json.dumps(response_payload, ensure_ascii=False) if all_calls else final_text
    _set_sample_output(sample, prompt_token_ids, response_token_ids, loss_mask, sample_response, final_finish_type)
    return sample


async def _custom_generate_task_loop(args, sample, sampling_params: dict, tokenizer, task: str, metadata: dict[str, Any]) -> Any:
    """Run the bound llm_gateway task loop while recording Slime train tokens.

    This is the trainable path for ``MEMORY_RL_TASK_VERSION=t2_agent_loop``:
    task.run() still owns prompt construction and tool execution, but every
    internal ``llm.generate`` call is served by Slime's rollout router and its
    assistant tokens are appended with ``loss_mask=1``.
    """
    data_root = _snapshot_data_root(task)
    if not data_root:
        raise RuntimeError("SNAPSHOT_DATA_ROOT or <TASK>_SNAPSHOT_DATA_ROOT is required for task_loop custom_generate")

    adapter = SlimeTaskLoopLLM(args=args, sampling_params=sampling_params, tokenizer=tokenizer)
    session = SnapshotSession(
        data_root=data_root,
        enable_git=False,
        task_version=os.environ.get("MEMORY_RL_TASK_VERSION", "t2_agent_loop"),
    )
    loaded = await _load_snapshot_for_task_loop(session, metadata, adapter)
    step_index = metadata.get("step_index")
    post_consolidate_snapshot: dict[str, Any] | None = None
    try:
        if task == "ingest":
            ingest_extras: dict[str, Any] = {
                "session_id": metadata.get("session_id", ""),
                "session_time": metadata.get("session_time", ""),
                "pending_messages": metadata.get("pending_messages", []),
            }
            if step_index is not None:
                ingest_extras["ingest_number"] = int(step_index)
            step = await loaded.env.apply_ingest_tool_calls(
                [],
                session_time=metadata.get("session_time", ""),
                extras=ingest_extras,
            )
        elif task == "consolidate":
            consolidate_extras: dict[str, Any] = {
                "session_id": metadata.get("session_id", ""),
            }
            if step_index is not None:
                consolidate_extras["step_index"] = int(step_index)
            step = await loaded.env.apply_consolidate_tool_calls(
                [],
                extras=consolidate_extras,
            )
            try:
                post_consolidate_snapshot = await asyncio.to_thread(
                    session.save_env_snapshot,
                    loaded.env,
                    traj_id=str(metadata.get("traj_id") or loaded.traj_id),
                    source_snapshot_id=str(metadata.get("snapshot_id") or loaded.snapshot_id),
                    subdir=os.environ.get("MEMORY_RL_ROLLOUT_SNAPSHOT_DIR", "rollout_snapshots"),
                    meta={
                        "phase": "post_consolidate_rollout",
                        "session_id": metadata.get("session_id", ""),
                        "step_index": step_index,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - keep rollout usable; reward can fall back to replay
                logger.warning("failed to persist post-consolidate rollout snapshot: %s", exc)
        elif task == "retrieve":
            query = _retrieve_query_from_metadata(metadata)
            retrieve_extra: dict[str, Any] = {}
            if metadata.get("session_time"):
                retrieve_extra["session_time"] = metadata["session_time"]
            if step_index is not None:
                retrieve_extra["step_index"] = int(step_index)
            step = await loaded.env.step_query(
                query=query,
                session_id=str(metadata.get("session_id", "")),
                extra_payload=retrieve_extra or None,
            )
        else:
            raise RuntimeError(f"task_loop custom_generate does not support task={task!r}")
    finally:
        await _release_loaded_env(loaded)
        # 主动释放 session 的快照索引缓存以回收内存
        session._snapshot_index = None

    adapter.sync_observations_from_messages([])
    task_final_output = step.task_result.final_output or step.task_result.finish_summary or ""
    retrieved_context = _retrieved_context_from_step(step) if task == "retrieve" else ""
    stop_reason = "finish" if any(call.get("tool") in {"finish", "submit"} for call in adapter.all_calls) else "no_tool_calls" if not adapter.all_calls else "max_turns"
    response_payload = {
        "tool_calls": adapter.all_calls,
        "summary": _last_finish_summary(adapter.all_calls) or step.task_result.finish_summary or adapter.final_text[:1000],
        "agentic_trace": adapter.trace,
        "stop_reason": stop_reason,
        "retrieved_context": retrieved_context,
        "post_consolidate_snapshot": post_consolidate_snapshot,
        "task_result": {
            "task_name": step.task_name,
            "finish_reason": step.task_result.finish_reason,
            "finish_summary": step.task_result.finish_summary,
            "final_output": task_final_output,
            "error": step.task_result.error,
            "stats": step.task_result.stats,
        },
    }
    sample_response = json.dumps(response_payload, ensure_ascii=False)
    _set_sample_output(
        sample,
        adapter.prompt_token_ids,
        adapter.response_token_ids,
        adapter.loss_mask,
        sample_response,
        adapter.final_finish_type,
    )
    return sample


class SlimeTaskLoopLLM:
    """LLMInterface-compatible adapter backed by Slime/SGLang ``/generate``."""

    provider = "slime_router"
    model = "slime_policy"
    temperature = 0.0
    max_tokens = 4096

    def __init__(self, *, args, sampling_params: dict, tokenizer) -> None:
        self.args = args
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.prompt_token_ids: list[int] = []
        self.response_token_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.all_calls: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self._last_rendered = ""
        self._call_index = 0
        self.final_text = ""
        self.final_finish_type = "stop"

    async def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.sync_observations_from_messages(messages)
        from llm_gateway.rl.slime_train.memory_rl.response_parser import set_tool_schemas

        set_tool_schemas(tools)

        sampling_params = dict(self.sampling_params)
        if temperature is not None:
            sampling_params["temperature"] = temperature
        if max_tokens is not None:
            sampling_params["max_new_tokens"] = max_tokens
            sampling_params["max_tokens"] = max_tokens

        if _use_sglang_tool_parser() and tools:
            # --- SGLang /generate + /parse_function_call path ---
            rendered = _render_chat_for_task_loop(system, messages, tools, self.tokenizer)
            self._append_prompt_delta(rendered)

            # Generate with skip_special_tokens=False to preserve tool-call markers
            sp = dict(sampling_params)
            sp["skip_special_tokens"] = False
            output = await _post_generate(self.args, rendered, sp)
            text = str(output.get("text", ""))
            self.final_text = text
            self.final_finish_type = _finish_type(output)

            ids = _encode(self.tokenizer, text)
            self.response_token_ids.extend(ids)
            self.loss_mask.extend([1] * len(ids))
            self._last_rendered = rendered + text

            # Parse tool calls via /parse_function_call endpoint;
            # router 不支持时降级到本地正则，避免整条 trajectory 因 404 报废。
            parse_result = await _post_parse_function_call(self.args, text, tools)
            if parse_result is None:
                # Local regex fallback
                local_tool_calls = _parse_generated_tool_calls(text, call_offset=self._call_index)
                self._call_index += len(local_tool_calls)
                normalized_calls = [
                    {"tool": tc.name, "arguments": tc.arguments, "id": tc.id}
                    for tc in local_tool_calls
                ]
                self.all_calls.extend(normalized_calls)
                self.trace.append({
                    "turn": len(self.trace),
                    "content": text,
                    "tool_calls": normalized_calls,
                    "observations": [],
                })
                tool_calls = local_tool_calls
            else:
                normalized_calls = _parse_function_call_response(parse_result)
                self._call_index += len(normalized_calls)
                self.all_calls.extend(normalized_calls)
                self.trace.append({
                    "turn": len(self.trace),
                    "content": text,
                    "tool_calls": normalized_calls,
                    "observations": [],
                })

                # Convert to ToolCall objects for LLMResponse
                tool_calls = [
                    ToolCall(
                        id=str(c.get("id") or f"slime_task_loop_call_{self._call_index - len(normalized_calls) + i}"),
                        name=c["tool"],
                        arguments=c.get("arguments", {}),
                    )
                    for i, c in enumerate(normalized_calls)
                ]
        else:
            # --- Local regex parsing path (original) ---
            rendered = _render_chat_for_task_loop(system, messages, tools, self.tokenizer)
            self._append_prompt_delta(rendered)

            output = await _post_generate(self.args, rendered, sampling_params)
            text = str(output.get("text", ""))
            self.final_text = text
            self.final_finish_type = _finish_type(output)

            ids = _encode(self.tokenizer, text)
            self.response_token_ids.extend(ids)
            self.loss_mask.extend([1] * len(ids))
            self._last_rendered = rendered + text

            tool_calls = _parse_generated_tool_calls(text, call_offset=self._call_index)
            self._call_index += len(tool_calls)
            normalized_calls = [
                {"tool": tc.name, "arguments": tc.arguments, "id": tc.id}
                for tc in tool_calls
            ]
            self.all_calls.extend(normalized_calls)
            self.trace.append({
                "turn": len(self.trace),
                "content": text,
                "tool_calls": normalized_calls,
                "observations": [],
            })

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            model=str(getattr(self.args, "model", "slime_policy")),
            input_tokens=len(_encode(self.tokenizer, rendered)),
            output_tokens=len(ids),
        )

    def _append_prompt_delta(self, rendered: str) -> None:
        if not self.prompt_token_ids:
            self.prompt_token_ids.extend(_encode(self.tokenizer, rendered))
            self._last_rendered = rendered
            return
        prefix_len = _common_prefix_len(self._last_rendered, rendered)
        delta = rendered[prefix_len:]
        if delta:
            ids = _encode(self.tokenizer, delta)
            self.response_token_ids.extend(ids)
            self.loss_mask.extend([0] * len(ids))

    def sync_observations_from_messages(self, messages: list[dict[str, Any]]) -> None:
        observations_by_id: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if call_id:
                observations_by_id[call_id] = str(message.get("content") or "")
        if not observations_by_id:
            return
        for turn in self.trace:
            calls = turn.get("tool_calls", [])
            if not isinstance(calls, list):
                continue
            observations = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                if call_id in observations_by_id:
                    observations.append({
                        "id": call_id,
                        "tool": call.get("tool", ""),
                        "result": observations_by_id[call_id],
                    })
            if observations:
                turn["observations"] = observations


def _use_native_task_loop_rollout(task: str, metadata: dict[str, Any]) -> bool:
    # NOTE: an empty snapshot_id is the trajectory-first-step case after
    # build_mixed_data.shift_pre_ingest_snapshot. _load_snapshot_for_task_loop
    # below transparently constructs a freshly-reset MemoryEnv for it.
    if os.environ.get("MEMORY_RL_TRAIN_TASK_LOOP", "1") == "0":
        return False
    if os.environ.get("MEMORY_RL_APPLY_MODE", "").strip().lower() != "task_loop":
        return False
    task_version = os.environ.get("MEMORY_RL_TASK_VERSION", "")
    if task == "retrieve":
        # retrieve still needs a real snapshot to query against.
        if not metadata.get("snapshot_id"):
            return False
        return task_version in {"t2_agent_loop", "atomic_code_t2"}
    return task in {"ingest", "consolidate"} and task_version == "t2_agent_loop"


async def _load_snapshot_for_task_loop(session: SnapshotSession, metadata: dict[str, Any], llm: Any):
    from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
    return await acquire_loaded_env(metadata, session, llm=llm)


async def _release_loaded_env(loaded) -> None:
    import asyncio as _asyncio

    await _asyncio.to_thread(loaded.__exit__, None, None, None)


async def _post_generate(args, text: str, sampling_params: dict) -> dict[str, Any]:
    url = _router_generate_url(args)
    timeout = aiohttp.ClientTimeout(total=int(os.environ.get("MEMORY_RL_GENERATE_TIMEOUT", "300")))
    payload = {"text": text, "sampling_params": sampling_params}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"sglang generate http {resp.status}: {body[:500]}")
            return json.loads(body)


_PARSE_FN_CALL_DISABLED: bool = False  # 进程级软开关：一旦 router 不支持，自动降级到本地正则


async def _post_parse_function_call(
    args,
    text: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Call SGLang /parse_function_call endpoint to parse tool calls from
    generated text using the server-side qwen3_coder parser.

    Returns ``None`` (instead of raising) when the endpoint is unavailable
    (404 / 405 / network error / timeout) or when the parser has been
    permanently disabled for this process. Callers MUST treat ``None`` as
    "fall back to local regex parsing"; this avoids killing entire
    trajectories when the router (e.g. slime's sglang router) does not
    expose ``/parse_function_call``.

    Args:
        args: slime args (contains router address info)
        text: the raw model-generated text from /generate
        tools: tool definitions in OpenAI format

    Returns a dict with:
      - "normal_text": non-tool-call text content
      - "calls": list of {"name": str, "parameters": str(JSON)} or []
    Or ``None`` to signal "use local fallback".
    """
    global _PARSE_FN_CALL_DISABLED
    if _PARSE_FN_CALL_DISABLED:
        return None

    url = _router_parse_function_call_url(args)
    timeout = aiohttp.ClientTimeout(total=int(os.environ.get("MEMORY_RL_GENERATE_TIMEOUT", "300")))

    tool_call_parser = os.environ.get("MEMORY_RL_TOOL_CALL_PARSER", "qwen3_coder")
    payload: dict[str, Any] = {
        "text": text,
        "tool_call_parser": tool_call_parser,
        "tools": tools,
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status == 200:
                    return json.loads(body)
                # 4xx / 5xx：判定 endpoint 不可用，永久降级，避免每条 trajectory 都打一次然后失败
                if resp.status in (404, 405, 501):
                    _PARSE_FN_CALL_DISABLED = True
                    logger.warning(
                        "sglang /parse_function_call endpoint unavailable on router (%s, http=%s); "
                        "falling back to local regex parser for the rest of this run.",
                        url,
                        resp.status,
                    )
                    return None
                # 其他状态码当作瞬时错误，单次降级，不全局禁用
                logger.warning(
                    "sglang /parse_function_call returned http=%s: %s; falling back locally for this turn.",
                    resp.status,
                    body[:200],
                )
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        logger.warning(
            "sglang /parse_function_call network error (%s); falling back locally for this turn.",
            exc,
        )
        return None


def _parse_function_call_response(parse_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert /parse_function_call response into our internal
    ``[{"tool": name, "arguments": {...}}]`` format.

    The parse_function_call endpoint returns:
      {"normal_text": "...", "calls": [{"name": "fn_name", "parameters": "{...}"}]}
    """
    results: list[dict[str, Any]] = []
    calls = parse_result.get("calls") or []
    for call in calls:
        name = call.get("name", "")
        if not name:
            continue
        parameters_raw = call.get("parameters", "{}")
        if isinstance(parameters_raw, str):
            try:
                arguments = json.loads(parameters_raw)
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(parameters_raw, dict):
            arguments = parameters_raw
        else:
            arguments = {}
        results.append({"tool": name, "arguments": arguments})
    return results


def _use_sglang_tool_parser() -> bool:
    """Whether to use SGLang's server-side /parse_function_call endpoint.

    NOTE: slime's sglang router does NOT expose ``/parse_function_call`` (it is
    only available on the underlying single-engine SGLang HTTP server). Hitting
    the router with this path returns 404 and kills the entire trajectory, so
    we **default to the local regex parser** (Qwen3 Coder XML + JSON + GLM
    fallbacks in ``response_parser.py``).

    The env switch ``MEMORY_RL_USE_SGLANG_TOOL_PARSER=1`` is kept as an opt-in
    escape hatch (e.g. when pointing directly at a single sglang worker), but
    the default is now ``0``.
    """
    return os.environ.get("MEMORY_RL_USE_SGLANG_TOOL_PARSER", "0") == "1"


def _router_generate_url(args) -> str:
    if os.environ.get("SGLANG_ROUTER_URL"):
        return os.environ["SGLANG_ROUTER_URL"].rstrip("/") + "/generate"
    if getattr(args, "sglang_router_url", None):
        return str(args.sglang_router_url).rstrip("/") + "/generate"
    host = getattr(args, "sglang_router_ip", "127.0.0.1")
    port = getattr(args, "sglang_router_port", 30000)
    return f"http://{host}:{port}/generate"


def _router_parse_function_call_url(args) -> str:
    """Get the /parse_function_call endpoint URL from the SGLang router."""
    if os.environ.get("SGLANG_ROUTER_URL"):
        return os.environ["SGLANG_ROUTER_URL"].rstrip("/") + "/parse_function_call"
    if getattr(args, "sglang_router_url", None):
        return str(args.sglang_router_url).rstrip("/") + "/parse_function_call"
    host = getattr(args, "sglang_router_ip", "127.0.0.1")
    port = getattr(args, "sglang_router_port", 30000)
    return f"http://{host}:{port}/parse_function_call"


def _get_tokenizer(args):
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer

        checkpoint = (
            getattr(args, "hf_checkpoint", None)
            or getattr(args, "tokenizer", None)
            or getattr(args, "tokenizer_path", None)
            or os.environ.get("HF_CHECKPOINT")
        )
        if not checkpoint:
            raise RuntimeError("hf_checkpoint/tokenizer_path is required for custom_generate")
        TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return TOKENIZER


def _render_prompt(prompt: Any, tokenizer, tools: list[dict] | None = None) -> str:
    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt, tools=tools or None, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt if isinstance(m, dict))
    return str(prompt)


def _render_chat_for_task_loop(system: str, messages: list[dict[str, Any]], tools: list[dict] | None, tokenizer) -> str:
    chat = ([{"role": "system", "content": system}] if system else []) + [dict(m) for m in messages]
    try:
        return tokenizer.apply_chat_template(chat, tools=tools or None, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: manual Qwen3.6-style rendering
        tool_text = ""
        if tools:
            tool_text = (
                "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
                + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
                + "\n</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
                "value_1\n</parameter>\n</function>\n</tool_call>"
            )
        parts = []
        if system or tool_text:
            parts.append(f"<|im_start|>system\n{tool_text}\n\n{system}<|im_end|>" if tool_text else f"<|im_start|>system\n{system}<|im_end|>")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n<think>\n")
        return "\n".join(parts)


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _finish_type(output: dict[str, Any]) -> str:
    finish = output.get("meta_info", {}).get("finish_reason", {})
    if isinstance(finish, dict):
        return str(finish.get("type", "stop"))
    return str(finish or "stop")


def _set_sample_output(
    sample,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    response: str,
    finish_type: str,
) -> None:
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.response = response

    status_cls = getattr(sample, "Status", None)
    if status_cls is not None:
        if finish_type == "length":
            sample.status = status_cls.TRUNCATED
        elif finish_type == "abort":
            sample.status = status_cls.ABORTED
        else:
            sample.status = status_cls.COMPLETED
    else:
        sample.status = "TRUNCATED" if finish_type == "length" else "ABORTED" if finish_type == "abort" else "COMPLETED"


def _retrieve_query_from_metadata(metadata: dict[str, Any]) -> str:
    query = metadata.get("query") or metadata.get("probe_query")
    if query:
        return str(query)
    ground_truth = metadata.get("ground_truth", {})
    if isinstance(ground_truth, dict):
        return str(ground_truth.get("question") or "")
    return ""


def _retrieved_context_from_step(step: Any) -> str:
    stats = getattr(step.task_result, "stats", None)
    if isinstance(stats, dict):
        for key in ("retrieved_context", "query_memory"):
            value = stats.get(key)
            if value:
                return str(value)
    return str(step.task_result.final_output or step.task_result.finish_summary or "")


def _snapshot_data_root(task: str) -> str | None:
    return os.environ.get(f"{task.upper()}_SNAPSHOT_DATA_ROOT") or os.environ.get("SNAPSHOT_DATA_ROOT")


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _parse_generated_tool_calls(text: str, *, call_offset: int = 0) -> list[ToolCall]:
    calls, _ = parse_tool_calls_response(text)
    if not calls:
        calls = _parse_glm_tool_tags(text)
    parsed: list[ToolCall] = []
    for i, call in enumerate(calls):
        name = str(call.get("tool") or call.get("name") or "")
        if not name:
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        parsed.append(ToolCall(
            id=str(call.get("id") or f"slime_task_loop_call_{call_offset + i}"),
            name=name,
            arguments=arguments,
        ))
    return parsed


def _parse_glm_tool_tags(text: str) -> list[dict[str, Any]]:
    """Parse GLM-style ``<tool_call>name<arg_key>...`` text fallback."""
    results: list[dict[str, Any]] = []
    for block in re.finditer(r"<tool_call>\s*([A-Za-z_][\w.-]*)(.*?)(?=<tool_call>|$)", text or "", re.DOTALL):
        name = block.group(1).strip()
        body = block.group(2)
        args: dict[str, Any] = {}
        for m in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.DOTALL):
            key = m.group(1).strip()
            value = m.group(2).strip()
            args[key] = _maybe_json(value)
        results.append({"tool": name, "arguments": args})
    return results


def _maybe_json(value: str) -> Any:
    if not value:
        return value
    if value[0] in "[{\"" or value in {"true", "false", "null"}:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _task_from_metadata(metadata: dict[str, Any]) -> str:
    task = str(metadata.get("task", "")).lower()
    if task.startswith("ingest"):
        return "ingest"
    if task.startswith("consolidate") or task.startswith("evolve"):
        return "consolidate"
    if task.startswith("retrieve") or task.startswith("query") or task.startswith("consume"):
        return "retrieve"
    if isinstance(metadata.get("ground_truth"), dict):
        return "retrieve"
    return task


def _format_observations_for_prompt(observations: list[dict[str, Any]]) -> str:
    """Format tool observations as Qwen3 chat_template <tool_response> blocks.

    This matches the Qwen3.6 chat_template format where tool responses are
    wrapped in ``<tool_response>...</tool_response>`` XML tags inside a user message.
    """
    parts: list[str] = []
    for obs in observations:
        result = str(obs.get("result", ""))
        if len(result) > 2000:
            result = result[:2000] + "\n... (truncated)"
        parts.append(f"<tool_response>\n{result}\n</tool_response>")
    return "\n".join(parts)


def _last_finish_summary(calls: list[dict[str, Any]]) -> str:
    for call in reversed(calls):
        if call.get("tool") == "finish":
            args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
            return str(args.get("summary") or args.get("changes_summary") or "")
    return ""
