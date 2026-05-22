from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from llm_gateway.rl.slime_train.memory_rl.response_parser import parse_tool_calls_response

ModelCall = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]
ToolExecutor = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


def extract_tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized tool calls from OpenAI message or JSON content."""
    native_calls = message.get("tool_calls")
    if isinstance(native_calls, list) and native_calls:
        calls: list[dict[str, Any]] = []
        for call in native_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                calls.append({
                    "tool": function.get("name", ""),
                    "arguments": function.get("arguments", {}),
                    "id": call.get("id", ""),
                })
            elif isinstance(call, dict):
                calls.append(call)
        parsed, _ = parse_tool_calls_response(json.dumps({"tool_calls": calls}, ensure_ascii=False))
        return parsed

    content = message.get("content", "") or ""
    calls, _ = parse_tool_calls_response(content)
    return calls


def has_finish_call(calls: list[dict[str, Any]]) -> bool:
    return any(call.get("tool") == "finish" for call in calls)


async def run_agentic_rollout(
    initial_messages: list[dict[str, Any]],
    *,
    call_model: ModelCall,
    execute_tool_calls: ToolExecutor,
    max_turns: int = 4,
    use_native_tool_messages: bool = False,
) -> dict[str, Any]:
    """Run a weakly-coupled agent loop.

    The loop is task-agnostic: callers provide a model function and a tool-call
    executor. Switching task execution semantics only requires passing a new
    ``execute_tool_calls`` function.
    """
    messages = [dict(m) for m in initial_messages]
    all_calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    final_content = ""
    stop_reason = "max_turns"

    for turn in range(max_turns):
        message = await call_model(messages)
        content = message.get("content", "") or ""
        final_content = content
        calls = extract_tool_calls_from_message(message)

        assistant_message = {"role": "assistant", "content": content}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        messages.append(assistant_message)

        if not calls:
            stop_reason = "no_tool_calls"
            trace.append({
                "turn": turn,
                "content": content,
                "tool_calls": [],
                "observations": [],
                "stop_reason": stop_reason,
            })
            break

        observations = await execute_tool_calls(calls)
        all_calls.extend(calls)
        trace.append({
            "turn": turn,
            "content": content,
            "tool_calls": calls,
            "observations": observations,
        })

        if use_native_tool_messages and message.get("tool_calls"):
            for i, obs in enumerate(observations):
                call_id = calls[i].get("id") if i < len(calls) else ""
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id or f"call_{turn}_{i}",
                    "content": _format_single_observation(obs),
                })
        else:
            messages.append({
                "role": "user",
                "content": _format_observations_for_next_turn(observations),
            })

        if has_finish_call(calls):
            stop_reason = "finish"
            trace[-1]["stop_reason"] = stop_reason
            break

    response_payload = {
        "tool_calls": all_calls,
        "summary": _last_finish_summary(all_calls) or final_content[:1000],
        "agentic_trace": trace,
        "stop_reason": stop_reason,
    }
    return {
        "response": json.dumps(response_payload, ensure_ascii=False),
        "messages": messages,
        "trace": trace,
        "tool_calls": all_calls,
        "final_content": final_content,
        "stop_reason": stop_reason,
    }


def _format_single_observation(obs: dict[str, Any]) -> str:
    result = str(obs.get("result", ""))
    if len(result) > 4000:
        result = result[:4000] + "\n... (truncated)"
    return result


def _format_observations_for_next_turn(observations: list[dict[str, Any]]) -> str:
    lines = ["Tool observations from previous turn:"]
    for i, obs in enumerate(observations, 1):
        tool = obs.get("tool", "")
        result = str(obs.get("result", ""))
        if len(result) > 2000:
            result = result[:2000] + "\n... (truncated)"
        lines.append(f"{i}. {tool}: {result}")
    lines.append("Continue with more tool calls if needed, otherwise call finish.")
    return "\n".join(lines)


def _last_finish_summary(calls: list[dict[str, Any]]) -> str:
    for call in reversed(calls):
        if call.get("tool") == "finish":
            args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
            return str(args.get("summary", ""))
    return ""
