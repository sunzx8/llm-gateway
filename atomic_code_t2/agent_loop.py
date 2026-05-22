"""Atomic_Code_T2 方案的通用 agent loop 主循环。

设计要点：
- 输入：system prompt、初始 messages、tool registry、llm interface、max_turns
- 输出：AgentLoopResult，含 finish_summary、turn 数、所有 tool calls 的轨迹
- 退出条件（任一）：
    1. 模型返回包含 `finish` 的 tool_call → 提取 summary，正常结束
    2. 模型返回纯文本无 tool_calls → 视为完成，content 作为 summary
    3. 到达 max_turns → 强制结束，记录警告
- 工具执行：所有非 finish 的 tool_calls 顺序执行（同一轮内可并发，但内存版 store
  本身非线程安全，先用顺序执行，简单可靠）

注意：
- 本模块不依赖 BaseContextTask，独立可测；调用方（IngestContextAtomicCodeT2Task
  等）会把 LLM 调用统计接到自己的 stats 上。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import logger.logger as logger
from utils.memory_llm_interface import LLMInterface, LLMResponse

from .tool.registry import ToolRegistry

# Opik 追踪装饰器（不可用时退化为 no-op）
try:
    import opik
    _opik_track = opik.track
except Exception:
    def _opik_track(*args, **kwargs):
        """no-op fallback"""
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator


@dataclass
class ToolCallTrace:
    """单次 tool call 的轨迹记录（用于调试/可视化）。"""

    turn: int
    name: str
    arguments: dict[str, Any]
    result_content: str = ""
    is_error: bool = False


@dataclass
class AgentLoopResult:
    """agent loop 的最终结果。"""

    finish_summary: str = ""
    turns_used: int = 0
    finish_reason: str = ""  # "finish_tool" | "no_tool_calls" | "max_turns"
    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    final_messages: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 日志辅助
# ---------------------------------------------------------------------------

_PREVIEW_LIMIT = 200


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    """把可能很长的文本压成一行预览，便于日志阅读。"""
    if text is None:
        return ""
    s = str(text).replace("\n", " ⏎ ")
    if len(s) > limit:
        s = s[:limit] + f"...(+{len(text) - limit})"
    return s


def _args_preview(arguments: dict[str, Any], limit: int = _PREVIEW_LIMIT) -> str:
    """工具参数压成 JSON 单行预览。"""
    try:
        s = json.dumps(arguments, ensure_ascii=False, default=str)
    except Exception:
        s = str(arguments)
    return _preview(s, limit)


def _tool_calls_summary(tool_calls: list) -> str:
    """汇总一轮的所有 tool_call 名称，例如 `ls x1, vec_search x2, write_file x1`。"""
    names = [getattr(tc, "name", "?") for tc in tool_calls]
    counter = Counter(names)
    return ", ".join(f"{name} x{cnt}" for name, cnt in counter.items())


def _backend_breakdown(traces: list[ToolCallTrace]) -> str:
    """按 fs / vec / graph / control 分桶统计调用次数，对治"光做 fs"问题。"""
    fs_names = {"ls", "read_file", "write_file", "edit_file", "grep"}
    vec_names = {"vec_search", "vec_write"}
    graph_names = {"graph_search", "graph_write"}
    buckets = {"fs": 0, "vec": 0, "graph": 0, "control": 0, "other": 0}
    by_name: Counter = Counter()
    for t in traces:
        by_name[t.name] += 1
        if t.name in fs_names:
            buckets["fs"] += 1
        elif t.name in vec_names:
            buckets["vec"] += 1
        elif t.name in graph_names:
            buckets["graph"] += 1
        elif t.name == "finish":
            buckets["control"] += 1
        else:
            buckets["other"] += 1
    by_name_str = ", ".join(f"{n}={c}" for n, c in sorted(by_name.items()))
    return (
        f"fs={buckets['fs']} vec={buckets['vec']} graph={buckets['graph']} "
        f"control={buckets['control']} | {by_name_str}"
    )


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


@_opik_track(name="agent_loop")
async def run_agent_loop(
    *,
    llm: LLMInterface,
    system_prompt: str,
    initial_messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_turns: int = 15,
    label: str = "agent_loop",
) -> AgentLoopResult:
    """运行 agent loop 直到 finish / 无工具调用 / 到达上限。

    Args:
        llm: LLM 接口实例。
        system_prompt: 系统提示词。
        initial_messages: 起始 messages（一般是单条 user 消息）。
        tools: 已注册好的工具集。
        max_turns: 最大轮次。
        label: 日志前缀，便于在日志里区分 ingest / retrieve。

    Returns:
        AgentLoopResult。
    """
    messages: list[dict[str, Any]] = list(initial_messages)
    tool_defs = tools.get_definitions()
    tool_traces: list[ToolCallTrace] = []
    finish_summary = ""
    finish_reason = "max_turns"
    turns_used = 0

    # 起始上下文摘要：让你一眼看出本次 ingest 的输入规模
    initial_user_preview = ""
    if initial_messages and initial_messages[0].get("role") == "user":
        initial_user_preview = _preview(initial_messages[0].get("content", ""), 300)
    logger.info(
        "[%s] start (max_turns=%d, tools=%d): %s",
        label, max_turns, len(tool_defs), initial_user_preview,
    )

    for turn in range(1, max_turns + 1):
        turns_used = turn
        logger.info(
            "[%s] turn %d/%d: calling LLM (msgs=%d, tools=%d)",
            label, turn, max_turns, len(messages), len(tool_defs),
        )

        response: LLMResponse = await llm.generate(
            system=system_prompt,
            messages=messages,
            tools=tool_defs,
        )

        # LLM 响应概览：reasoning / content / tool_calls
        reasoning = getattr(response, "reasoning_content", "") or ""
        if reasoning:
            logger.info("[%s] turn %d reasoning: %s", label, turn, _preview(reasoning, 300))
        content = response.content or ""
        if content:
            logger.info("[%s] turn %d content : %s", label, turn, _preview(content, 300))
        if response.tool_calls:
            logger.info(
                "[%s] turn %d tool_calls (%d): %s",
                label, turn, len(response.tool_calls), _tool_calls_summary(response.tool_calls),
            )

        # 没有 tool_calls：模型返回纯文本，视为完成
        if not response.tool_calls:
            finish_summary = content
            finish_reason = "no_tool_calls"
            logger.info(
                "[%s] turn %d ended without tool_calls (content_len=%d)",
                label, turn, len(finish_summary),
            )
            messages.append(response.to_message())
            break

        # 有 tool_calls：先把 assistant 消息追加进 messages
        messages.append(response.to_message())

        # 处理 finish + 其它工具
        finished = False
        for tc in response.tool_calls:
            if tc.name == "finish":
                finish_summary = tc.arguments.get("summary", "") or ""
                finish_reason = "finish_tool"
                logger.info(
                    "[%s] turn %d call finish args=%s",
                    label, turn, _args_preview(dict(tc.arguments)),
                )
                tool_traces.append(ToolCallTrace(
                    turn=turn,
                    name="finish",
                    arguments=dict(tc.arguments),
                    result_content=f"finish: {finish_summary}",
                    is_error=False,
                ))
                # 给 finish 也回灌一条 tool message，保持 OpenAI 协议合规
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"finish acknowledged: {finish_summary}",
                })
                finished = True
                continue

            # 普通工具：先打印输入参数，再执行，最后打印结果
            logger.info(
                "[%s] turn %d call %s args=%s",
                label, turn, tc.name, _args_preview(dict(tc.arguments)),
            )
            result = await tools.execute(tc.name, dict(tc.arguments))
            tool_traces.append(ToolCallTrace(
                turn=turn,
                name=tc.name,
                arguments=dict(tc.arguments),
                result_content=result.content,
                is_error=result.is_error,
            ))
            messages.append(result.to_tool_message(tc.id))
            logger.info(
                "[%s] turn %d done %s %s-> %s",
                label, turn, tc.name,
                "ERR " if result.is_error else "",
                _preview(result.content),
            )

        if finished:
            logger.info(
                "[%s] turn %d ended via finish tool (summary_len=%d)",
                label, turn, len(finish_summary),
            )
            break
    else:
        # for 循环正常结束（没 break）= 到达 max_turns
        logger.warning("[%s] reached max_turns=%d without finish", label, max_turns)

    # 收尾：按后端分桶统计本次 loop 的工具调用次数
    logger.info(
        "[%s] DONE reason=%s turns=%d tool_calls=%d | %s",
        label, finish_reason, turns_used, len(tool_traces), _backend_breakdown(tool_traces),
    )
    logger.info("[%s] summary: %s", label, _preview(finish_summary, 300))

    return AgentLoopResult(
        finish_summary=finish_summary,
        turns_used=turns_used,
        finish_reason=finish_reason,
        tool_traces=tool_traces,
        final_messages=messages,
    )
