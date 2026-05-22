"""本地化的 Event/TaskResult/ToolCall dataclass。

历史上老 rl_env 直连 ``agent_memory.core.models`` 中的 ``Event/EventType/
TaskResult/ToolCall``，迁移到 llm_gateway 后 ``agent_memory`` 不再可用，
而 llm_gateway 自己的 task 实现也不接受 Event 包装。所以我们在 rl_env 内
本地定义这些轻量结构体：

- ``Event/EventType``：作为 ``MemoryEnv._step_generic`` → ``TaskDispatcher``
  之间的中间结构，封装 ``user_id/session_id/messages/query`` 等参数。
  同时也兼容 dev-0421-t3 分支的 T3 task（这些 task 通过 ``execute(event)``
  接收 ``Event`` 对象并访问 ``event.payload``/``event.session_id``/
  ``event.timestamp``）。
- ``TaskResult``：封装 task 实际产出。同时支持两套字段命名：
    * ``finish_summary``/``final_output``/``finish_reason``/``metadata``：
      老 rl_env 的命名，``MemoryEnv.step_*`` 直接消费
    * ``reasoning``/``retrieved_context``/``finish_result``/``stats``：
      dev-0421-t3 T3 task 的命名，由 task 自己填写
  对外契约保持稳定：上层既可以读 ``finish_summary``，也可以读
  ``finish_result`` —— 二者本质同义。
- ``ToolCall``：兼容 OpenAI 原生 tool call 与 T3 风格 ``apply_*_tool_calls``
  策略 rollout 路径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """三类 RL step 对应的事件类型。"""

    MESSAGE = "message"  # ingest
    MEMORY_CONSOLIDATION = "memory_consolidation"  # consolidate / evolve
    MEMORY_QUERY = "memory_query"  # retrieve / query


@dataclass
class Event:
    """rl_env 内部使用的轻量事件结构。

    ``payload`` 包含：

    - ingest: ``user_id`` / ``session_id`` / ``content`` / ``messages`` /
      ``ingest_number`` / ``message_count`` / ``is_streaming_ingest`` /
      ``trajectory_id`` (可选)
    - consolidate: ``user_id`` / ``session_id``
    - query: ``user_id`` / ``session_id`` / ``query`` / ``messages`` (可选) /
      ``latest_memory`` (可选)
    """

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def user_id(self) -> str | None:
        return self.payload.get("user_id")

    @property
    def session_id(self) -> str | None:
        return self.payload.get("session_id")


@dataclass
class ToolCall:
    """OpenAI 风格的 tool call 规范化结构。

    与 ``llm_gateway.utils.memory_llm_interface.ToolCall`` 字段对齐，可用于
    T3 task ``execute()`` 中的 LLM 响应 tool calls 直接消费。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @classmethod
    def from_openai(cls, tc) -> "ToolCall":
        """从 OpenAI 原生 tool_call 对象构造。

        与 ``llm_gateway.utils.memory_llm_interface.ToolCall.from_openai``
        行为完全一致：兼容 OpenAI 标准（arguments 为 JSON 字符串）和
        Qwen3.5/vLLM（arguments 已是 dict）两种格式。
        """
        args: dict[str, Any] = {}
        if getattr(tc.function, "arguments", None):
            raw_args = tc.function.arguments
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {"command": raw_args}
        return cls(id=tc.id, name=tc.function.name, arguments=args, raw=tc)


@dataclass
class TaskResult:
    """统一的 task 输出。

    Attributes（老命名 — MemoryEnv.step_* 消费）:
        task_name: 产出该结果的 task 名称。
        finish_summary: 摘要 / finish 描述。
        final_output: 检索类 task 的最终上下文输出（query_memory）。
        finish_reason: ``"finish"`` / ``"max_turns"`` / ``"error"`` / ``""``。
        error: 异常字符串。无错时为空串。
        metadata: 其他元数据。
        stats: 工具轨迹与统计。

    Attributes（dev-0421-t3 命名 — T3 task 自填）:
        reasoning: 模型在所有步骤上的推理摘要。
        retrieved_context: 检索类 task 输出的上下文（与 ``final_output`` 同义）。
        finish_result: finish 工具调用的结果文本（与 ``finish_summary`` 同义）。

    访问任一组命名都会得到一致结果（通过 ``__post_init__`` 同步）。
    """

    task_name: str
    finish_summary: str = ""
    final_output: str = ""
    finish_reason: str = ""
    error: str | None = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    # ---- dev-0421-t3 命名（同义别名） ----
    reasoning: str = ""
    retrieved_context: str = ""
    finish_result: str = ""

    def __post_init__(self) -> None:
        # 双向同步：让两组字段名指向同一逻辑值
        if self.finish_result and not self.finish_summary:
            self.finish_summary = self.finish_result
        elif self.finish_summary and not self.finish_result:
            self.finish_result = self.finish_summary
        if self.retrieved_context and not self.final_output:
            self.final_output = self.retrieved_context
        elif self.final_output and not self.retrieved_context:
            self.retrieved_context = self.final_output
        # error 兼容：T3 task 可能传 None 表示成功
        if self.error is None:
            self.error = ""


__all__ = [
    "Event",
    "EventType",
    "ToolCall",
    "TaskResult",
]
