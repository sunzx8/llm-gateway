"""Atomic T2 Agent Loop — agent loop 模式的多后端记忆任务。

来源：``memory-ai-agent-workspace`` 的 ``dev-0421-t3`` 分支。原本是 T3 系统的 AGENT/
AGENT_LOOP 模式，迁移到 llm_gateway 时统一对齐 :class:`BaseContextTask` 接口、
重命名为 "atomic t2 agent loop"（保留 atomic 词缀强调原子写入；T2 强调多后端
FS+Vec+Graph；agent loop 强调由 LLM 自主决定何时停止的 budget-constrained loop）。

接口契约：

- 全部继承 :class:`context_task.base_context_task.BaseContextTask`
- ``__init__(llm, fs_store, vec_store, graph_store, **kwargs)``
- 子类只覆写 ``async def run(user_id, session_id, **kwargs) -> None``
- 调用方走 ``await task.execute(user_id=..., session_id=..., **kwargs)`` 拿 dict
- agent loop 内部 LLM 调用一律走 ``self.llm_generate_with_stat()``（自动统计）
- agent loop 内部工具调用走 ``self.tool_with_stat()``（自动统计）

仅保留 agent loop 模式 — 原 T3 系统中的 fast/slow 两轮、ENGINEERING/LLM 单次改写、
Plan→Submit 两轮等非 agent loop 路径已物理移除。
"""

from llm_gateway.atomic_t2_agent_loop.consolidate_task import (
    CONSOLIDATE_T2_AGENT_SYSTEM_PROMPT,
    CONSOLIDATE_T2_AGENT_TOOLS,
    CONSOLIDATE_T2_AGENT_USER_TEMPLATE,
    ConsolidateT2AgentLoopTask,
    ConsolidateTrigger,
)
from llm_gateway.atomic_t2_agent_loop.ingest_task import (
    INGEST_T2_AGENT_SYSTEM_PROMPT,
    INGEST_T2_AGENT_TOOLS,
    INGEST_T2_AGENT_USER_TEMPLATE,
    IngestT2AgentLoopTask,
)
from llm_gateway.atomic_t2_agent_loop.retrieve_task import (
    RETRIEVE_T2_AGENT_SYSTEM_PROMPT,
    RETRIEVE_T2_AGENT_TOOLS,
    RETRIEVE_T2_AGENT_USER_TEMPLATE,
    RetrieveT2AgentLoopTask,
)

__all__ = [
    # ingest
    "IngestT2AgentLoopTask",
    "INGEST_T2_AGENT_SYSTEM_PROMPT",
    "INGEST_T2_AGENT_USER_TEMPLATE",
    "INGEST_T2_AGENT_TOOLS",
    # consolidate
    "ConsolidateT2AgentLoopTask",
    "ConsolidateTrigger",
    "CONSOLIDATE_T2_AGENT_SYSTEM_PROMPT",
    "CONSOLIDATE_T2_AGENT_USER_TEMPLATE",
    "CONSOLIDATE_T2_AGENT_TOOLS",
    # retrieve
    "RetrieveT2AgentLoopTask",
    "RETRIEVE_T2_AGENT_SYSTEM_PROMPT",
    "RETRIEVE_T2_AGENT_USER_TEMPLATE",
    "RETRIEVE_T2_AGENT_TOOLS",
]
