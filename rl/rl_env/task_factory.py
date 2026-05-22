"""多版本 Task Dispatcher 工厂。

把老 `MemoryEnv._step_generic(task=..., event=Event(...))` 风格统一翻译为
llm_gateway 5 套真实 task 实现的实际调用签名（``await task.execute(user_id, session_id, **kwargs)``），
并把返回的 ``dict`` 反向映射成 :class:`llm_gateway.rl.rl_env._models.TaskResult`。

支持 ``task_version``：

- ``"atomic_code_t2"``：``llm_gateway.atomic_code_t2.{ingest_task,consolidate_task,retrieve_task}`` 中的
  ``IngestContextAtomicCodeT2Task`` / ``ConsolidateContextAtomicCodeT2Task`` /
  ``RetrieveContextAtomicCodeT2Task``（默认）
- ``"code_t2"``：``llm_gateway.context_task.{ingest_context_code_task, consolidate_context_task,
  retrieve_context_code_task}`` 中的 ``IngestContextCodeTask`` /
  ``ConsolidateContextTask`` / ``RetrieveContextCodeTask``
- ``"multi_code_t2"``：``IngestContextMultiCodeTask`` / ``ConsolidateContextTask`` /
  ``RetrieveContextMultiCodeTask``
- ``"t2"``：``IngestContextTask`` / ``ConsolidateContextTask`` / ``RetrieveContextTask``
- ``"t2_agent_loop"``：``llm_gateway.atomic_t2_agent_loop.{ingest_task,consolidate_task,retrieve_task}``
  中的 ``IngestT2AgentLoopTask`` / ``ConsolidateT2AgentLoopTask`` /
  ``RetrieveT2AgentLoopTask``（来自 ``dev-0421-t3`` 分支的 agent loop 模式 T3 任务，
  统一对齐 :class:`BaseContextTask` 接口；max_turns 默认 5/50/5）

工厂在 ``MemoryEnv.reset()`` 时一次性构造 ``TaskTriad``，避免每个 step 反射 import；
真正进入 ``MemoryEnv.step_*`` 的热路径完全是同步派发。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_gateway.rl.rl_env._models import Event, EventType, TaskResult

if TYPE_CHECKING:
    from context_task.base_context_task import BaseContextTask
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase, VectorStoreBase
    from utils.memory_llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 三件套：一个 dispatcher 内同时持有 ingest / consolidate / retrieve 三个 task
# ---------------------------------------------------------------------------


@dataclass
class TaskTriad:
    """RL env 一次 reset 后绑定的 (ingest, consolidate, retrieve) 三件套。

    三个 task 共享同一组 stores / llm，state 是可见的。task_version 决定
    具体类型；外部调用走 dispatcher 适配层即可，无需关心实际类型。
    """

    ingest: "BaseContextTask"
    consolidate: "BaseContextTask"
    retrieve: "BaseContextTask"
    task_version: str

    def names(self) -> tuple[str, str, str]:
        return (
            getattr(self.ingest, "task_name", "ingest_unknown"),
            getattr(self.consolidate, "task_name", "consolidate_unknown"),
            getattr(self.retrieve, "task_name", "retrieve_unknown"),
        )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


_VALID_VERSIONS = {"atomic_code_t2", "code_t2", "multi_code_t2", "t2", "t2_agent_loop"}


def build_task_triad(
    task_version: str,
    *,
    llm: "LLMInterface",
    fs: "FileSystemStore",
    vec: "VectorStoreBase",
    graph: "GraphStoreBase",
    max_turns: int | None = None,
) -> TaskTriad:
    """根据 ``task_version`` 构造三件套 task。

    Args:
        task_version: ``"atomic_code_t2"`` / ``"code_t2"`` /
            ``"multi_code_t2"`` / ``"t2"``。
        llm/fs/vec/graph: 共享给三个 task 的依赖。
        max_turns: 若提供，覆盖 task 的 ``max_turns`` 属性。
    """
    if task_version not in _VALID_VERSIONS:
        raise ValueError(
            f"unknown task_version={task_version!r}; expected one of {sorted(_VALID_VERSIONS)}"
        )

    if task_version == "atomic_code_t2":
        from atomic_code_t2.consolidate_task import ConsolidateContextAtomicCodeT2Task
        from atomic_code_t2.ingest_task import IngestContextAtomicCodeT2Task
        from atomic_code_t2.retrieve_task import RetrieveContextAtomicCodeT2Task

        ingest_kwargs: dict[str, Any] = {}
        if max_turns is not None:
            ingest_kwargs["max_turns"] = max_turns
        ingest_task = IngestContextAtomicCodeT2Task(llm, fs, vec, graph, **ingest_kwargs)
        consolidate_task = ConsolidateContextAtomicCodeT2Task(llm, fs, vec, graph)
        retrieve_task = RetrieveContextAtomicCodeT2Task(llm, fs, vec, graph)
    elif task_version == "code_t2":
        from context_task.consolidate_context_task import ConsolidateContextTask
        from context_task.ingest_context_code_task import IngestContextCodeTask
        from context_task.retrieve_context_code_task import RetrieveContextCodeTask

        ingest_task = IngestContextCodeTask(llm, fs, vec, graph)
        consolidate_task = ConsolidateContextTask(llm, fs, vec, graph)
        retrieve_task = RetrieveContextCodeTask(llm, fs, vec, graph)
    elif task_version == "multi_code_t2":
        from context_task.consolidate_context_task import ConsolidateContextTask
        from context_task.ingest_context_multi_code_task import IngestContextMultiCodeTask
        from context_task.retrieve_context_multi_code_task import RetrieveContextMultiCodeTask

        ingest_task = IngestContextMultiCodeTask(llm, fs, vec, graph)
        consolidate_task = ConsolidateContextTask(llm, fs, vec, graph)
        retrieve_task = RetrieveContextMultiCodeTask(llm, fs, vec, graph)
    elif task_version == "t2_agent_loop":
        # 来自 dev-0421-t3 分支的 agent loop 模式 T3 任务，统一适配到 BaseContextTask
        from llm_gateway.atomic_t2_agent_loop.consolidate_task import (
            ConsolidateT2AgentLoopTask,
        )
        from llm_gateway.atomic_t2_agent_loop.ingest_task import IngestT2AgentLoopTask
        from llm_gateway.atomic_t2_agent_loop.retrieve_task import RetrieveT2AgentLoopTask

        ingest_kwargs: dict[str, Any] = {}
        if max_turns is not None:
            ingest_kwargs["max_turns"] = max_turns
        ingest_task = IngestT2AgentLoopTask(llm, fs, vec, graph, **ingest_kwargs)
        consolidate_task = ConsolidateT2AgentLoopTask(llm, fs, vec, graph)
        retrieve_task = RetrieveT2AgentLoopTask(llm, fs, vec, graph)
    else:  # task_version == "t2"
        from context_task.consolidate_context_task import ConsolidateContextTask
        from context_task.ingest_context_task import IngestContextTask
        from context_task.retrieve_context_task import RetrieveContextTask

        ingest_task = IngestContextTask(llm, fs, vec, graph)
        consolidate_task = ConsolidateContextTask(llm, fs, vec, graph)
        retrieve_task = RetrieveContextTask(llm, fs, vec, graph)

    if max_turns is not None:
        # IngestContextTask / RetrieveContextTask 都有 max_turns 属性，
        # 但 ConsolidateContextAtomicCodeT2Task 不一定有；用 setattr 兜底。
        for t in (ingest_task, consolidate_task, retrieve_task):
            if hasattr(t, "max_turns"):
                try:
                    setattr(t, "max_turns", max_turns)
                except Exception:  # pragma: no cover - defensive
                    pass

    triad = TaskTriad(
        ingest=ingest_task,
        consolidate=consolidate_task,
        retrieve=retrieve_task,
        task_version=task_version,
    )
    logger.info(
        "build_task_triad: task_version=%s names=%s",
        task_version, triad.names(),
    )
    return triad


# ---------------------------------------------------------------------------
# Event → task.execute(...) 适配
# ---------------------------------------------------------------------------


async def dispatch_event(
    triad: TaskTriad,
    event: Event,
) -> TaskResult:
    """把内部 Event 翻译成对应 task 的 ``execute()`` 调用，再把返回 dict
    反向映射为统一的 :class:`TaskResult`。

    rl_env 中所有 ``step_ingest`` / ``step_consolidate`` / ``step_query`` 都
    走这条路径，确保 4 个 task_version 在 RL 侧完全等价。
    """
    payload = event.payload
    user_id = payload.get("user_id", "default_user")
    session_id = payload.get("session_id", "")

    if event.type == EventType.MESSAGE:
        task = triad.ingest
        kwargs: dict[str, Any] = {
            "messages": payload.get("messages") or [],
        }
        # 可选透传：session_time、ingest_number、message_offset、trajectory_id
        for key in ("session_time", "ingest_number", "message_offset", "trajectory_id"):
            if key in payload:
                kwargs[key] = payload[key]
    elif event.type == EventType.MEMORY_CONSOLIDATION:
        task = triad.consolidate
        kwargs = {}
        # 透传可选 hint：min_items_for_evolution / max_turns 等不在此处覆盖
    elif event.type == EventType.MEMORY_QUERY:
        task = triad.retrieve
        kwargs = {
            "query": payload.get("query", ""),
        }
        for key in ("messages", "latest_memory", "session_time"):
            if key in payload:
                kwargs[key] = payload[key]
    else:
        raise ValueError(f"unsupported event.type: {event.type!r}")

    task_name = getattr(task, "task_name", task.__class__.__name__)
    try:
        result_dict: dict[str, Any] = await task.execute(
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — 把异常封装到 TaskResult.error
        logger.exception("dispatch_event: task=%s failed: %s", task_name, exc)
        return TaskResult(task_name=task_name, error=str(exc))

    return _to_task_result(task, task_name, result_dict, event.type)


def _to_task_result(
    task: "BaseContextTask",
    task_name: str,
    result_dict: dict[str, Any],
    event_type: EventType,
) -> TaskResult:
    """把 BaseContextTask.execute() 返回的 dict 映射为 TaskResult。

    不同 task_version + event_type 组合的字段约定：

    | event             | 关键字段 (来源)                                                       |
    |-------------------|-----------------------------------------------------------------------|
    | MESSAGE           | ``finish_result`` / ``ingest_finish_summary`` / ``tools_called``       |
    | MEMORY_QUERY      | ``query_memory`` / ``retrieved_context`` / ``status``                  |
    | MEMORY_CONSOLIDA. | ``status`` / 通用 LLM/工具统计                                          |

    遗留：``IngestContextAtomicCodeT2Task`` 把摘要写在 ``self.result_extra`` 而非 ``_stats``，
    这里读取 fallback。
    """
    # 通用 error 提取
    error_str = ""
    if not bool(result_dict.get("success", True)):
        error_str = str(result_dict.get("error", "") or "")

    # 摘要 / 输出
    finish_summary = ""
    final_output = ""
    finish_reason = ""

    if event_type == EventType.MESSAGE:
        # 优先从 task.result_extra 读 atomic_code_t2 的摘要
        result_extra = getattr(task, "result_extra", None)
        if isinstance(result_extra, dict):
            finish_summary = str(result_extra.get("ingest_finish_summary", "") or "")
            finish_reason = str(result_extra.get("ingest_finish_reason", "") or "")
        # T2 ingest 把摘要散落在多个 batch_stats 里；取最后一个的 finish_result
        if not finish_summary:
            batches = result_dict.get("ingest_batches") or []
            if isinstance(batches, list) and batches:
                last = batches[-1]
                if isinstance(last, dict):
                    finish_summary = str(last.get("finish_result", "") or "")
        if not finish_summary:
            # 最后的兜底：直接 stringify summary 字段
            finish_summary = str(result_dict.get("finish_result", "") or "")
    elif event_type == EventType.MEMORY_QUERY:
        final_output = str(
            result_dict.get("query_memory")
            or result_dict.get("retrieved_context")
            or ""
        )
        finish_reason = str(result_dict.get("status", "") or "")
    else:  # MEMORY_CONSOLIDATION
        finish_reason = str(result_dict.get("status", "") or "")

    return TaskResult(
        task_name=task_name,
        finish_summary=finish_summary,
        final_output=final_output,
        finish_reason=finish_reason,
        error=error_str,
        metadata={
            "task_name": result_dict.get("task_name", task_name),
            "total_latency_s": result_dict.get("total_latency_s"),
            "llm_calls": result_dict.get("llm_calls"),
            "tool_calls": result_dict.get("tool_calls"),
        },
        stats=result_dict,
    )


__all__ = [
    "TaskTriad",
    "build_task_triad",
    "dispatch_event",
]
