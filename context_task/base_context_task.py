"""Context Task 基类 — 定义统一的任务生命周期和统计封装。

BaseContextTask 是 context_task 目录下所有任务的抽象基类，提供：

1. **生命周期方法**：pre_run → run → post_run，子类通过覆写实现具体逻辑
2. **llm_generate_with_stat**：封装 LLM 调用，自动记录输入/输出内容、耗时、token 量
3. **tool_with_stat**：封装工具操作，自动记录工具名称、参数、结果、耗时
4. **save_checkpoint**：将当前 file、vdb 文件、graph 文件打包上传到 COS

设计原则：
- 与现有 BaseTask / T1BaseTask / T2BaseTask 解耦，不继承它们
- 统计数据结构化存储，便于后续聚合和展示
- COS 上传采用异步方式，不阻塞主流程
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface, LLMResponse
import logger.logger as logger


# ---------------------------------------------------------------------------
# 统计数据结构
# ---------------------------------------------------------------------------

@dataclass
class LLMCallStat:
    """单次 LLM 调用的统计记录。"""

    call_id: str = ""
    """唯一标识（自动生成）"""

    timestamp: float = 0.0
    """调用发起时的 Unix 时间戳"""

    # ── 输入 ──
    system_prompt: str = ""
    """系统提示词"""

    input_messages: list[dict[str, Any]] = field(default_factory=list)
    """输入消息列表（OpenAI Chat 格式）"""

    tools: list[dict[str, Any]] | None = None
    """工具定义（如有）"""

    # ── 输出 ──
    output_content: str = ""
    """模型输出的文本内容"""

    output_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """模型输出的工具调用"""

    reasoning_content: str = ""
    """推理模型的思维链内容"""

    model: str = ""
    """实际使用的模型名称"""

    # ── 统计 ──
    input_tokens: int = 0
    """输入 token 数"""

    output_tokens: int = 0
    """输出 token 数"""

    total_tokens: int = 0
    """总 token 数（input + output）"""

    latency_ms: float = 0.0
    """调用耗时（毫秒）"""

    success: bool = True
    """是否成功"""

    error: str = ""
    """错误信息（如有）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "system_prompt_len": len(self.system_prompt),
            "input_messages_count": len(self.input_messages),
            "tools_count": len(self.tools) if self.tools else 0,
            "output_content_len": len(self.output_content),
            "output_tool_calls_count": len(self.output_tool_calls),
            "reasoning_content_len": len(self.reasoning_content),
            "model": self.model,
            "input_tokens_k": round(self.input_tokens / 1000, 2),
            "output_tokens_k": round(self.output_tokens / 1000, 2),
            "total_tokens_k": round(self.total_tokens / 1000, 2),
            "latency_s": round(self.latency_ms / 1000, 2),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ToolCallStat:
    """单次工具调用的统计记录。"""

    call_id: str = ""
    """唯一标识（自动生成）"""

    timestamp: float = 0.0
    """调用发起时的 Unix 时间戳"""

    tool_name: str = ""
    """工具名称"""

    arguments: dict[str, Any] = field(default_factory=dict)
    """工具参数"""

    result: str = ""
    """工具返回结果（截断后）"""

    result_length: int = 0
    """完整结果长度"""

    latency_ms: float = 0.0
    """调用耗时（毫秒）"""

    success: bool = True
    """是否成功"""

    error: str = ""
    """错误信息（如有）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result_preview": self.result[:500] if self.result else "",
            "result_length": self.result_length,
            "latency_s": round(self.latency_ms / 1000, 2),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TaskStats:
    """任务级别的聚合统计。"""

    task_name: str = ""
    """任务名称"""

    start_time: float = 0.0
    """任务开始时间戳"""

    end_time: float = 0.0
    """任务结束时间戳"""

    total_latency_ms: float = 0.0
    """任务总耗时（毫秒）"""

    # ── LLM 统计 ──
    llm_calls: int = 0
    """LLM 调用次数"""

    llm_total_input_tokens: int = 0
    """LLM 总输入 token 数"""

    llm_total_output_tokens: int = 0
    """LLM 总输出 token 数"""

    llm_total_tokens: int = 0
    """LLM 总 token 数"""

    llm_total_latency_ms: float = 0.0
    """LLM 总调用耗时（毫秒）"""

    llm_call_details: list[LLMCallStat] = field(default_factory=list)
    """每次 LLM 调用的详细记录"""

    # ── 工具统计 ──
    tool_calls: int = 0
    """工具调用次数"""

    tool_total_latency_ms: float = 0.0
    """工具总调用耗时（毫秒）"""

    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    """按工具名称分组的调用次数"""

    tool_latency_by_name: dict[str, float] = field(default_factory=dict)
    """按工具名称分组的总耗时（毫秒）"""

    tool_call_details: list[ToolCallStat] = field(default_factory=list)
    """每次工具调用的详细记录"""

    # ── 错误统计 ──
    llm_errors: int = 0
    """LLM 调用失败次数"""

    tool_errors: int = 0
    """工具调用失败次数"""

    success: bool = True
    """任务是否成功"""

    error: str = ""
    """任务级错误信息"""

    def _compute_store_access_counts(self) -> dict[str, int]:
        """从 tool_calls_by_name 按前缀聚合各存储后端的访问次数。"""
        fs_count = 0
        vec_count = 0
        graph_count = 0
        for name, count in self.tool_calls_by_name.items():
            if name.startswith("fs_") or name == "quick_fs_search":
                fs_count += count
            elif name.startswith("vec_") or name == "quick_vec_search":
                vec_count += count
            elif name.startswith("graph_") or name == "quick_graph_search":
                graph_count += count
        return {
            "fs_access_count": fs_count,
            "vec_access_count": vec_count,
            "graph_access_count": graph_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（不含详细记录，用于摘要展示）。

        单位约定：
        - 耗时统一使用秒（s）
        - Token 数统一使用千（k）
        """
        store_access = self._compute_store_access_counts()
        return {
            "task_name": self.task_name,
            "total_latency_s": round(self.total_latency_ms / 1000, 2),
            "llm_calls": self.llm_calls,
            "llm_total_input_tokens_k": round(self.llm_total_input_tokens / 1000, 2),
            "llm_total_output_tokens_k": round(self.llm_total_output_tokens / 1000, 2),
            "llm_total_tokens_k": round(self.llm_total_tokens / 1000, 2),
            "llm_total_latency_s": round(self.llm_total_latency_ms / 1000, 2),
            "llm_errors": self.llm_errors,
            "tool_calls": self.tool_calls,
            "tool_total_latency_s": round(self.tool_total_latency_ms / 1000, 2),
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "tool_latency_by_name_s": {
                k: round(v / 1000, 2) for k, v in self.tool_latency_by_name.items()
            },
            "tool_errors": self.tool_errors,
            "success": self.success,
            "error": self.error,
            # ── 存储访问次数 ──
            **store_access,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """序列化为字典（含详细记录，用于完整日志）。"""
        d = self.to_dict()
        d["llm_call_details"] = [s.to_dict() for s in self.llm_call_details]
        d["tool_call_details"] = [s.to_dict() for s in self.tool_call_details]
        return d


# ---------------------------------------------------------------------------
# BaseContextTask
# ---------------------------------------------------------------------------

class BaseContextTask(ABC):
    """Context Task 基类 — 定义统一的任务生命周期和统计封装。

    生命周期：
        execute() → pre_run() → run() → post_run()

    子类需要实现：
        - run(): 核心任务逻辑
        - 可选覆写 pre_run() / post_run() 进行前置/后置处理

    内置封装：
        - llm_generate_with_stat(): 调用 LLM 并自动记录统计
        - tool_with_stat(): 执行工具操作并自动记录统计
        - save_checkpoint(): 将当前状态上传到 COS
    """

    task_name: str = "base_context"
    """任务名称（子类覆写）"""

    def __init__(self, llm: LLMInterface):
        """初始化 BaseContextTask。

        Args:
            llm: LLM 接口实例，用于调用大模型。
        """
        self.llm = llm
        self._stats = TaskStats(task_name=self.task_name)
        self._run_id: str = ""

    @property
    def stats(self) -> TaskStats:
        """获取当前任务的统计数据。"""
        return self._stats

    # ------------------------------------------------------------------
    # 生命周期方法
    # ------------------------------------------------------------------

    async def execute(self, user_id: str = "default_user", session_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """执行任务的完整生命周期：pre_run → run → post_run。

        这是外部调用的统一入口，子类不应覆写此方法，
        而是覆写 pre_run / run / post_run。

        Args:
            user_id: 用户ID，默认为"default_user"。
            session_id: 会话ID，默认为空字符串。
            **kwargs: 传递给 pre_run / run / post_run 的其他参数。

        Returns:
            合并后的任务统计字典（业务统计 + 通用统计），由子类的 _stats.to_dict() 生成。
        """
        self._run_id = uuid.uuid4().hex[:12]
        self._stats = TaskStats(task_name=self.task_name)
        self._stats.start_time = time.time()

        logger.info(
            "ContextTask [%s] execute start (run_id=%s, user_id=%s, session_id=%s)",
            self.task_name, self._run_id, user_id, session_id,
        )

        try:
            # Phase 1: 前置处理
            await self.pre_run(user_id=user_id, session_id=session_id, **kwargs)

            # Phase 2: 核心逻辑
            await self.run(user_id=user_id, session_id=session_id, **kwargs)

            # Phase 3: 后置处理
            await self.post_run(user_id=user_id, session_id=session_id, **kwargs)

            self._stats.success = True
            self._stats.error = ""

        except Exception as e:
            logger.exception(
                "BaseContextTask [%s] execute failed (run_id=%s, user_id=%s, session_id=%s): %s",
                self.task_name, self._run_id, user_id, session_id, e,
            )
            self._stats.success = False
            self._stats.error = str(e)

        finally:
            self._stats.end_time = time.time()
            self._stats.total_latency_ms = (
                (self._stats.end_time - self._stats.start_time) * 1000
            )
            logger.info(
                "BaseContextTask [%s] execute done (run_id=%s, user_id=%s, session_id=%s, "
                "latency=%.1fms, llm_calls=%d, tool_calls=%d, success=%s)",
                self.task_name, self._run_id, user_id, session_id,
                self._stats.total_latency_ms,
                self._stats.llm_calls,
                self._stats.tool_calls,
                self._stats.success,
            )

        return self._stats.to_dict()

    async def pre_run(self, user_id: str = "default_user", session_id: str = "", **kwargs: Any) -> None:
        """前置处理（子类可覆写）。

        在 run() 之前执行，用于：
        - 初始化资源
        - 校验参数
        - 加载上下文

        Args:
            user_id: 用户ID，默认为"default_user"。
            session_id: 会话ID，默认为空字符串。
            **kwargs: 从 execute() 透传的参数。
        """
        logger.debug("BaseContextTask [%s] pre_run (default noop, user_id=%s, session_id=%s)", self.task_name, user_id, session_id)

    @abstractmethod
    async def run(self, user_id: str = "default_user", session_id: str = "", **kwargs: Any) -> None:
        """核心任务逻辑（子类必须实现）。

        子类在此方法中执行业务逻辑，并将结果写入 self._stats。
        不需要返回值，所有结果通过 self._stats 传递。

        Args:
            user_id: 用户ID，默认为"default_user"。
            session_id: 会话ID，默认为空字符串。
            **kwargs: 从 execute() 透传的参数。
        """
        ...

    async def post_run(self, *, user_id: str = "default_user", session_id: str = "", **kwargs: Any) -> None:
        """后置处理（子类可覆写）。

        在 run() 之后执行，用于：
        - 清理资源
        - 持久化结果
        - 触发后续流程

        Args:
            user_id: 用户ID，默认为"default_user"。
            session_id: 会话ID，默认为空字符串。
            **kwargs: 从 execute() 透传的参数。
        """
        logger.debug("BaseContextTask [%s] post_run (default noop, user_id=%s, session_id=%s)", self.task_name, user_id, session_id)

    # ------------------------------------------------------------------
    # LLM 调用封装（自动统计）
    # ------------------------------------------------------------------

    async def llm_generate_with_stat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "",
    ) -> "LLMResponse":
        """封装 LLM 调用，自动记录输入/输出内容、耗时、token 量。

        Args:
            system: 系统提示词。
            messages: 输入消息列表（OpenAI Chat 格式）。
            tools: 工具定义（可选）。
            temperature: 温度参数（可选，使用 LLM 默认值）。
            max_tokens: 最大输出 token 数（可选）。
            label: 本次调用的标签（用于日志区分，如 "ingest_extract"）。

        Returns:
            LLMResponse 对象。

        Raises:
            原样抛出 LLM 调用的异常，但会先记录到统计中。
        """
        call_id = f"{self._run_id}_{self._stats.llm_calls + 1}"
        stat = LLMCallStat(
            call_id=call_id,
            timestamp=time.time(),
            system_prompt=system,
            input_messages=messages,
            tools=tools,
        )

        logger.info(
            "BaseContextTask [%s] llm_generate #%d start "
            "(label=%s, msgs=%d, tools=%s, system_len=%d)",
            self.task_name, self._stats.llm_calls + 1, label or "-",
            len(messages), len(tools) if tools else "none",
            len(system),
        )
        # DEBUG: 输入内容摘要，便于 trace 分析
        if logger.isEnabledFor(logging.DEBUG):
            last_msg = messages[-1] if messages else {}
            last_content = str(last_msg.get("content", ""))[:300]
            logger.debug(
                "BaseContextTask [%s] llm_generate #%d input_detail "
                "(label=%s, last_msg_role=%s, last_msg_preview=%.300s)",
                self.task_name, self._stats.llm_calls + 1, label or "-",
                last_msg.get("role", "?"), last_content,
            )

        start = time.monotonic()
        try:
            response = await self.llm.generate(
                system, messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )

            stat.latency_ms = (time.monotonic() - start) * 1000
            stat.output_content = response.content or ""
            stat.reasoning_content = response.reasoning_content or ""
            stat.model = response.model or ""
            stat.input_tokens = response.input_tokens
            stat.output_tokens = response.output_tokens
            stat.total_tokens = response.input_tokens + response.output_tokens
            stat.success = True

            # 记录工具调用
            if response.tool_calls:
                stat.output_tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, dict)
                        else str(tc.arguments),
                    }
                    for tc in response.tool_calls
                ]

            logger.info(
                "BaseContextTask [%s] llm_generate #%d done "
                "(label=%s, model=%s, in=%d, out=%d, total=%d, "
                "latency=%.1fms, tool_calls=%d)",
                self.task_name, self._stats.llm_calls + 1, label or "-",
                stat.model, stat.input_tokens, stat.output_tokens,
                stat.total_tokens, stat.latency_ms,
                len(stat.output_tool_calls),
            )
            # DEBUG: 输出内容摘要，便于 trace 分析
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "BaseContextTask [%s] llm_generate #%d output_detail "
                    "(label=%s, content_preview=%.500s, "
                    "reasoning_len=%d, tool_calls=%s)",
                    self.task_name, self._stats.llm_calls + 1, label or "-",
                    stat.output_content[:500],
                    len(stat.reasoning_content),
                    [
                        {"name": tc["name"], "args_keys": list(tc["arguments"].keys()) if isinstance(tc["arguments"], dict) else "..."}
                        for tc in stat.output_tool_calls
                    ] if stat.output_tool_calls else "none",
                )

            return response

        except Exception as e:
            stat.latency_ms = (time.monotonic() - start) * 1000
            stat.success = False
            stat.error = str(e)
            self._stats.llm_errors += 1

            logger.error(
                "BaseContextTask [%s] llm_generate #%d failed "
                "(label=%s, latency=%.1fms): %s",
                self.task_name, self._stats.llm_calls + 1, label or "-",
                stat.latency_ms, e,
            )
            raise

        finally:
            # 无论成功失败都记录统计
            self._stats.llm_calls += 1
            self._stats.llm_total_input_tokens += stat.input_tokens
            self._stats.llm_total_output_tokens += stat.output_tokens
            self._stats.llm_total_tokens += stat.total_tokens
            self._stats.llm_total_latency_ms += stat.latency_ms
            self._stats.llm_call_details.append(stat)

    # ------------------------------------------------------------------
    # 工具调用封装（自动统计）
    # ------------------------------------------------------------------

    async def tool_with_stat(
        self,
        tool_name: str,
        func,
        *args: Any,
        arguments_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """封装工具操作，自动记录工具名称、参数、结果、耗时。

        支持同步和异步函数。

        Args:
            tool_name: 工具名称（如 "fs_write", "vec_search", "graph_query"）。
            func: 要执行的工具函数（同步或异步）。
            *args: 传递给 func 的位置参数。
            arguments_summary: 参数摘要（用于日志记录，避免记录过大的参数）。
            **kwargs: 传递给 func 的关键字参数。

        Returns:
            工具函数的返回值。

        Raises:
            原样抛出工具函数的异常，但会先记录到统计中。
        """
        import asyncio
        import inspect

        call_id = f"{self._run_id}_tool_{self._stats.tool_calls + 1}"
        stat = ToolCallStat(
            call_id=call_id,
            timestamp=time.time(),
            tool_name=tool_name,
            arguments=arguments_summary or {},
        )

        logger.info(
            "BaseContextTask [%s] tool_call #%d: %s start (args_summary=%s)",
            self.task_name, self._stats.tool_calls + 1, tool_name,
            {k: str(v)[:80] for k, v in (arguments_summary or {}).items()},
        )

        start = time.monotonic()
        try:
            # 支持同步和异步函数
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            stat.latency_ms = (time.monotonic() - start) * 1000
            result_str = str(result) if result is not None else ""
            stat.result = result_str[:2000]  # 截断存储
            stat.result_length = len(result_str)
            stat.success = True

            logger.info(
                "BaseContextTask [%s] tool_call #%d: %s done "
                "(latency=%.1fms, result_len=%d)",
                self.task_name, self._stats.tool_calls + 1, tool_name,
                stat.latency_ms, stat.result_length,
            )
            # DEBUG: 工具结果摘要
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "BaseContextTask [%s] tool_call #%d: %s result_preview=%.500s",
                    self.task_name, self._stats.tool_calls + 1, tool_name,
                    result_str[:500],
                )

            return result

        except Exception as e:
            stat.latency_ms = (time.monotonic() - start) * 1000
            stat.success = False
            stat.error = str(e)
            self._stats.tool_errors += 1

            logger.error(
                "BaseContextTask [%s] tool_call #%d: %s failed "
                "(latency=%.1fms): %s",
                self.task_name, self._stats.tool_calls + 1, tool_name,
                stat.latency_ms, e,
            )
            raise

        finally:
            # 无论成功失败都记录统计
            self._stats.tool_calls += 1
            self._stats.tool_total_latency_ms += stat.latency_ms
            self._stats.tool_calls_by_name[tool_name] = (
                self._stats.tool_calls_by_name.get(tool_name, 0) + 1
            )
            self._stats.tool_latency_by_name[tool_name] = (
                self._stats.tool_latency_by_name.get(tool_name, 0.0) + stat.latency_ms
            )
            self._stats.tool_call_details.append(stat)

    # ------------------------------------------------------------------
    # 统一工具分派（通过 ToolRegistry）
    # ------------------------------------------------------------------

    async def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        user_id: str = "",
        session_id: str = "",
        allow_write: bool = False,
    ) -> str:
        """通过 ToolRegistry 统一分派并执行工具，自动记录统计。

        Args:
            tool_name: 工具名称（如 "fs_write", "vec_search"）。
            args: LLM 传入的工具参数。
            user_id: 当前用户 ID（传给 session_view / eval_code 等工具）。
            session_id: 当前 session ID（传给需要的工具）。
            allow_write: 是否允许写操作（用于 fs_execute_bash）。

        Returns:
            工具执行结果字符串。
        """
        from context_task.tool import default_registry

        # 构建 deps
        deps: dict[str, Any] = {
            "fs": self.fs,
            "vec": self.vec,
            "graph": self.graph,
            "llm": self.llm,
            "user_id": user_id,
            "session_id": session_id,
            "allow_write": allow_write,
        }

        result = await self.tool_with_stat(
            tool_name,
            default_registry.execute,
            tool_name, args, **deps,
            arguments_summary={"tool": tool_name, **{k: str(v)[:100] for k, v in args.items()}},
        )
        return str(result)

    # ------------------------------------------------------------------
    # Checkpoint 保存（上传到 COS）
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self,
        *,
        file_dir: str | Path | None = None,
        vdb_dir: str | Path | None = None,
        graph_dir: str | Path | None = None,
        cos_bucket: str = "",
        cos_prefix: str = "",
        cos_region: str = "ap-beijing",
        cos_secret_id: str = "",
        cos_secret_key: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将当前的 file、vdb 文件、graph 文件打包上传到 COS。

        内部流程：
        1. 收集需要上传的文件列表（file_dir / vdb_dir / graph_dir）
        2. 打包为 tar.gz 归档
        3. 上传到腾讯云 COS
        4. 记录 checkpoint 元信息

        COS 配置优先级：参数 > 环境变量。

        Args:
            file_dir: 文件系统存储目录（FileSystemStore 的 base_path）。
            vdb_dir: 向量数据库文件目录。
            graph_dir: 图数据库文件目录。
            cos_bucket: COS 桶名称（默认从环境变量 COS_BUCKET 获取）。
            cos_prefix: COS 对象键前缀（默认从环境变量 COS_PREFIX 获取）。
            cos_region: COS 区域（默认 ap-beijing）。
            cos_secret_id: COS SecretId（默认从环境变量 COS_SECRET_ID 获取）。
            cos_secret_key: COS SecretKey（默认从环境变量 COS_SECRET_KEY 获取）。
            extra_metadata: 额外的元信息（会写入 checkpoint_meta.json）。

        Returns:
            上传结果字典，包含 cos_key、文件大小、上传耗时等。
        """
        import tarfile
        import tempfile

        # ── 解析 COS 配置 ──
        bucket = cos_bucket or os.environ.get("COS_BUCKET", "")
        prefix = cos_prefix or os.environ.get("COS_PREFIX", "checkpoints")
        region = cos_region or os.environ.get("COS_REGION", "ap-beijing")
        secret_id = cos_secret_id or os.environ.get("COS_SECRET_ID", "")
        secret_key = cos_secret_key or os.environ.get("COS_SECRET_KEY", "")

        if not bucket:
            logger.warning(
                "BaseContextTask [%s] save_checkpoint: COS_BUCKET not configured, "
                "skipping upload",
                self.task_name,
            )
            return {"status": "skipped", "reason": "COS_BUCKET not configured"}

        if not secret_id or not secret_key:
            logger.warning(
                "BaseContextTask [%s] save_checkpoint: COS credentials not configured, "
                "skipping upload",
                self.task_name,
            )
            return {"status": "skipped", "reason": "COS credentials not configured"}

        # ── 收集文件 ──
        dirs_to_archive: list[tuple[str, Path]] = []
        if file_dir:
            p = Path(file_dir)
            if p.exists():
                dirs_to_archive.append(("files", p))
        if vdb_dir:
            p = Path(vdb_dir)
            if p.exists():
                dirs_to_archive.append(("vdb", p))
        if graph_dir:
            p = Path(graph_dir)
            if p.exists():
                dirs_to_archive.append(("graph", p))

        if not dirs_to_archive:
            logger.info(
                "BaseContextTask [%s] save_checkpoint: no directories to archive",
                self.task_name,
            )
            return {"status": "skipped", "reason": "no directories to archive"}

        # ── 打包为 tar.gz ──
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"{self.task_name}_{self._run_id}_{timestamp_str}.tar.gz"

        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / archive_name

                # 写入元信息
                meta = {
                    "task_name": self.task_name,
                    "run_id": self._run_id,
                    "timestamp": timestamp_str,
                    "stats_summary": self._stats.to_dict(),
                    "archived_dirs": {
                        label: str(path) for label, path in dirs_to_archive
                    },
                    **(extra_metadata or {}),
                }
                meta_path = Path(tmpdir) / "checkpoint_meta.json"
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                with tarfile.open(archive_path, "w:gz") as tar:
                    # 添加元信息
                    tar.add(str(meta_path), arcname="checkpoint_meta.json")
                    # 添加各目录
                    for label, dir_path in dirs_to_archive:
                        tar.add(str(dir_path), arcname=label)

                archive_size = archive_path.stat().st_size
                logger.info(
                    "BaseContextTask [%s] save_checkpoint: archive created "
                    "(%s, %.2f MB)",
                    self.task_name, archive_name,
                    archive_size / (1024 * 1024),
                )

                # ── 上传到 COS ──
                cos_key = f"{prefix}/{archive_name}"
                upload_result = await self._upload_to_cos(
                    file_path=str(archive_path),
                    bucket=bucket,
                    cos_key=cos_key,
                    region=region,
                    secret_id=secret_id,
                    secret_key=secret_key,
                )

            latency_ms = (time.monotonic() - start) * 1000
            result = {
                "status": "success",
                "cos_key": cos_key,
                "archive_name": archive_name,
                "archive_size_bytes": archive_size,
                "upload_latency_s": round(latency_ms / 1000, 2),
                **upload_result,
            }
            logger.info(
                "BaseContextTask [%s] save_checkpoint: uploaded to COS "
                "(key=%s, size=%.2f MB, latency=%.1fms)",
                self.task_name, cos_key,
                archive_size / (1024 * 1024), latency_ms,
            )
            return result

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "BaseContextTask [%s] save_checkpoint failed "
                "(latency=%.1fms): %s",
                self.task_name, latency_ms, e,
            )
            return {
                "status": "error",
                "error": str(e),
                "latency_s": round(latency_ms / 1000, 2),
            }

    @staticmethod
    async def _upload_to_cos(
        file_path: str,
        bucket: str,
        cos_key: str,
        region: str,
        secret_id: str,
        secret_key: str,
    ) -> dict[str, Any]:
        """上传文件到腾讯云 COS。

        使用 cos-python-sdk-v5，在线程池中执行同步上传以避免阻塞事件循环。

        Args:
            file_path: 本地文件路径。
            bucket: COS 桶名称。
            cos_key: COS 对象键。
            region: COS 区域。
            secret_id: COS SecretId。
            secret_key: COS SecretKey。

        Returns:
            上传结果字典。
        """
        import asyncio

        def _sync_upload() -> dict[str, Any]:
            try:
                from qcloud_cos import CosConfig, CosS3Client
            except ImportError:
                logger.warning(
                    "cos-python-sdk-v5 not installed, "
                    "install via: pip install cos-python-sdk-v5"
                )
                return {"upload_status": "skipped", "reason": "cos-python-sdk-v5 not installed"}

            config = CosConfig(
                Region=region,
                SecretId=secret_id,
                SecretKey=secret_key,
            )
            client = CosS3Client(config)

            response = client.upload_file(
                Bucket=bucket,
                Key=cos_key,
                LocalFilePath=file_path,
                EnableMD5=True,
            )
            return {
                "upload_status": "success",
                "etag": response.get("ETag", ""),
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_upload)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_stats_summary(self) -> dict[str, Any]:
        """获取当前任务的统计摘要。"""
        return self._stats.to_dict()

    def get_stats_full(self) -> dict[str, Any]:
        """获取当前任务的完整统计（含每次调用详情）。"""
        return self._stats.to_full_dict()

    def reset_stats(self) -> None:
        """重置统计数据（用于任务复用场景）。"""
        self._stats = TaskStats(task_name=self.task_name)
        self._run_id = ""
