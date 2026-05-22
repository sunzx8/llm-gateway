"""Retrieve Context Multi Code Task — 多子类选择/生成消费任务。

与 RetrieveContextCodeTask 的区别：
- RetrieveContextCodeTask 直接加载固定的 .codegen/retrieve_memory.py 执行
- RetrieveContextMultiCodeTask 支持从多个历史消费子类中选择最优方案，
  或在无合适方案时通过 agent loop 生成新的子类

执行逻辑：
1. Phase 1: 子类选择 — 通过 1 次 LLM 调用，在历史消费子类代码中选择最合适的
2. Phase 2: 新子类生成 — 如果 Phase 1 未选择，通过 agent loop 生成新子类
3. Phase 3: 执行代码 — 加载并执行选择的或新生成的子类代码

使用方式：
    task = RetrieveContextMultiCodeTask(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        query="用户的编程语言偏好是什么？",
        session_id="session_001",
        user_id="user_001",
    )
    # result["retrieved_context"] 即为检索到的记忆上下文
"""
from __future__ import annotations

import inspect
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.codegen.loader import load_consumer_class
from context_task.utils.text_utils import truncate_with_hint
from context_task.tool import (
    EVAL_CODE_TOOL,
    FINISH_TOOL,
    PYTHON_SYNTAX_CHECK_TOOL,
    READ_ONLY_TOOLS,
    SUBMIT_RETRIEVE_CODE_TOOL,
)
from context_task.prompt.chinese_prompt import (
    _T2_QUERY_GUIDANCE_TEMPLATE,
    RETRIEVE_MULTI_CODE_SELECT_SYSTEM_PROMPT,
    RETRIEVE_MULTI_CODE_SELECT_USER_TEMPLATE,
    RETRIEVE_MULTI_CODE_GEN_SYSTEM_PROMPT,
    RETRIEVE_MULTI_CODE_GEN_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logger.logger as logger

from storage.file_system_store import (
    CODEGEN_DIR,
    RETRIEVE_SUBCLASS_PATTERN,
    RETRIEVE_BASE_CLASS_FILE_PATH,
    RETRIEVE_NEW_SUBCLASS_FILE_PATH as RETRIEVE_NEW_SUBCLASS_PATH,
)

# 预置子类标识前缀
PRESET_SUBCLASS_PREFIX = "__preset__:"
# 预置子类模块路径映射
PRESET_SUBCLASS_MAP: dict[str, str] = {
    "preset_retrieve_t3_consumer.py": "context_task.codegen.preset_retrieve_t3_consumer",
}

# Agent loop 最大轮次
MAX_TURNS = 20


# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class RetrieveContextMultiCodeTaskStats(TaskStats):
    """RetrieveContextMultiCodeTask 的任务级业务统计数据。"""

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'skipped' | 'empty' | 'error'"""

    reason: str = ""
    """失败或空结果的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    query: str = ""
    """检索查询文本"""

    # ── Phase 1: 子类选择 ──
    available_subclass_count: int = 0
    """可用的消费子类数量"""

    selected_subclass: str = ""
    """选择的子类文件名"""

    selection_reason: str = ""
    """选择/不选择的原因"""

    # ── Phase 2: 新子类生成 ──
    new_subclass_generated: bool = False
    """是否生成了新子类"""

    new_subclass_path: str = ""
    """新生成的子类文件路径"""

    generation_turns: int = 0
    """生成过程的 agent loop 轮次"""

    # ── Phase 3: 代码执行 ──
    code_path: str = ""
    """实际执行的消费代码路径"""

    execute_latency_ms: float = 0.0
    """代码执行耗时（毫秒）"""

    # ── 检索结果 ──
    retrieved_context: str = ""
    """最终检索到的上下文内容"""

    retrieved_context_length: int = 0
    """检索上下文内容的长度"""

    query_memory: str = ""
    """完整的记忆上下文（带 prompt 解释）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "available_subclass_count": self.available_subclass_count,
            "selected_subclass": self.selected_subclass,
            "selection_reason": self.selection_reason,
            "new_subclass_generated": self.new_subclass_generated,
            "new_subclass_path": self.new_subclass_path,
            "generation_turns": self.generation_turns,
            "code_path": self.code_path,
            "execute_latency_ms": round(self.execute_latency_ms, 1),
            "retrieved_context": self.retrieved_context,
            "retrieved_context_length": self.retrieved_context_length,
            "query_memory": self.query_memory,
        })
        return base


# ---------------------------------------------------------------------------
# RetrieveContextMultiCodeTask
# ---------------------------------------------------------------------------


class RetrieveContextMultiCodeTask(BaseContextTask):
    """通过多子类选择/生成机制来检索记忆的 ContextTask。

    整合以下逻辑：
    1. **Phase 1 - 子类选择**：通过 1 次 LLM 调用，在历史消费子类中选择最合适的
    2. **Phase 2 - 新子类生成**：如果 Phase 1 未选择，通过 agent loop 生成新子类
    3. **Phase 3 - 执行代码**：加载并执行选择的或新生成的子类代码

    生命周期：
        execute(query, session_id, user_id) →
            pre_run: 初始化统计
            run:
                Phase 1: 子类选择
                Phase 2: 新子类生成（可选）
                Phase 3: 执行消费代码
            post_run: 记录完成日志
    """

    task_name = "retrieve_context_multi_code"

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        **kwargs: Any,
    ):
        """初始化 RetrieveContextMultiCodeTask。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            **kwargs: 兼容工厂方法传入的额外参数。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, query: str = "", session_id: str = "", **kwargs: Any) -> None:
        """前置处理：初始化业务统计。"""
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0
        self._stats = RetrieveContextMultiCodeTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    async def run(
        self,
        query: str = "",
        session_id: str = "",
        user_id: str = "default_user",
        messages: list[dict[str, Any]] | None = None,
        latest_memory: str = "",
        **kwargs: Any,
    ) -> None:
        """核心逻辑：子类选择 → 新子类生成（可选） → 执行代码。

        Args:
            query: 检索查询文本（必需）。
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话的完整消息列表。
            latest_memory: 上一轮召回的最新记忆信息（保留接口兼容性）。
        """
        biz_stats: RetrieveContextMultiCodeTaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.query = query

        if not query:
            biz_stats.status = "empty"
            biz_stats.reason = "empty query"
            return

        logger.info(
            "RetrieveContextMultiCodeTask: run start "
            "(query_len=%d, user_id=%s, session_id=%s)",
            len(query), user_id, session_id,
        )

        # ── Phase 1: 子类选择 ──
        selected_path = await self._phase1_select_subclass(
            query=query,
            user_id=user_id,
            session_id=session_id,
            biz_stats=biz_stats,
        )

        # ── Phase 2: 新子类生成（仅当 Phase 1 未选择时） ──
        # NOTE: 暂时注释掉 Phase 2 新代码生成逻辑
        # if not selected_path:
        #     selected_path = await self._phase2_generate_new_subclass(
        #         query=query,
        #         user_id=user_id,
        #         session_id=session_id,
        #         messages=messages,
        #         biz_stats=biz_stats,
        #     )

        # ── Phase 3: 执行代码 ──
        if selected_path:
            await self._phase3_execute_code(
                code_rel_path=selected_path,
                query=query,
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                biz_stats=biz_stats,
            )
        else:
            biz_stats.status = "skipped"
            biz_stats.reason = "无可用的消费子类"
            logger.warning(
                "RetrieveContextMultiCodeTask: 无可用代码，跳过执行"
            )

        # 拼接 query_memory
        context = biz_stats.retrieved_context
        if context:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_body = context.strip() or "(No relevant memories found)"
            t2_schema_notes = _T2_QUERY_GUIDANCE_TEMPLATE.format(now=now)
            biz_stats.query_memory = (
                f"# Memory Context\n\n"
                f"{memory_body}\n\n"
                f"---\n\n"
                f"{t2_schema_notes}\n"
            )

        logger.info(
            "RetrieveContextMultiCodeTask: run completed "
            "(status=%s, context_len=%d)",
            biz_stats.status, biz_stats.retrieved_context_length,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: RetrieveContextMultiCodeTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "RetrieveContextMultiCodeTask: completed, "
            "status=%s, code_path=%s, context_len=%d",
            biz_stats.status,
            biz_stats.code_path,
            biz_stats.retrieved_context_length,
        )

    # ------------------------------------------------------------------
    # Phase 1: 子类选择
    # ------------------------------------------------------------------

    async def _phase1_select_subclass(
        self,
        query: str,
        user_id: str,
        session_id: str,
        biz_stats: RetrieveContextMultiCodeTaskStats,
    ) -> str:
        """Phase 1: 通过 1 次 LLM 调用，在历史消费子类中选择最合适的。

        Args:
            query: 用户查询。
            user_id: 用户 ID。
            session_id: 会话 ID。
            biz_stats: 业务统计实例。

        Returns:
            选择的子类文件相对路径，未选择返回空字符串。
        """
        logger.info("RetrieveContextMultiCodeTask: Phase 1 - 子类选择开始")

        # 1. 收集可用的消费子类
        subclass_files = self._find_available_subclasses()
        biz_stats.available_subclass_count = len(subclass_files)

        if not subclass_files:
            logger.info(
                "RetrieveContextMultiCodeTask: Phase 1 - 无可用子类，跳过选择"
            )
            biz_stats.selection_reason = "无可用的消费子类文件"
            return ""

        # 如果只有一个子类，直接选择无需调用 LLM
        if len(subclass_files) == 1:
            only_file = subclass_files[0]
            biz_stats.selected_subclass = only_file.split("/")[-1]
            biz_stats.selection_reason = "仅有一个可用子类，直接选择"
            logger.info(
                "RetrieveContextMultiCodeTask: Phase 1 - 仅一个子类，直接选择: %s",
                only_file,
            )
            return only_file

        # 2. 读取子类代码
        subclass_codes_parts = []
        for path in subclass_files:
            if path.startswith(PRESET_SUBCLASS_PREFIX):
                # 预置子类：从代码仓库中读取源码
                filename = path[len(PRESET_SUBCLASS_PREFIX):]
                code = self._read_preset_subclass_code(filename)
            else:
                code = self.fs.read_file(path)
                filename = path.split("/")[-1]
            if code and not code.startswith("ERROR"):
                subclass_codes_parts.append(
                    f"### {filename}\n\n```python\n{code}\n```"
                )

        if not subclass_codes_parts:
            biz_stats.selection_reason = "所有子类文件读取失败"
            return ""

        subclass_codes = "\n\n---\n\n".join(subclass_codes_parts)

        # 3. 读取 index.md
        index_path, index_content = self.fs.read_index()

        # 5. 构建 user prompt
        user_prompt = RETRIEVE_MULTI_CODE_SELECT_USER_TEMPLATE.format(
            subclass_codes_dir=CODEGEN_DIR,
            subclass_codes=subclass_codes,
            index_path=index_path,
            index_content=truncate_with_hint(index_content, 4000, index_path),
            query=query,
        )

        # 6. 调用 LLM（单次调用，无工具）
        try:
            response = await self.llm_generate_with_stat(
                RETRIEVE_MULTI_CODE_SELECT_SYSTEM_PROMPT,
                [{"role": "user", "content": user_prompt}],
                tools=None,
                label="phase1_select_subclass",
            )

            # 7. 解析 JSON 结果
            result_text = response.content or response.reasoning_content or ""
            selected_file = self._parse_selection_result(result_text, subclass_files)

            if selected_file:
                biz_stats.selected_subclass = selected_file.split("/")[-1]
                biz_stats.selection_reason = self._extract_reason(result_text)
                logger.info(
                    "RetrieveContextMultiCodeTask: Phase 1 - 选择子类: %s",
                    selected_file,
                )
                return selected_file
            else:
                biz_stats.selection_reason = self._extract_reason(result_text) or "LLM 未选择任何子类"
                logger.info(
                    "RetrieveContextMultiCodeTask: Phase 1 - 未选择任何子类 (reason=%s)",
                    biz_stats.selection_reason,
                )
                return ""

        except Exception as e:
            logger.error(
                "RetrieveContextMultiCodeTask: Phase 1 - LLM 调用失败: %s", e
            )
            biz_stats.selection_reason = f"LLM 调用异常: {e}"
            # 降级：选择第一个可用子类
            if subclass_files:
                fallback = subclass_files[0]
                biz_stats.selected_subclass = fallback.split("/")[-1]
                biz_stats.selection_reason += f"，降级选择 {fallback}"
                return fallback
            return ""

    # ------------------------------------------------------------------
    # Phase 2: 新子类生成
    # ------------------------------------------------------------------

    async def _phase2_generate_new_subclass(
        self,
        query: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None,
        biz_stats: RetrieveContextMultiCodeTaskStats,
    ) -> str:
        """Phase 2: 通过 agent loop 生成新的消费子类。

        Args:
            query: 用户查询。
            user_id: 用户 ID。
            session_id: 会话 ID。
            messages: 当前对话消息列表。
            biz_stats: 业务统计实例。

        Returns:
            新生成的子类文件相对路径，失败返回空字符串。
        """
        logger.info("RetrieveContextMultiCodeTask: Phase 2 - 新子类生成开始")

        # 1. 读取增强基类代码
        base_class_code = self.fs.read_file(RETRIEVE_BASE_CLASS_FILE_PATH)
        if base_class_code.startswith("ERROR"):
            biz_stats.status = "error"
            biz_stats.reason = "增强基类不存在，无法生成新子类"
            logger.warning(
                "RetrieveContextMultiCodeTask: Phase 2 - 无基类代码可用"
            )
            return ""

        # 2. 收集可参考的历史子类
        reference_parts = []
        subclass_files = self._find_available_subclasses()
        for path in subclass_files[:3]:  # 最多参考 3 个
            code = self.fs.read_file(path)
            if not code.startswith("ERROR"):
                filename = path.split("/")[-1]
                reference_parts.append(
                    f"### {filename}\n\n```python\n{code}\n```"
                )

        reference_subclasses = (
            "\n\n---\n\n".join(reference_parts) if reference_parts
            else "(无历史子类可参考)"
        )

        # 3. 收集记忆库状态
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        index_path, index_content = self.fs.read_index()

        # 4. 构建 user prompt
        user_prompt = RETRIEVE_MULTI_CODE_GEN_USER_TEMPLATE.format(
            base_class_path=RETRIEVE_BASE_CLASS_FILE_PATH,
            base_class_code=truncate_with_hint(
                base_class_code, 10000, RETRIEVE_BASE_CLASS_FILE_PATH
            ),
            reference_subclasses_dir=CODEGEN_DIR,
            reference_subclasses=truncate_with_hint(
                reference_subclasses, 8000, CODEGEN_DIR
            ),
            fs_file_count=len(fs_files),
            fs_tree=self.fs.tree(max_depth=3),
            vec_total=vec_stats.get("total_entries", 0),
            vec_collections_count=len(vec_stats.get("collections", {})),
            vec_collections=str(vec_stats.get("collections", {})),
            graph_nodes=graph_stats.get("total_nodes", 0),
            graph_edges=graph_stats.get("total_edges", 0),
            graph_stats=str(graph_stats),
            index_content=truncate_with_hint(index_content, 4000, index_path),
            query=query,
        )

        # 5. 构建工具列表
        tools = self._get_generation_tools()

        # 6. 执行 agent loop
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        submitted_code = ""

        for turn in range(MAX_TURNS):
            logger.info(
                "RetrieveContextMultiCodeTask: Phase 2 - agent loop turn %d/%d",
                turn + 1, MAX_TURNS,
            )
            biz_stats.generation_turns = turn + 1

            response = await self.llm_generate_with_stat(
                RETRIEVE_MULTI_CODE_GEN_SYSTEM_PROMPT,
                messages_history,
                tools=tools,
                label=f"phase2_gen_turn_{turn + 1}",
            )

            if not response.tool_calls:
                # 模型返回纯文本，无法继续
                logger.warning(
                    "RetrieveContextMultiCodeTask: Phase 2 - 模型返回纯文本，结束循环"
                )
                break

            messages_history.append(response.to_message())
            should_break = False

            for tc in response.tool_calls:
                if tc.name == "submit_retrieve_code":
                    # 拦截代码提交
                    code = tc.arguments.get("code", "")
                    if code:
                        # 验证代码
                        verify_result = self._verify_consumer_code(code)
                        if verify_result == "":
                            submitted_code = code
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "✅ 消费代码验证通过，已成功提交。请调用 finish 结束任务。",
                            })
                            logger.info(
                                "RetrieveContextMultiCodeTask: Phase 2 - 代码提交成功"
                            )
                        else:
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"❌ 代码验证失败：{verify_result}\n请修复后重新提交。",
                            })
                    else:
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "❌ 提交失败：code 参数为空。",
                        })
                    continue

                if tc.name == "finish":
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(task finished)",
                    })
                    should_break = True
                    continue

                # 执行其他工具调用（只读 + eval_code + python_syntax_check）
                result_str = await self.dispatch_tool(
                    tc.name, tc.arguments,
                    user_id=user_id,
                    session_id=session_id,
                    allow_write=False,
                )
                messages_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            if should_break:
                break
        else:
            logger.warning(
                "RetrieveContextMultiCodeTask: Phase 2 - 达到最大轮次 %d",
                MAX_TURNS,
            )

        # 7. 写入新子类文件
        if submitted_code:
            self.fs.write_file(RETRIEVE_NEW_SUBCLASS_PATH, submitted_code)
            biz_stats.new_subclass_generated = True
            biz_stats.new_subclass_path = RETRIEVE_NEW_SUBCLASS_PATH
            logger.info(
                "RetrieveContextMultiCodeTask: Phase 2 - 新子类已写入: %s",
                RETRIEVE_NEW_SUBCLASS_PATH,
            )
            return RETRIEVE_NEW_SUBCLASS_PATH
        else:
            logger.warning(
                "RetrieveContextMultiCodeTask: Phase 2 - 未生成有效代码"
            )
            return ""

    # ------------------------------------------------------------------
    # Phase 3: 执行代码
    # ------------------------------------------------------------------

    async def _phase3_execute_code(
        self,
        code_rel_path: str,
        query: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None,
        biz_stats: RetrieveContextMultiCodeTaskStats,
    ) -> None:
        """Phase 3: 加载并执行消费代码。

        Args:
            code_rel_path: 消费代码的相对路径。
            query: 用户查询。
            user_id: 用户 ID。
            session_id: 会话 ID。
            messages: 当前对话消息列表。
            biz_stats: 业务统计实例。
        """
        logger.info(
            "RetrieveContextMultiCodeTask: Phase 3 - 执行代码: %s",
            code_rel_path,
        )

        biz_stats.code_path = code_rel_path

        start = time.monotonic()
        try:
            # 1. 动态加载子类
            if code_rel_path.startswith(PRESET_SUBCLASS_PREFIX):
                # 预置子类：通过 importlib 直接加载
                ConsumerClass = self._load_preset_subclass(code_rel_path)
            else:
                code_abs_path = f"{self.fs.base_path}/{code_rel_path}"

                # 诊断日志：检查增强基类文件是否存在
                import os
                base_class_abs = f"{self.fs.base_path}/{RETRIEVE_BASE_CLASS_FILE_PATH}"
                base_exists = os.path.isfile(base_class_abs)
                logger.info(
                    "RetrieveContextMultiCodeTask: Phase 3 - 加载子类 "
                    "(code=%s, base_exists=%s, base_path=%s)",
                    code_abs_path, base_exists, base_class_abs,
                )

                ConsumerClass = load_consumer_class(code_abs_path)

            # 2. 实例化
            consumer = ConsumerClass(
                fs=self.fs,
                vec=self.vec,
                graph=self.graph,
                llm=self.llm,
            )

            # 3. 执行（防御性参数过滤：兼容签名不完整的子类）
            call_kwargs = dict(
                query=query,
                messages=messages or [],
                user_id=user_id,
                session_id=session_id,
            )
            try:
                sig = inspect.signature(consumer.retrieve_memory)
                accepted = set(sig.parameters.keys()) - {"self"}
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
            except (ValueError, TypeError):
                pass
            result = await consumer.retrieve_memory(**call_kwargs)

            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000

            # 4. 处理结果
            if result:
                biz_stats.retrieved_context = result
                biz_stats.retrieved_context_length = len(result)
                biz_stats.status = "success"
                logger.info(
                    "RetrieveContextMultiCodeTask: Phase 3 - 检索成功 "
                    "(context_len=%d, latency=%.1fms)",
                    len(result), biz_stats.execute_latency_ms,
                )
            else:
                biz_stats.status = "empty"
                biz_stats.reason = "消费代码执行成功但无输出"
                logger.info(
                    "RetrieveContextMultiCodeTask: Phase 3 - 执行成功但无输出"
                )

        except FileNotFoundError as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"消费代码文件不存在: {e}"
            logger.error(
                "RetrieveContextMultiCodeTask: Phase 3 - 文件不存在: %s", e,
            )

        except Exception as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"消费代码执行异常: {e}"
            logger.error(
                "RetrieveContextMultiCodeTask: Phase 3 - 执行失败: %s (path=%s)\n%s",
                e, code_rel_path, traceback.format_exc()[:1000],
            )


    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _find_available_subclasses(self) -> list[str]:
        """查找所有可用的消费子类文件。

        查找路径：
        - 预置子类（代码仓库中，始终可用）
        - .codegen/retrieve_memory_v1.py ~ v5.py（MultiCodeT2Task 生成）
        - .codegen/retrieve_memory_new.py（本任务生成）

        Returns:
            可用子类文件的路径列表。
            预置子类使用 "__preset__:<filename>" 格式标识。
        """
        available = []

        # 1. 预置子类（代码仓库中，始终可用）
        for filename in PRESET_SUBCLASS_MAP:
            available.append(f"{PRESET_SUBCLASS_PREFIX}{filename}")

        # 2. 查找 v1 ~ v5 子类
        for idx in range(1, 6):
            path = f"{CODEGEN_DIR}/{RETRIEVE_SUBCLASS_PATTERN.format(idx=idx)}"
            content = self.fs.read_file(path)
            if not content.startswith("ERROR"):
                available.append(path)

        # 3. 查找新生成的子类
        content = self.fs.read_file(RETRIEVE_NEW_SUBCLASS_PATH)
        if not content.startswith("ERROR"):
            available.append(RETRIEVE_NEW_SUBCLASS_PATH)

        return available

    def _parse_selection_result(
        self, result_text: str, available_files: list[str]
    ) -> str:
        """解析 LLM 的选择结果 JSON。

        Args:
            result_text: LLM 返回的文本（应为 JSON）。
            available_files: 可用的子类文件路径列表。

        Returns:
            选择的子类文件相对路径，未选择或解析失败返回空字符串。
        """
        try:
            # 尝试从文本中提取 JSON
            json_text = result_text.strip()
            # 处理 markdown 代码块包裹的情况
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            data = json.loads(json_text)
            selected = data.get("selected", "")

            if not selected:
                return ""

            # 匹配文件名到完整路径
            for path in available_files:
                if path.endswith(selected) or selected in path:
                    return path

            # 尝试匹配预置子类
            preset_path = f"{PRESET_SUBCLASS_PREFIX}{selected}"
            if preset_path in available_files:
                return preset_path

            # 尝试直接作为路径匹配
            full_path = f"{CODEGEN_DIR}/{selected}"
            if full_path in available_files:
                return full_path

            logger.warning(
                "RetrieveContextMultiCodeTask: 选择的文件 '%s' 不在可用列表中",
                selected,
            )
            return ""

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                "RetrieveContextMultiCodeTask: 解析选择结果失败: %s (text=%s)",
                e, result_text[:200],
            )
            return ""

    def _extract_reason(self, result_text: str) -> str:
        """从 LLM 返回的 JSON 中提取 reason 字段。"""
        try:
            json_text = result_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            data = json.loads(json_text)
            return data.get("reason", "")
        except Exception:
            return ""

    def _verify_consumer_code(self, code: str) -> str:
        """验证消费代码是否可以正确加载。

        Args:
            code: Python 代码内容。

        Returns:
            空字符串表示验证通过，否则返回错误信息。
        """
        import ast
        import tempfile
        import os

        # 1. 语法检查
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"语法错误: {e.msg} (行 {e.lineno})"

        # 2. 尝试动态加载验证
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            try:
                load_consumer_class(tmp_path)
                return ""  # 验证通过
            except ImportError as e:
                return f"加载失败: {e}"
            except Exception as e:
                return f"验证异常: {e}"
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            return f"临时文件创建失败: {e}"

    def _get_generation_tools(self) -> list[dict]:
        """获取 Phase 2 新子类生成阶段的工具列表。

        包含：只读工具 + eval_code + python_syntax_check + submit_retrieve_code + finish

        Returns:
            工具 schema 列表。
        """
        tools = list(READ_ONLY_TOOLS)
        tools.append(EVAL_CODE_TOOL)
        tools.append(PYTHON_SYNTAX_CHECK_TOOL)
        tools.append(SUBMIT_RETRIEVE_CODE_TOOL)
        tools.append(FINISH_TOOL)
        return tools

    # ------------------------------------------------------------------
    # 预置子类辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _read_preset_subclass_code(filename: str) -> str:
        """读取预置子类的源码。

        Args:
            filename: 预置子类文件名（如 'preset_retrieve_t3_consumer.py'）。

        Returns:
            源码字符串，读取失败返回 'ERROR: ...'。
        """
        import importlib.util
        from pathlib import Path

        module_path_str = PRESET_SUBCLASS_MAP.get(filename)
        if not module_path_str:
            return f"ERROR: 未知的预置子类: {filename}"

        # 通过模块路径找到源文件
        try:
            spec = importlib.util.find_spec(module_path_str)
            if spec and spec.origin:
                source_path = Path(spec.origin)
                if source_path.exists():
                    return source_path.read_text(encoding="utf-8")
        except Exception:
            pass

        # Fallback：直接构造路径
        try:
            parts = module_path_str.split(".")
            rel_path = "/".join(parts) + ".py"
            # 从当前工作目录查找
            source_path = Path(rel_path)
            if source_path.exists():
                return source_path.read_text(encoding="utf-8")
        except Exception:
            pass

        return f"ERROR: 无法读取预置子类源码: {filename}"

    @staticmethod
    def _load_preset_subclass(path_identifier: str) -> type:
        """加载预置子类。

        Args:
            path_identifier: 预置子类标识（如 '__preset__:preset_retrieve_t3_consumer.py'）。

        Returns:
            加载的 BaseMemoryConsumer 子类。

        Raises:
            ImportError: 加载失败。
        """
        import importlib

        filename = path_identifier[len(PRESET_SUBCLASS_PREFIX):]
        module_path_str = PRESET_SUBCLASS_MAP.get(filename)
        if not module_path_str:
            raise ImportError(f"未知的预置子类: {filename}")

        try:
            module = importlib.import_module(module_path_str)
        except ImportError as e:
            raise ImportError(f"预置子类模块加载失败: {module_path_str}: {e}") from e

        # 查找 BaseMemoryConsumer 的非抽象具体子类
        import inspect
        from context_task.codegen.base_memory_consumer import BaseMemoryConsumer

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseMemoryConsumer)
                and obj is not BaseMemoryConsumer
                and not inspect.isabstract(obj)
            ):
                return obj

        raise ImportError(
            f"在预置子类模块 {module_path_str} 中未找到 BaseMemoryConsumer 的非抽象子类"
        )

    # ------------------------------------------------------------------
    # 便捷方法：query_memory
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        session_id: str = "",
        user_id: str = "default_user",
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """查询记忆，返回检索到的上下文字符串。

        对齐 RetrieveContextTask.query_memory 的接口语义，
        便于上层系统直接替换调用。

        Args:
            query: 查询文本。
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话消息列表（可选）。

        Returns:
            检索到的记忆上下文字符串，无结果时返回空字符串。
        """
        logger.info(
            "RetrieveContextMultiCodeTask.query_memory: query=%s...",
            query[:80],
        )
        result = await self.execute(
            query=query,
            session_id=session_id,
            user_id=user_id,
            messages=messages,
        )
        context = result.get("retrieved_context", "")
        if context:
            logger.info(
                "RetrieveContextMultiCodeTask.query_memory: retrieved %d chars",
                len(context),
            )
        else:
            logger.info(
                "RetrieveContextMultiCodeTask.query_memory: no relevant context found"
            )
        return context
