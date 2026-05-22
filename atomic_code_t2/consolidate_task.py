"""Atomic_Code_T2 - Consolidate Task（演进任务）。

对齐 T3 设计思想：
- ingest 快速粗粒度提取（1-2 次 LLM call）
- consolidate 是重型 agent loop（最多 30 轮），负责深度记忆维护
- 每 N 次 ingest 后触发一次（由 ingest_scheduler 控制）

职责：
1. 去重合并：语义重复的条目跨后端合并
2. 立场演化追踪：检测态度变化并显式记录
3. 文件重组：过大文件拆分、碎片文件合并
4. 跨后端冗余：确保重要事实在 FS/Vec/Graph 三后端都可检索
5. Vec/Graph 健康度维护
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import logger.logger as logger

from context_task.base_context_task import BaseContextTask
from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

from .agent_loop import run_agent_loop
from .prompt.consolidate_prompt import CONSOLIDATE_SYSTEM_PROMPT, CONSOLIDATE_USER_TEMPLATE
from .tool import build_consolidate_registry


class ConsolidateContextAtomicCodeT2Task(BaseContextTask):
    """Atomic_Code_T2 模式下的演进任务（重型 agent loop）。"""

    task_name = "consolidate_context_atomic_code_t2"

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStoreBase,
        graph_store: GraphStoreBase,
        *,
        min_items: int = 5,
        max_turns: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        self.min_items = min_items
        self.max_turns = max_turns
        self.extra_kwargs = kwargs
        self.result_extra: dict[str, Any] = {}

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """执行演进：检查记忆规模 → 构建上下文 → 跑 agent loop。"""

        # 1. 检查记忆规模，过小则跳过
        total_items = self._count_memory_items()
        if total_items < self.min_items:
            logger.info(
                "[%s] 记忆规模过小 (%d < %d)，跳过演进",
                self.task_name, total_items, self.min_items,
            )
            self.result_extra["consolidate_skipped"] = True
            self.result_extra["consolidate_reason"] = f"too few items ({total_items})"
            return

        # 2. 构建 user prompt（注入当前记忆状态快照 + 用户历史问题）
        session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S, %a")
        user_prompt = await self._build_user_prompt(user_id)

        # 3. 装配工具集（含删除工具）
        embedder = getattr(self.graph, "embedder", None) or getattr(self.vec, "embedder", None)
        registry = build_consolidate_registry(
            self.fs, self.vec, self.graph,
            session_time=session_time,
            embedder=embedder,
        )

        # 4. 跑 agent loop
        result = await run_agent_loop(
            llm=self.llm,
            system_prompt=CONSOLIDATE_SYSTEM_PROMPT,
            initial_messages=[{"role": "user", "content": user_prompt}],
            tools=registry,
            max_turns=self.max_turns,
            label=f"{self.task_name}[{user_id}]",
        )

        # 5. 记录结果
        self.result_extra["consolidate_finish_summary"] = result.finish_summary
        self.result_extra["consolidate_finish_reason"] = result.finish_reason
        self.result_extra["consolidate_turns_used"] = result.turns_used
        self.result_extra["consolidate_tool_calls"] = [
            {
                "turn": t.turn,
                "name": t.name,
                "is_error": t.is_error,
                "result_preview": t.result_content[:200],
            }
            for t in result.tool_traces
        ]

        logger.info(
            "[%s] consolidate done (turns=%d, reason=%s, tool_calls=%d, summary=%.200s)",
            self.task_name, result.turns_used, result.finish_reason,
            len(result.tool_traces), result.finish_summary,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _count_memory_items(self) -> int:
        """统计三后端的记忆条目总数。"""
        fs_count = len(self.fs.list_files(""))
        vec_count = 0
        try:
            stats = self.vec.get_stats()
            if isinstance(stats, dict):
                vec_count = stats.get("total_entries", 0)
        except Exception:
            pass
        graph_count = 0
        try:
            stats = self.graph.get_stats()
            if isinstance(stats, dict):
                graph_count = stats.get("total_nodes", 0)
        except Exception:
            pass
        return fs_count + vec_count + graph_count

    async def _build_user_prompt(self, user_id: str) -> str:
        """构建 user prompt：注入当前记忆状态的精简快照 + 用户历史对话/问题。"""

        memory_root = self.fs.base_path

        # FS 文件 + 大小（展示相对路径省 token，LLM 自行拼接根目录）
        fs_files = self.fs.list_files("")
        fs_size_lines: list[str] = []
        fs_file_count = 0
        for fpath in sorted(fs_files):
            # 跳过 .codegen/ 下的检索方案代码，不是用户记忆
            if fpath.startswith(".codegen/") or fpath.startswith(".codegen\\"):
                continue
            content = self.fs.read_file(fpath)
            if content.startswith("ERROR"):
                continue
            char_count = len(content)
            size_tag = ""
            if char_count > 3000:
                size_tag = " ⚠️ OVERSIZED"
            elif char_count > 2000:
                size_tag = " ⚠️ LARGE"
            fs_size_lines.append(f"  {fpath}: {char_count} chars{size_tag}")
            fs_file_count += 1

        fs_sizes = "\n".join(fs_size_lines) if fs_size_lines else "  (empty)"

        # Vec 采样（直接遍历内部数据结构，避免 async 嵌套问题）
        vec_entry_count = 0
        vec_sample_lines: list[str] = []
        try:
            stats = self.vec.get_stats()
            if isinstance(stats, dict):
                vec_entry_count = stats.get("total_entries", 0)
            # 直接访问内部数据拿采样（内存后端）
            if hasattr(self.vec, "_collections"):
                for coll_name, entries in self.vec._collections.items():
                    for entry in entries[:15]:
                        rid = entry.get("id", "?")
                        text = entry.get("text", "")[:200]
                        meta = entry.get("metadata", {}) or {}
                        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v) if meta else ""
                        line = f"  [{rid}] {text}"
                        if meta_str:
                            line += f"\n    metadata: {meta_str}"
                        vec_sample_lines.append(line)
                    if len(entries) > 15:
                        vec_sample_lines.append(f"  ... ({len(entries) - 15} more in '{coll_name}')")
        except Exception:
            pass
        vec_sample = "\n".join(vec_sample_lines) if vec_sample_lines else "  (no entries)"

        # Graph 摘要
        graph_node_count = 0
        graph_edge_count = 0
        graph_summary_lines: list[str] = []
        try:
            stats = self.graph.get_stats()
            if isinstance(stats, dict):
                graph_node_count = stats.get("total_nodes", 0)
                graph_edge_count = stats.get("total_edges", 0)
                labels = stats.get("node_labels", {})
                relations = stats.get("relation_types", {})
                if labels:
                    graph_summary_lines.append(f"  Labels: {labels}")
                if relations:
                    graph_summary_lines.append(f"  Relations: {relations}")
        except Exception:
            pass
        # 列出所有边（最多 50 条）
        try:
            if hasattr(self.graph, "_edges"):
                for edge in self.graph._edges[:50]:
                    src = edge.get("source", "?")
                    rel = edge.get("relation", "?")
                    tgt = edge.get("target", "?")
                    graph_summary_lines.append(f"  {src} --[{rel}]--> {tgt}")
                if len(self.graph._edges) > 50:
                    graph_summary_lines.append(f"  ... ({len(self.graph._edges) - 50} more)")
        except Exception:
            pass
        graph_summary = "\n".join(graph_summary_lines) if graph_summary_lines else "  (empty)"

        # 用户历史对话/问题（从 session_store 读取，辅助演进决策）
        user_history = await self._get_user_history(user_id)

        return CONSOLIDATE_USER_TEMPLATE.format(
            memory_root=memory_root,
            fs_file_count=fs_file_count,
            fs_sizes=fs_sizes,
            vec_entry_count=vec_entry_count,
            vec_sample=vec_sample,
            graph_node_count=graph_node_count,
            graph_edge_count=graph_edge_count,
            graph_summary=graph_summary,
            user_history=user_history,
        )

    async def _get_user_history(self, user_id: str) -> str:
        """从 session_store 读取用户的历史对话/问题，供演进参考。"""
        from storage.session_store import session_manager

        try:
            user_sessions = await session_manager.get_user_sessions(user_id)
            if not user_sessions:
                return "  (no history)"

            lines: list[str] = []
            for session_id, store in user_sessions.items():
                for key, msg in sorted(store.messages.items(), key=lambda x: int(x[0])):
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if role == "user" and content:
                        # 只取前 1000 字符，省 token
                        lines.append(f"  - {content[:1000]}")
            # 最多展示最近 20 条用户消息
            recent = lines[-20:] if len(lines) > 20 else lines
            return "\n".join(recent) if recent else "  (no history)"
        except Exception:
            return "  (no history)"
