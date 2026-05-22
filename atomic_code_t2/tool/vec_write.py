"""vec_write 工具 — 向向量库写入或更新一条记录。

对齐 T3 ingest 的关键行为：
- ADD 时自动从 text 提取时间 → 写入 metadata（occurred_at / ingest_time / ingest_turn）
- UPDATE 时同样更新 metadata
"""

from __future__ import annotations

from storage.stores_base import VectorStoreBase

from .base import BaseTool, ToolDefinition, ToolResult
from .session_context import SessionContext


_DEFAULT_COLLECTION = "memory"

_TOOL_DESCRIPTION = """\
向向量库写入一条记录：
- 不传 entry_id → ADD（新增）
- 传 entry_id   → UPDATE（更新已有条目）

适用场景：
- ADD：把抽取到的"原子化、第一人称叙述"写入向量库
- UPDATE：vec_search 找到相似条目的 id 后，把"旧+新"合并成时间序列再写回

最佳实践：
- text 用第一人称、原子陈述（一条独立可理解的事实）
- 含时间则在末尾以括号标注：`(2023-07-01, Sat)` / `(2023-07)` / `(2023)`
- UPDATE 时要保留旧的所有时间锚，不要直接覆盖丢失历史
"""


class VecWriteTool(BaseTool):
    """ADD / UPDATE 向量库条目。"""

    def __init__(
        self,
        vec: VectorStoreBase,
        *,
        default_collection: str = _DEFAULT_COLLECTION,
        session_ctx: SessionContext | None = None,
    ) -> None:
        self.vec = vec
        self.default_collection = default_collection
        self.session_ctx = session_ctx

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vec_write",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "原子化、第一人称的事实文本，含时间锚（如有）",
                    },
                    "entry_id": {
                        "type": "string",
                        "description": "已有条目的 id；不传 = ADD，传 = UPDATE",
                    },
                },
                "required": ["text"],
            },
        )

    async def execute(self, text: str, entry_id: str | None = None) -> ToolResult:
        if not text:
            return ToolResult(content="text 不能为空", is_error=True)

        # 递增 turn counter
        if self.session_ctx:
            self.session_ctx.next_turn()

        # 构建 metadata（对齐 T3：occurred_at + ingest_time + ingest_turn）
        metadata: dict | None = None
        if self.session_ctx:
            metadata = self.session_ctx.build_vec_metadata(text)

        # 确保 collection 存在（幂等）
        try:
            self.vec.create_collection(self.default_collection)
        except Exception:
            pass

        try:
            if entry_id:
                msg = self.vec.update(self.default_collection, entry_id, new_text=text, new_metadata=metadata)
                if "not found" in msg.lower() or "未找到" in msg:
                    # 跨 collection 兜底查找归属
                    found = False
                    for col in self.vec.list_collections():
                        if col == self.default_collection:
                            continue
                        try:
                            msg2 = self.vec.update(col, entry_id, new_text=text, new_metadata=metadata)
                            if "not found" not in msg2.lower() and "未找到" not in msg2:
                                msg = msg2
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        return ToolResult(
                            content=f"UPDATE 失败：未找到 entry_id={entry_id} 的条目",
                            is_error=True,
                        )
                return ToolResult(content=f"已更新 id={entry_id}: {msg}")

            ids = await self.vec.add(
                collection=self.default_collection,
                texts=[text],
                metadatas=[metadata] if metadata else None,
            )
            new_id = ids[0] if ids else "?"
            return ToolResult(content=f"已新增 id={new_id}")
        except Exception as e:
            return ToolResult(content=f"向量写入失败: {e}", is_error=True)
