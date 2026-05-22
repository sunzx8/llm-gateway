"""vec_delete 工具 — 从向量库中删除指定条目。

用于 consolidate 阶段的去重合并：删除语义重复的旧条目。
"""

from __future__ import annotations

from storage.stores_base import VectorStoreBase

from .base import BaseTool, ToolDefinition, ToolResult


_DEFAULT_COLLECTION = "memory"

_TOOL_DESCRIPTION = """\
从向量库中按 ID 删除一条或多条记录。

适用场景：
- 去重：发现两条语义相同的条目后，删除较旧/较差的那条
- 合并：将多条碎片合并成一条后，删除原始碎片

参数：
- ids: 要删除的条目 ID 列表（可通过 vec_search 获取 id）
"""


class VecDeleteTool(BaseTool):
    """删除向量库条目。"""

    def __init__(
        self,
        vec: VectorStoreBase,
        *,
        default_collection: str = _DEFAULT_COLLECTION,
    ) -> None:
        self.vec = vec
        self.default_collection = default_collection

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vec_delete",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要删除的条目 ID 列表",
                    },
                },
                "required": ["ids"],
            },
        )

    async def execute(self, ids: list[str]) -> ToolResult:
        if not ids:
            return ToolResult(content="ids 不能为空", is_error=True)

        try:
            # 先在默认 collection 尝试删除
            result = self.vec.delete(self.default_collection, ids)
            return ToolResult(content=f"已删除 {len(ids)} 条: {result}")
        except Exception as e:
            return ToolResult(content=f"向量删除失败: {e}", is_error=True)
