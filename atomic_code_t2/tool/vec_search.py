"""vec_search 工具 — 在向量库中做语义检索。"""

from __future__ import annotations

from storage.stores_base import VectorStoreBase

from .base import BaseTool, ToolDefinition, ToolResult


_DEFAULT_COLLECTION = "memory"

_TOOL_DESCRIPTION = """\
在向量库中做语义检索，跨所有记忆条目按相似度返回 top_k。

适用场景：
- 判断新信息在已有记忆里有没有相似/冲突项（决定 ADD vs UPDATE）
- 按用户当前话题召回相关历史

返回格式：
- 每行 `[id=<id>] (score=<score>) <text>`
- id 可作为 vec_write 的 entry_id 用来 UPDATE 这条记录

参数：
- query: 检索文本
- top_k: 返回 K 条，默认 10
"""


class VecSearchTool(BaseTool):
    """跨集合语义检索。"""

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
            name="vec_search",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索 query 文本",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关的前 K 条，默认 10",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, query: str, top_k: int = 10) -> ToolResult:
        if not query:
            return ToolResult(content="query 不能为空", is_error=True)
        try:
            results = await self.vec.search_all(query=query, top_k=top_k)
        except Exception as e:
            return ToolResult(content=f"向量检索失败: {e}", is_error=True)

        if not results:
            return ToolResult(content="(无相关条目)")

        lines: list[str] = []
        for r in results:
            rid = r.get("id", "?")
            score = r.get("score", 0.0)
            text = r.get("text", "")
            lines.append(f"[id={rid}] (score={score:.3f}) {text}")
        return ToolResult(content="\n".join(lines))
