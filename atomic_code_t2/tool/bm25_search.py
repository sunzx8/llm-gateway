"""bm25_search 工具 — 在记忆库内做 BM25 全文检索。

相比 grep（正则精确匹配），BM25 是按词频/逆文档频率打分的语义粗筛，
能找到"内容相关但用词不同"的文件。适合在不知道精确字面的情况下用。
"""

from __future__ import annotations

from storage.file_system_store import FileSystemStore

from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
对记忆库做 BM25 全文检索，返回最相关的若干文件 + 命中分数 + 内容预览。

适用场景：
- 想找"内容相关但不知道精确字面"的文件（语义粗筛）
- 与 grep 互补：grep 是正则精确匹配，BM25 是按词频/相关性打分

参数：
- query: 检索文本，可以是关键词组合（用空格分隔）或自然语言短句
- top_k: 最多返回多少个文件（默认 8）
"""

_DEFAULT_TOP_K = 8
_MAX_TOP_K = 30
_PREVIEW_CHARS = 200


class BM25SearchTool(BaseTool):
    """BM25 全文检索。"""

    def __init__(self, fs: FileSystemStore) -> None:
        self.fs = fs

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bm25_search",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索文本（关键词或自然语言）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"最多返回的文件数，默认 {_DEFAULT_TOP_K}",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, query: str, top_k: int | None = None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(content="query 不能为空", is_error=True)

        k = top_k if top_k and top_k > 0 else _DEFAULT_TOP_K
        k = min(k, _MAX_TOP_K)

        try:
            results = self.fs.search_bm25(query, top_k=k)
        except Exception as e:
            return ToolResult(content=f"BM25 检索失败：{e}", is_error=True)

        if not results:
            return ToolResult(content=f"BM25 未命中任何文件：'{query}'")

        lines: list[str] = []
        for path, score, snippet in results:
            preview = snippet[:_PREVIEW_CHARS]
            lines.append(f"[score={score:.2f}] {path}\n      {preview}")
        return ToolResult(content="\n".join(lines))
