"""read_file 工具 — 读取记忆库中的文件内容。"""

from __future__ import annotations

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult


_DEFAULT_MAX_LINES = 2000
_MAX_LINE_CHARS = 2000
_MAX_OUTPUT = 60_000


_TOOL_DESCRIPTION = """\
读取记忆库中文件的内容，以带行号的格式输出（行号从 1 开始）。

⚠️ 行号前缀（形如 `     42\\t`）**不属于文件内容本身**，调用 edit_file 时 old_string 中不要包含这些前缀。

适用场景：
- 查看文件完整内容或指定行范围
- edit_file 之前先读一下，确认要替换的字符串与文件内容逐字符一致
- 可同时发起多个 read_file 调用以提高效率

参数：
- path: 文件的绝对路径（必须位于记忆库根目录内）
- offset: 起始行号（1-based），可选，不传从头开始
- limit: 读取行数，可选，不传读到末尾或单次最多 2000 行

输出末尾若出现形如 `(显示第 X-Y 行，共 N 行，剩余 K 行未显示)` 的提示，表示内容被截断（行数超限或单条输出超过 60KB），需要继续用 offset/limit 读后面的部分。
"""


def _format_with_line_numbers(
    lines: list[str],
    start_lineno: int,
    total: int,
) -> str:
    """cat -n 风格输出，受单行字符和总输出双重限制；末尾告知截断状态。"""
    parts: list[str] = []
    budget = _MAX_OUTPUT
    shown = 0

    for i, line in enumerate(lines, start=start_lineno):
        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + "..."
        formatted = f"     {i}\t{line}"
        if budget - len(formatted) - 1 < 0:
            break
        parts.append(formatted)
        budget -= len(formatted) + 1
        shown += 1

    end_lineno = start_lineno + shown - 1
    result = "\n".join(parts)

    if end_lineno < total:
        remaining = total - end_lineno
        result += f"\n\n(显示第 {start_lineno}-{end_lineno} 行，共 {total} 行，剩余 {remaining} 行未显示)"
    return result


class ReadFileTool(BaseTool):
    """读取文件内容，带行号输出。"""

    def __init__(self, fs: FileSystemStore) -> None:
        self.fs = fs

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的绝对路径（必须位于记忆库根目录内）",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "起始行号（1-based），不传从头开始",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "读取行数，不传读到末尾（单次最多 2000 行）",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> ToolResult:
        ok, rel_or_err = safe_resolve(self.fs.base_path, path)
        if not ok:
            return ToolResult(content=rel_or_err, is_error=True)
        rel = rel_or_err

        if not rel:
            return ToolResult(content="不是文件: 给的是记忆库根目录", is_error=True)

        content = self.fs.read_file(rel)
        if content.startswith("ERROR:"):
            return ToolResult(content=content, is_error=True)

        all_lines = content.splitlines()
        total = len(all_lines)

        start_idx = (offset - 1) if offset and offset > 0 else 0
        if start_idx >= total and total > 0:
            return ToolResult(
                content=f"offset={offset} 超出文件总行数 {total}",
                is_error=True,
            )
        count = limit if limit and limit > 0 else _DEFAULT_MAX_LINES
        selected = all_lines[start_idx : start_idx + count]

        return ToolResult(
            content=_format_with_line_numbers(selected, start_lineno=start_idx + 1, total=total)
        )
