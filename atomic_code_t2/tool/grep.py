"""grep 工具 — 在记忆库中按正则递归搜索文件内容。"""

from __future__ import annotations

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
在指定目录下递归搜索匹配正则表达式的**文件内容**，输出格式为 `路径:行号:命中行`。

适用场景：
- 按关键词检索某个主题/实体在记忆中的所有出处
- 判断某条新信息是否已存在（决定 ADD vs UPDATE）
- 可同时发起多个 grep 调用以提高效率

正则语法：
- Python re 标准语法。例如：`跑步|马拉松`、`Italian\\s*food`、`2023-0[789]`、`prefer.*Italian`
- 默认大小写不敏感

参数：
- pattern: 正则表达式
- path: 搜索起点的绝对路径（必须位于记忆库根目录内）。会递归到所有子目录

注意：
- 自动跳过 `.git` 等内部元数据
- 单次最多返回 50 条命中；达到上限时输出末尾会出现 `(命中数已达上限...)` 提示，可以换更精准的 pattern 重试
"""


class GrepTool(BaseTool):
    """正则内容搜索（递归）。"""

    def __init__(self, fs: FileSystemStore) -> None:
        self.fs = fs

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="grep",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python re 语法的正则表达式",
                    },
                    "path": {
                        "type": "string",
                        "description": "搜索起点的绝对路径（必须位于记忆库根目录内），会递归所有子目录",
                    },
                },
                "required": ["pattern", "path"],
            },
        )

    async def execute(self, pattern: str, path: str) -> ToolResult:
        ok, rel_or_err = safe_resolve(self.fs.base_path, path)
        if not ok:
            return ToolResult(content=rel_or_err, is_error=True)
        rel = rel_or_err  # 可能是 ""，表示根目录

        results = self.fs.grep(pattern, rel)
        if not results:
            return ToolResult(content=f"没有匹配 '{pattern}' 的内容")

        # 底层错误：路径不存在 / 正则非法 / 目录里没文件
        if len(results) == 1 and "error" in results[0]:
            err = results[0]["error"]
            if err.startswith("No files found"):
                return ToolResult(content=f"没有匹配 '{pattern}' 的内容（搜索范围内没有文件）")
            return ToolResult(content=err, is_error=True)

        # 收集命中 + 检测截断
        truncated = False
        lines: list[str] = []
        for r in results:
            if r.get("truncated"):
                truncated = True
                continue
            if "error" in r:
                continue
            lines.append(f"{r['path']}:{r['line']}:{r.get('match', '')}")

        if not lines:
            return ToolResult(content=f"没有匹配 '{pattern}' 的内容")

        out = "\n".join(lines)
        if truncated:
            out += f"\n\n(命中数已达上限 {len(lines)} 条，可能还有更多结果未显示，建议用更精准的 pattern 重试)"
        return ToolResult(content=out)
