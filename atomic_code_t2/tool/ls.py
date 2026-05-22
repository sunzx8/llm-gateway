"""ls 工具 — 列出记忆库目录的直接子项。"""

from __future__ import annotations

import os

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
列出指定目录的直接子项（文件和子目录），按目录优先、名称字母序排列。
以 "." 开头的隐藏文件/目录会被自动过滤，不会出现在结果中。

适用场景：
- 浏览记忆库当前的目录结构
- 配合 read_file 进一步查看文件内容

不适用场景：
- 按内容查找 → 用 grep
- 知道目标路径直接读 → 用 read_file

参数：
- path: 目录的绝对路径，必须位于记忆库根目录之内。根目录本身也可以，传记忆库根目录的绝对路径即可。
"""


def _format_size(num_bytes: int) -> str:
    """人友好的文件大小：B / KB / MB / GB / TB。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class LsTool(BaseTool):
    """列出指定目录直接子项。"""

    def __init__(self, fs: FileSystemStore) -> None:
        self.fs = fs

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ls",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录的绝对路径（必须位于记忆库根目录内）",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str) -> ToolResult:
        ok, rel_or_err = safe_resolve(self.fs.base_path, path)
        if not ok:
            return ToolResult(content=rel_or_err, is_error=True)
        rel = rel_or_err
        full = os.path.join(self.fs.base_path, rel) if rel else self.fs.base_path

        if not os.path.exists(full):
            return ToolResult(content=f"路径不存在: {path}", is_error=True)
        if not os.path.isdir(full):
            return ToolResult(content=f"不是目录: {path}", is_error=True)

        try:
            entries = sorted(
                os.listdir(full),
                key=lambda n: (not os.path.isdir(os.path.join(full, n)), n.lower()),
            )
        except OSError as e:
            return ToolResult(content=f"读取目录失败: {e}", is_error=True)

        lines: list[str] = []
        for name in entries:
            # 隐藏所有 "." 开头的文件/目录（如 .DS_Store / .git），
            # 这些都是系统/工具元数据，不是用户记忆。
            if name.startswith("."):
                continue
            sub = os.path.join(full, name)
            if os.path.isdir(sub):
                lines.append(f"[目录] {name}/")
            else:
                try:
                    size = os.path.getsize(sub)
                    lines.append(f"[文件] {name}  ({_format_size(size)})")
                except OSError:
                    lines.append(f"[文件] {name}  (大小未知)")

        if not lines:
            return ToolResult(content="(目录为空)")
        return ToolResult(content="\n".join(lines))
