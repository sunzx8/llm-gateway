"""fs_delete 工具 — 删除记忆库中的文件。

用于 consolidate 阶段的文件重组：拆分后删除旧的大文件等。
"""

from __future__ import annotations

import os

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
删除记忆库中的一个文件。

适用场景：
- 文件拆分后删除原始大文件
- 清理过时/废弃的记忆文件

注意：
- 只能删除文件，不能删除目录
- 路径必须位于记忆库根目录内
- 操作不可逆，删除前请确认内容已迁移到新位置

参数：
- path: 文件的绝对路径（必须位于记忆库根目录内）
"""


class FsDeleteTool(BaseTool):
    """删除文件。"""

    def __init__(self, fs: FileSystemStore) -> None:
        self.fs = fs

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fs_delete",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的绝对路径（必须位于记忆库根目录内）",
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

        if not rel:
            return ToolResult(content="不能删除记忆库根目录", is_error=True)

        full = os.path.join(self.fs.base_path, rel)
        if not os.path.exists(full):
            return ToolResult(content=f"文件不存在: {path}", is_error=True)
        if not os.path.isfile(full):
            return ToolResult(content=f"不是文件（不能删除目录）: {path}", is_error=True)

        result = self.fs.delete_file(rel)
        return ToolResult(content=f"已删除 {path}: {result}")
