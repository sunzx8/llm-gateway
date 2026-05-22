"""write_file 工具 — 创建记忆库中的新文件。

FS 的时间标记完全由 LLM 自己在正文里管理（圆括号嵌入事件时间）。
代码不做任何前缀注入，写什么就存什么。
"""

from __future__ import annotations

import os

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult
from .session_context import SessionContext


_TOOL_DESCRIPTION = """\
创建一个**新**文件。父目录不存在时自动创建。

仅用于全新文件。目标已存在且非空时会直接拒绝。
要修改/追加已有文件，请用 edit_file。

参数：
- path: 文件的绝对路径（必须位于记忆库根目录内）
- content: 要写入的完整文件内容

文件格式规范：
- 第一行必须是 JSON 元数据，例如：
  {"description":"Alex 的饮食偏好：素食、海鲜过敏、偏爱意大利菜与日料"}
  description 字段要 **keyword-rich**——把文件覆盖的主要实体、话题、立场词
  都列进去（BM25 会部分基于这一行打分），避免笼统如 "饮食"。
- 后续行是正文事实（一行一条）
"""


class WriteFileTool(BaseTool):
    """创建新文件。"""

    def __init__(
        self,
        fs: FileSystemStore,
        *,
        session_ctx: SessionContext | None = None,
    ) -> None:
        self.fs = fs
        self.session_ctx = session_ctx

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的绝对路径（必须位于记忆库根目录内）",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容（第一行应为 JSON 元数据）",
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, path: str, content: str) -> ToolResult:
        ok, rel_or_err = safe_resolve(self.fs.base_path, path)
        if not ok:
            return ToolResult(content=rel_or_err, is_error=True)
        rel = rel_or_err

        if not rel:
            return ToolResult(content="不能写到记忆库根目录本身", is_error=True)

        full = os.path.join(self.fs.base_path, rel)
        # 拦截"覆写非空已有文件"
        if os.path.isfile(full) and os.path.getsize(full) > 0:
            return ToolResult(
                content=(
                    f"目标文件已存在且非空：{path}。"
                    f"write_file 仅用于创建新文件，不允许覆写已有内容。"
                    f"请改用 edit_file 做精确替换/追加。"
                ),
                is_error=True,
            )

        # 递增 turn（写操作计数，用于 vec/graph 的 ingest_turn）
        if self.session_ctx:
            self.session_ctx.next_turn()

        # 直接写入 LLM 给的内容，不做任何前缀注入
        result = self.fs.write_file(rel, content)
        return ToolResult(content=f"已创建 {path}（{len(content)} 字符）；{result}")
