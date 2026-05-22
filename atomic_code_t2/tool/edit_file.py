"""edit_file 工具 — 精确字符串替换。

FS 的时间标记完全由 LLM 自己在正文里管理。
代码不做任何前缀注入，只做精确的字符串替换。
"""

from __future__ import annotations

import os

from storage.file_system_store import FileSystemStore

from ._path_safety import safe_resolve
from .base import BaseTool, ToolDefinition, ToolResult
from .session_context import SessionContext


_TOOL_DESCRIPTION = """\
通过**精确字符串匹配**做搜索替换，自带匹配数量验证以避免误改。

核心规则：
- old_string 必须与文件内容**逐字符精确匹配**（包括空白、缩进、换行）
- read_file 输出带行号前缀，old_string **不要包含**这些行号前缀
- expected_replacements 默认为 1；匹配数量与预期不符时直接失败

适用场景：
- 对文件中某段内容做精确修改
- 需要保留时序的更新：把"喜欢跑步 (2023-03)"改为"曾喜欢跑步 (2023-03); 后改为游泳 (2023-09)"
- 在文末追加新行：old_string=最后一行完整内容，new_string=最后一行+\\n+新内容

不适用场景：
- 创建新文件 / 整体重写 → 用 write_file

参数：
- path: 文件的绝对路径（必须位于记忆库根目录内）
- old_string: 要被替换的原始文本
- new_string: 替换后的新文本
- expected_replacements: 预期替换次数，默认 1
"""


class EditFileTool(BaseTool):
    """精确字符串搜索替换。"""

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
            name="edit_file",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件的绝对路径（必须位于记忆库根目录内）",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的原始文本（精确匹配，不要带 read_file 的行号前缀）",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的新文本",
                    },
                    "expected_replacements": {
                        "type": "integer",
                        "description": "预期替换次数，默认 1。匹配数量不符时操作失败。",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        )

    async def execute(
        self,
        path: str,
        old_string: str,
        new_string: str,
        expected_replacements: int = 1,
    ) -> ToolResult:
        ok, rel_or_err = safe_resolve(self.fs.base_path, path)
        if not ok:
            return ToolResult(content=rel_or_err, is_error=True)
        rel = rel_or_err

        if not rel:
            return ToolResult(content="不能编辑记忆库根目录本身", is_error=True)

        full = os.path.join(self.fs.base_path, rel)
        if not os.path.exists(full):
            return ToolResult(content=f"文件不存在: {path}", is_error=True)
        if not os.path.isfile(full):
            return ToolResult(content=f"不是文件: {path}", is_error=True)
        if old_string == new_string:
            return ToolResult(
                content="old_string 与 new_string 相同，无需替换", is_error=True
            )

        content = self.fs.read_file(rel)
        if content.startswith("ERROR:"):
            return ToolResult(content=content, is_error=True)

        count = content.count(old_string)
        if count == 0:
            return ToolResult(content="old_string 在文件中未找到匹配", is_error=True)
        if count != expected_replacements:
            return ToolResult(
                content=f"预期替换 {expected_replacements} 处，但找到 {count} 处匹配",
                is_error=True,
            )

        # 递增 turn（写操作计数，用于 vec/graph 的 ingest_turn）
        if self.session_ctx:
            self.session_ctx.next_turn()

        # 直接替换，不做任何前缀注入
        new_content = content.replace(old_string, new_string)
        self.fs.write_file(rel, new_content)
        return ToolResult(content=f"已替换 {count} 处（{path}）")
