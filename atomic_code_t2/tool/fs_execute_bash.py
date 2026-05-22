"""fs_execute_bash 工具 — 在记忆库根目录执行受限 shell 命令。

仅注册给 consolidate 阶段（重型演进 agent loop）。
工具描述只声明能力边界（白名单 / 禁止项），具体使用场景与策略由
consolidate prompt 引导，避免 schema 与 prompt 职责混淆。
"""

from __future__ import annotations

from storage.file_system_store import FileSystemStore

from .base import BaseTool, ToolDefinition, ToolResult
from .session_context import SessionContext


_TOOL_DESCRIPTION = (
    "在记忆库根目录执行受限的 shell 命令。"
    "允许的命令：cat/ls/head/tail/wc/sort/uniq/find/grep/awk/sed/cut/tr/diff/xargs/"
    "mkdir/touch/mv/cp/rm/chmod/tee/echo/printf/python3/date。"
    "禁止：使用 `..` 进行路径穿越，以及任何 `git` 子命令。"
)


class FsExecuteBashTool(BaseTool):
    """在记忆库根目录执行受限 shell 命令（consolidate 专属）。"""

    def __init__(
        self,
        fs: FileSystemStore,
        *,
        session_ctx: SessionContext | None = None,
        allow_write: bool = True,
    ) -> None:
        self.fs = fs
        self.session_ctx = session_ctx
        self.allow_write = allow_write

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fs_execute_bash",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令（支持管道 | 与 && / ||）",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时时间（秒，默认 15，最大 60）",
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, command: str = "", timeout: int = 15) -> ToolResult:
        if not command or not command.strip():
            return ToolResult(
                content="ERROR: missing required arg 'command'",
                is_error=True,
            )

        try:
            timeout_int = int(timeout)
        except (TypeError, ValueError):
            return ToolResult(
                content=f"ERROR: invalid timeout: {timeout!r}",
                is_error=True,
            )
        timeout_int = max(1, min(timeout_int, 60))

        env_extra: dict[str, str] = {}
        if self.session_ctx and self.session_ctx.session_time:
            env_extra["CURRENT_TIME"] = self.session_ctx.session_time

        result = self.fs.execute_bash(
            command=command,
            timeout=timeout_int,
            allow_write=self.allow_write,
            env_extra=env_extra,
        )
        return ToolResult(
            content=result,
            is_error=isinstance(result, str) and result.startswith("ERROR"),
        )
