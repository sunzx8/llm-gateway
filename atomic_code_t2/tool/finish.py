"""finish 工具 — agent loop 退出信号。"""

from __future__ import annotations

from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
标记任务结束。**所有读写完成后**调用此工具退出 agent loop。

参数：
- summary: 一句话总结本次做了什么（写了哪些信息 / 召回了什么 / 决定了什么）。
  如果对话里没有任何值得写入的信息，summary 写 `no valuable information`。
"""


class FinishTool(BaseTool):
    """退出 agent loop。

    实际的"退出循环"逻辑由 agent_loop 在解析 tool_call 名字时处理；
    本工具的 execute 仅返回一个确认消息，便于轨迹查看。
    """

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="finish",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "一句话总结。无可写信息时填 'no valuable information'",
                    },
                },
                "required": ["summary"],
            },
        )

    async def execute(self, summary: str = "") -> ToolResult:
        return ToolResult(content=f"finish: {summary}")
