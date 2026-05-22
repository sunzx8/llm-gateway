"""工具基类 — Atomic_Code_T2 方案专用。

设计参考 anthony-agent 的 Claude Code 风格：
- definition() 返回 OpenAI function calling 格式的工具定义
- execute(**kwargs) 异步执行并返回 ToolResult
- 工具不持有 LLM，只持有需要的 store（fs/vec/graph），由外部注入
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """工具执行结果。

    - content: 给 LLM 看的文本结果（成功内容或错误说明）
    - is_error: 是否为错误（True 时 content 是错误信息）
    """

    content: str
    is_error: bool = False

    def to_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """转成可追加进 messages 的一条 OpenAI tool 消息。"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": self.content,
        }


class BaseTool(ABC):
    """所有工具的统一协议。"""

    @abstractmethod
    def definition(self) -> ToolDefinition:
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        ...
