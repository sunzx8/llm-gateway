"""工具注册表 — 收集本次 agent loop 可用的工具集合。"""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolResult


class ToolRegistry:
    """简易工具注册表。

    - register / register_many: 注册工具
    - get_definitions: 给 LLM 用的 OpenAI tools 列表
    - execute(name, arguments): 按名称分发执行
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.definition().name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """返回 OpenAI tools 列表（function calling 格式）。"""
        return [
            {"type": "function", "function": tool.definition().model_dump()}
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(content=f"未知工具: {name}", is_error=True)
        try:
            return await tool.execute(**arguments)
        except TypeError as e:
            return ToolResult(content=f"工具 {name} 参数错误: {e}", is_error=True)
        except Exception as e:
            return ToolResult(content=f"工具 {name} 执行异常: {e}", is_error=True)
