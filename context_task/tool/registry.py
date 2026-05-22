"""
工具注册表

管理所有工具实例，提供按条件过滤和统一执行能力。
"""

from typing import Any

from .base_tool import BaseTool


class ToolRegistry:
    """工具注册表，管理所有工具实例。"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册单个工具。"""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        """批量注册工具。"""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool | None:
        """按名称获取工具实例。"""
        return self._tools.get(name)

    def all_tools(self) -> list[BaseTool]:
        """返回所有已注册的工具列表。"""
        return list(self._tools.values())

    def get_schemas(
        self,
        *,
        readonly_only: bool = False,
        categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        backend: str | None = None,
        names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> list[dict]:
        """按条件过滤并生成 LLM schema 列表。

        Args:
            readonly_only: 只返回只读工具。
            categories: 只返回指定分类的工具。
            exclude_categories: 排除指定分类。
            backend: 当前后端类型（如 "pg"），用于决定是否包含后端专属工具。
            names: 只返回指定名称的工具。
            exclude_names: 排除指定名称的工具。
        """
        schemas = []
        for tool in self._tools.values():
            if readonly_only and not tool.is_readonly:
                continue
            if categories and tool.category not in categories:
                continue
            if exclude_categories and tool.category in exclude_categories:
                continue
            if tool.requires_backend and tool.requires_backend != backend:
                continue
            if names and tool.name not in names:
                continue
            if exclude_names and tool.name in exclude_names:
                continue
            schemas.append(tool.to_llm_schema())
        return schemas

    async def execute(self, name: str, args: dict[str, Any], **deps) -> str:
        """按名称执行工具，deps 透传给 tool.execute。

        Args:
            name: 工具名称。
            args: 工具参数。
            **deps: 执行依赖（fs, vec, graph, session_id 等）。

        Returns:
            工具执行结果字符串。

        Raises:
            如果工具不存在，返回 "Unknown tool: {name}"。
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return await tool.execute(args, **deps)


# 全局注册表实例
default_registry = ToolRegistry()
