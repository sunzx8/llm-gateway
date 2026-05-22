"""
Tool 基类定义

所有工具继承 BaseTool，实现自包含的定义 + 执行逻辑。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """工具基类，每个工具继承此类实现自包含的定义+执行。"""

    # ── 元信息（子类必须覆盖） ──
    name: str = ""
    """工具名，如 "fs_read" """

    description: str = ""
    """工具描述（给 LLM 看）"""

    parameters: dict = {}
    """JSON Schema 格式的参数定义"""

    # ── 权限与后端标记 ──
    is_readonly: bool = True
    """是否只读"""

    requires_backend: str | None = None
    """依赖的后端："pg" / None"""

    category: str = "base"
    """工具分类："fs" / "vec" / "graph" / "sql" / "session" / "special" / "code" """

    def to_llm_schema(self) -> dict:
        """生成 OpenAI function calling 格式的 JSON schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, args: dict[str, Any], **deps) -> str:
        """执行工具逻辑。

        Args:
            args: LLM 传入的工具参数（已解析为 dict）。
            **deps: 执行依赖，由 task 按需传入，如：
                    fs=..., vec=..., graph=..., session_id=..., allow_write=...

        Returns:
            工具执行结果的字符串表示。
        """
        ...
