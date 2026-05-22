"""检索方案抽象基类。

一个"检索方案"就是用原子操作（atomic.py）拼出的 retrieve workflow。
未来 LLM 演进可以生成新的子类放到 `<fs.base_path>/.codegen/retrieve/<plan_name>.py`，
通过 `loader.load_plan_class()` 动态加载并执行。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface


class BasePlan(ABC):
    """检索方案基类。

    子类只需实现 `run` 方法，返回格式化后的 memory context 字符串。

    Attributes:
        fs: FileSystemStore 实例
        vec: VectorStoreBase 实例
        graph: GraphStoreBase 实例
        llm: LLMInterface 实例
    """

    plan_name: str = "base"

    def __init__(
        self,
        fs: FileSystemStore,
        vec: VectorStoreBase,
        graph: GraphStoreBase,
        llm: LLMInterface,
    ) -> None:
        self.fs = fs
        self.vec = vec
        self.graph = graph
        self.llm = llm

    @abstractmethod
    async def run(
        self,
        query: str,
        session_time: str = "",
        **kwargs: Any,
    ) -> str:
        """执行检索方案。

        Args:
            query: 用户查询文本。
            session_time: 当前会话时间（用于相对时间词解析），格式 "YYYY-MM-DD HH:MM:SS, Day"。
            **kwargs: 扩展参数，子类可自由约定。

        Returns:
            格式化后的 memory context 字符串；无相关记忆时返回空串。
        """
        ...
