"""记忆消费抽象基类。

本文件是合法的 Python 模块，可以直接 import、lint、测试。
LLM 只需生成一个继承本类的子类，实现 retrieve_memory 抽象方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from storage.stores_base import GraphStoreBase, VectorStoreBase
from storage.file_system_store import FileSystemStore
from utils.memory_llm_interface import LLMInterface
from debug.logged_mixin import LoggedMethodsMixin

class BaseMemoryConsumer(ABC, LoggedMethodsMixin):
    """记忆消费基类。

    子类只需实现 retrieve_memory 方法。
    通过 self.fs / self.vec / self.graph / self.llm 访问依赖。

    Attributes:
        fs: FileSystemStore 实例
        vec: VectorStoreBase 实例（可能为 None）
        graph: GraphStoreBase 实例（可能为 None）
        llm: LLMInterface 实例（可能为 None）
    """
    _logged_methods = ["retrieve_memory"]
    
    def __init__(
        self,
        fs: FileSystemStore,
        vec: VectorStoreBase,
        graph: GraphStoreBase,
        llm: LLMInterface,
    ):
        self.fs: FileSystemStore = fs
        self.vec: VectorStoreBase = vec
        self.graph: GraphStoreBase = graph
        self.llm: LLMInterface = llm

    @abstractmethod
    async def retrieve_memory(
        self,
        query: str,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> str:
        """根据用户查询从记忆库中检索相关记忆。

        Args:
            query: 用户的原始查询问题
            messages: 当前对话消息列表
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            格式化的记忆检索结果字符串。
            返回空字符串表示无相关记忆。
        """
        ...


