"""记忆摄入抽象基类。

本文件是合法的 Python 模块，可以直接 import、lint、测试。
LLM 只需生成一个继承本类的子类，实现 ingest_memory 抽象方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from storage.stores_base import GraphStoreBase, VectorStoreBase
from storage.file_system_store import FileSystemStore
from utils.memory_llm_interface import LLMInterface
from debug.logged_mixin import LoggedMethodsMixin

class BaseMemoryIngestor(ABC,LoggedMethodsMixin):
    """记忆摄入基类。

    子类只需实现 ingest_memory 方法。
    通过 self.fs / self.vec / self.graph / self.llm / self.memory_base 访问依赖。

    Attributes:
        fs: FileSystemStore 实例
        vec: VectorStoreBase 实例（可能为 None）
        graph: GraphStoreBase 实例（可能为 None）
        llm: LLMInterface 实例（可能为 None）
        memory_base: 记忆库根目录路径
    """
    _logged_methods = ["ingest_memory"]

    def __init__(
        self,
        fs: FileSystemStore,
        vec: VectorStoreBase = None,
        graph: GraphStoreBase = None,
        llm: LLMInterface = None,
        memory_base: str = "",
    ):
        self.fs: FileSystemStore = fs
        self.vec: VectorStoreBase = vec
        self.graph: GraphStoreBase = graph
        self.llm: LLMInterface = llm
        self.memory_base: str = memory_base

    @abstractmethod
    async def ingest_memory(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> str:
        """从对话消息中提取有价值的信息并写入记忆库。

        Args:
            messages: 待摄入的对话消息列表，
                      格式 [{"role": "user"|"assistant", "content": "..."}]
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            摄入结果的描述字符串（如摄入了哪些信息、写入了哪些存储）。
            返回空字符串表示无需摄入。
        """
        ...
