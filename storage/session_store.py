"""Session 存储 — 管理会话消息与记忆摄入进度。

SessionStore 负责：
1. 根据 user_id + messages 哈希生成唯一 session_id
2. 维护当前会话的 messages 字典（key 为全局序号），通过 messages 状态推算摄入进度
3. 提供操作锁防止并发冲突
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any


class SessionStore:
    """会话存储，跟踪消息列表与记忆摄入进度。

    Attributes:
        session_id: 会话唯一标识，格式为 ``{messages_md5}``。
        user_id: 用户标识。
        messages: 当前未 ingest 的消息字典，key 为全局序号字符串（从 1 开始）。
        _lock: 异步操作锁，防止并发冲突。
        latest_memory: 最新的记忆信息。
    """

    def __init__(self, user_id: str , session_id: str) -> None:
        self.user_id: str = user_id
        self.session_id: str = session_id
        self.messages: dict[str, dict[str, Any]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self.latest_memory: str = ""

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    async def update_messages(
        self,
        full_messages: list[dict[str, Any]],
        duration_ms: int | None = None,
        received_at: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """覆盖 messages 方法 — 根据当前 messages 状态获取新消息片段。

        内部自动获取操作锁，防止并发冲突。
        从 ``uningest_start_index``（通过 messages 推算）开始截取 full_messages 中的新增部分，
        转换为以全局序号为 key 的字典，作为当前 SessionStore 对象的 messages。

        原始 messages 格式:
            [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]
        转换后 SessionStore.messages 格式（key 为全局序号）:
            {"1": {"role": "user", "content": "xxx", "received_at": "2026-05-08 21:20:12"}, ...}

        Args:
            full_messages: 完整的对话消息列表。
            duration_ms: 本次请求的耗时（毫秒），从接收到question到获取到response的总耗时。
                         若提供，将记录到最新一条 assistant 消息中。
            received_at: 消息接收时间字符串（格式 yyyy-mm-dd HH:MM:SS）。
                         若提供则使用该值，否则使用当前时间。

        Returns:
            更新后的带序号的消息字典。
        """
        async with self._lock:
            # 使用调用方传入的时间，若未传入则使用当前时间
            now_str = received_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 为最新的 assistant 消息记录请求耗时和接收时间
            if self.messages:
                max_key = str(max(int(k) for k in self.messages))
                latest_msg = self.messages[max_key]
                if latest_msg.get("role") == "assistant":
                    latest_msg["duration_ms"] = duration_ms
                    latest_msg["received_at"] = now_str
            return self.messages

    async def clear_ingested_messages(self, clear_index: int) -> None:
        """清理已 ingest 的 messages。

        内部自动获取操作锁，防止并发冲突。
        删除 SessionStore.messages 中全局序号小于等于 clear_index 的条目。
        剩余条目保留原有全局序号不变。

        Args:
            clear_index: 要清理的最大全局序号（含），删除该序号及之前的所有消息。
        """
        async with self._lock:
            # 删除全局序号 <= clear_index 的条目
            keys_to_remove = [k for k in self.messages if int(k) <= clear_index]
            for k in keys_to_remove:
                del self.messages[k]

    async def get_pending_ingest_messages(self) -> tuple[list[dict[str, Any]], int]:
        """获取需要摄入记忆的消息片段。

        内部自动获取操作锁，防止并发冲突。
        返回当前 SessionStore.messages 转换为原始列表格式，以及当前最大的全局序号值。
        该方法不会修改 messages 内容。

        Returns:
            tuple: (messages_list, max_index)
                - messages_list: 按全局序号排序的消息列表，格式为
                  [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]
                - max_index: 当前 messages 中最大的全局序号值，若为空则返回 0。
        """
        async with self._lock:
            if not self.messages:
                return [], 0
            max_index = max(int(k) for k in self.messages)
            # 按全局序号排序，转换为列表
            messages_list = [
                msg for _, msg in sorted(self.messages.items(), key=lambda x: int(x[0]))
            ]
            return messages_list, max_index

    def get_messages_as_list(self) -> list[dict[str, Any]]:
        """将当前 messages 转换为原始列表格式。

        按全局序号排序，返回 [{"role": "user", "content": "xxx"}, ...] 形式的列表。
        返回结果不包含内部附加的 duration_ms 等元数据字段，保持标准消息协议。
        该方法不会修改 messages 内容，也不获取锁（同步方法）。

        Returns:
            按全局序号排序的消息列表。
        """
        if not self.messages:
            return []
        # 过滤掉内部元数据字段，保持标准消息协议
        _internal_fields = {"duration_ms", "received_at"}
        return [
            {k: v for k, v in msg.items() if k not in _internal_fields}
            for _, msg in sorted(self.messages.items(), key=lambda x: int(x[0]))
        ]

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def uningest_start_index(self) -> int:
        """未 ingest 记忆的起始序号（只读），直接从 messages 推算。"""
        return self._calc_uningest_start_index()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _calc_uningest_start_index(self) -> int:
        """根据 messages 的 key 推算未 ingest 记忆的起始序号。

        如果 messages 为空，说明所有消息都已被 ingest，返回 0（无待处理消息）。
        如果 messages 非空，最小 key - 1 即为已 ingest 的消息数量（即起始偏移）。

        Returns:
            未 ingest 记忆的起始序号（相对于完整对话）。
        """
        if not self.messages:
            return 0
        # messages 的 key 为全局序号，最小 key - 1 = 已 ingest 的消息数
        return min(int(k) for k in self.messages) - 1

    # ------------------------------------------------------------------
    # 辅助 / 调试
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SessionStore(session_id={self.session_id!r}, "
            f"user_id={self.user_id!r}, "
            f"messages_count={len(self.messages)}, "
            f"uningest_start_index={self.uningest_start_index})"
        )


# ---------------------------------------------------------------------------
# 全局 Session 管理器
# ---------------------------------------------------------------------------

class SessionManager:
    """全局 Session 管理器 — 管理所有用户的 SessionStore 实例。

    数据结构: ``{user_id: {session_id: SessionStore}}``

    提供按 user_id + session_id 获取、创建、删除 SessionStore 的能力，
    内部使用锁保证并发安全。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, SessionStore]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_session(self, user_id: str, session_id: str) -> SessionStore | None:
        """获取指定用户的指定 session。

        Args:
            user_id: 用户标识。
            session_id: 会话标识。

        Returns:
            对应的 SessionStore 实例，若不存在则返回 None。
        """
        async with self._lock:
            user_sessions = self._sessions.get(user_id)
            if user_sessions is None:
                return None
            return user_sessions.get(session_id)

    async def get_or_create_session(
        self, user_id: str, session_id: str
    ) -> SessionStore:
        """获取或创建指定用户的指定 session。

        若 session 不存在则自动创建并注册。

        Args:
            user_id: 用户标识。
            session_id: 会话标识。

        Returns:
            对应的 SessionStore 实例。
        """
        async with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = {}
            if session_id not in self._sessions[user_id]:
                store = SessionStore(user_id=user_id, session_id=session_id)
                self._sessions[user_id][session_id] = store
            return self._sessions[user_id][session_id]

    async def register_session(self, user_id: str, store: SessionStore) -> None:
        """注册一个已有的 SessionStore 实例。

        Args:
            user_id: 用户标识。
            store: 要注册的 SessionStore 实例。
        """
        async with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = {}
            self._sessions[user_id][store.session_id] = store

    async def remove_session(self, user_id: str, session_id: str) -> bool:
        """移除指定用户的指定 session。

        Args:
            user_id: 用户标识。
            session_id: 会话标识。

        Returns:
            是否成功移除（不存在时返回 False）。
        """
        async with self._lock:
            user_sessions = self._sessions.get(user_id)
            if user_sessions is None or session_id not in user_sessions:
                return False
            del user_sessions[session_id]
            # 如果该用户已无 session，清理用户条目
            if not user_sessions:
                del self._sessions[user_id]
            return True

    async def get_user_sessions(self, user_id: str) -> dict[str, SessionStore]:
        """获取指定用户的所有 session。

        Args:
            user_id: 用户标识。

        Returns:
            该用户所有 session 的字典副本 ``{session_id: SessionStore}``，
            若用户不存在则返回空字典。
        """
        async with self._lock:
            user_sessions = self._sessions.get(user_id, {})
            return dict(user_sessions)

    async def list_users(self) -> list[str]:
        """列出所有有活跃 session 的用户 ID。

        Returns:
            用户 ID 列表。
        """
        async with self._lock:
            return list(self._sessions.keys())

    def __repr__(self) -> str:
        total_sessions = sum(len(s) for s in self._sessions.values())
        return (
            f"SessionManager(users={len(self._sessions)}, "
            f"total_sessions={total_sessions})"
        )


# 全局单例
session_manager = SessionManager()
