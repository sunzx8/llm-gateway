"""Session Context — 贯穿一次 ingest 生命周期的上下文信息。

职责：
- 维护 turn counter（每次写工具调用时递增）
- 为 vec_write 构造 metadata（occurred_at + ingest_time + ingest_turn）
- 为 graph_write 构造 properties（ingest_time + ingest_turn）
- 持有 embedder 引用（graph node/edge 自动计算向量）

不再负责 FS 时间前缀——FS 的时间标记完全由 LLM 自己在正文里管理。
"""

from __future__ import annotations

import re
from typing import Any


class SessionContext:
    """一次 ingest 生命周期内的共享上下文。

    Attributes:
        session_time: 本次摄入的对话时间（格式如 "2023-07-08 11:12:14, Sat"）
        embedder: 可选的 embedding 接口（有则为 graph node/edge 自动算向量）
        _turn_counter: 写操作计数器（自增）
    """

    def __init__(
        self,
        session_time: str = "",
        embedder: Any = None,
    ) -> None:
        self.session_time = session_time
        self.embedder = embedder
        self._turn_counter: int = 0

    def next_turn(self) -> int:
        """递增并返回当前 turn 序号（从 1 开始）。"""
        self._turn_counter += 1
        return self._turn_counter

    @property
    def turn(self) -> int:
        """当前 turn 序号（不递增）。"""
        return self._turn_counter

    # ------------------------------------------------------------------
    # Vec metadata 构造（对齐 T3 的时间提取 + ingest_time/ingest_turn）
    # ------------------------------------------------------------------

    def build_vec_metadata(self, text: str) -> dict[str, Any]:
        """为 vec_write 构造 metadata dict。

        包含：
        - occurred_at: 从 text 提取的时间（可能是单值或数组）
        - ingest_time: session_time 去掉星期
        - ingest_turn: 当前 turn 序号
        """
        ts = self.session_time.split(",")[0].strip() if "," in self.session_time else self.session_time.strip()

        # 提取 occurred_at（对齐 T3 ingest_t3.py:719-724）
        time_matches = re.findall(
            r'[(\[](\d{4}(?:-\d{2}(?:-\d{2})?)?)(?:,\s*\w+)?[)\]]', text
        )
        seen_times: list[str] = []
        for t in time_matches:
            if t not in seen_times:
                seen_times.append(t)
        occurred_at: list[str] | str = (
            seen_times if len(seen_times) > 1
            else seen_times[0] if seen_times
            else ""
        )

        return {
            "occurred_at": occurred_at,
            "ingest_time": ts,
            "ingest_turn": self._turn_counter,
        }

    # ------------------------------------------------------------------
    # Graph properties 构造（对齐 T3 的 ingest_time/ingest_turn）
    # ------------------------------------------------------------------

    def build_graph_properties(self) -> dict[str, Any]:
        """为 graph node/edge 的 properties 构造时间字段。"""
        ts = self.session_time.split(",")[0].strip() if "," in self.session_time else self.session_time.strip()
        return {
            "ingest_time": ts,
            "ingest_turn": self._turn_counter,
        }
