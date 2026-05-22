"""Base benchmark interface.

All benchmark adapters convert their native format into a unified event stream
that can be replayed through the memory system.

统一数据格式对齐 trajectory-schema.md：
- UserTrajectory: 按 user 组织的 sessions + questions
- 各 adapter 的 load_data() 返回 Conversation[]（兼容旧格式）
- to_trajectories() 将 Conversation[] 转换为 UserTrajectory[]（统一格式）
- from_trajectories() 将 UserTrajectory[] 展开回 Conversation[]（供 runner 使用）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Conversation:
    """A multi-session conversation from a benchmark dataset."""

    conv_id: str
    sessions: list[Session] = field(default_factory=list)
    questions: list[QAPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A single session within a conversation."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Questions to answer AFTER this session is ingested (for interleaved ingest-query mode).
    # When non-empty, the runner will answer these questions after ingesting this session,
    # instead of answering all questions after all sessions are ingested.
    questions: list[Any] = field(default_factory=list)  # list[QAPair]


# ---------------------------------------------------------------------------
# 统一数据格式：UserTrajectory（对齐 trajectory-schema.md）
# ---------------------------------------------------------------------------

@dataclass
class UserTrajectory:
    """按 user 组织的 trajectory，对齐 trajectory-schema.md 的顶层结构。

    核心思想：1 个 UserTrajectory = 1 个用户的全部 sessions + questions。
    不同 benchmark 的原始数据格式各异，但都可以转换为这个统一结构。

    字段说明：
    - user_id: 用户唯一标识（对应 trajectory-schema 的 user_id）
    - sessions: 该用户的全部会话，按时间顺序排列
    - questions: 该用户的全部评测问题
    - metadata: 聚合元数据（benchmark 名称、persona 信息等）
    - conversations: 原始 Conversation 列表（保留以便 runner 兼容处理）
    """

    user_id: str
    sessions: list[Session] = field(default_factory=list)
    questions: list[QAPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # 保留原始 conversations 引用，供 runner 兼容处理（如 interleaved 模式、context 隔离等）
    conversations: list[Conversation] = field(default_factory=list)

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def total_messages(self) -> int:
        return sum(len(s.messages) for s in self.sessions)


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class QAPair:
    """A question-answer pair from a benchmark."""

    question: str
    reference_answer: str
    question_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    question_for_retrieval: str = ""  # Pure question without options; used for memory retrieval


class BaseBenchmark(ABC):
    """Abstract base for benchmark adapters.

    Each benchmark adapter:
    1. Loads data from its native format
    2. Converts conversations into sessions with messages
    3. Provides QA pairs for evaluation
    4. Implements scoring logic
    """

    name: str = "base"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @abstractmethod
    def load_data(self) -> list[Conversation]:
        """Load benchmark dataset and return structured conversations."""
        ...

    @abstractmethod
    def get_questions(self, conversation: Conversation) -> list[QAPair]:
        """Extract QA pairs for a given conversation."""
        ...

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        qa_pairs: list[QAPair] | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            predictions: Model-generated answers.
            references: Ground-truth answers.
            qa_pairs: Optional original QAPair objects (for category-aware evaluation).

        Returns:
            Dict of metric_name → score.
        """
        ...

    # ------------------------------------------------------------------
    # 单条 QA 正确性判断
    # ------------------------------------------------------------------

    def check_correctness(
        self,
        predictions: list[str],
        references: list[str],
        qa_pairs: list[QAPair] | None = None,
    ) -> list[bool]:
        """判断每个 QA 是否回答正确，返回布尔列表。

        默认实现：调用 evaluate 方法计算整体准确率，
        子类应覆写此方法以提供逐条判断逻辑。

        Args:
            predictions: 模型生成的答案列表。
            references: 参考答案列表。
            qa_pairs: 可选的 QAPair 对象列表。

        Returns:
            与 predictions 等长的布尔列表，True 表示该 QA 回答正确。
        """
        # 默认实现：全部标记为 False（子类应覆写）
        return [False] * len(predictions)

    # ------------------------------------------------------------------
    # 统一格式转换：Conversation[] ↔ UserTrajectory[]
    # ------------------------------------------------------------------

    def to_trajectories(self, conversations: list[Conversation]) -> list[UserTrajectory]:
        """将 Conversation[] 转换为 UserTrajectory[]（按 user_id 聚合）。

        默认实现：按 metadata["user_id"] 分组，同一 user 的 conversations
        合并为一个 UserTrajectory。sessions 按原始顺序拼接，questions 合并。

        各 adapter 可以覆写此方法做特殊处理（如 PersonaMem 的 context 隔离）。

        Args:
            conversations: load_data() 返回的原始 Conversation 列表。

        Returns:
            按 user 聚合的 UserTrajectory 列表。
        """
        # 按 user_id 分组（保持插入顺序）
        user_groups: OrderedDict[str, list[Conversation]] = OrderedDict()
        for conv in conversations:
            uid = conv.metadata.get("user_id", conv.conv_id)
            user_groups.setdefault(uid, []).append(conv)

        trajectories: list[UserTrajectory] = []
        for uid, convs in user_groups.items():
            traj = UserTrajectory(
                user_id=uid,
                metadata={
                    "benchmark": self.name,
                    "num_conversations": len(convs),
                    "conv_ids": [c.conv_id for c in convs],
                },
                conversations=convs,
            )
            # 合并所有 sessions 和 questions
            for conv in convs:
                traj.sessions.extend(conv.sessions)
                traj.questions.extend(conv.questions)

            trajectories.append(traj)

        return trajectories

    @staticmethod
    def from_trajectories(trajectories: list[UserTrajectory]) -> list[Conversation]:
        """将 UserTrajectory[] 展开回 Conversation[]（供 runner 兼容使用）。

        直接返回每个 trajectory 中保留的原始 conversations 引用。

        Args:
            trajectories: to_trajectories() 返回的 UserTrajectory 列表。

        Returns:
            展开后的 Conversation 列表（保持原始顺序）。
        """
        conversations: list[Conversation] = []
        for traj in trajectories:
            conversations.extend(traj.conversations)
        return conversations

    def subset(self, conversations: list[Conversation], n: int) -> list[Conversation]:
        """Return first n conversations (for quick testing)."""
        return conversations[:n]
