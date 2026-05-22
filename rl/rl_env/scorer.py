"""Scorer abstractions for RLVR reward computation.

This module intentionally defines **interfaces only**. The concrete
scoring rules (regex gold-fact matching, tool-call penalties, etc.)
are implemented by training-side packages — we only commit to the
shape of the hook here so that :class:`llm_gateway.rl.rl_env.MemoryEnv` can
call them uniformly.

Reference reward structure (for context; not enforced here)
-----------------------------------------------------------
Phase 1 main reward = Python-scorer score on the per-step gold question.

Supplementary per-step penalties / bonuses (all aggregated elsewhere):

    1. tool error                              -1
    2. each successful tool call               +0.25
    3. empty search result (bad query)         -0.15
    4. full read (rows > 100)                  -2
    5. > 6 tool calls                          -0.3 / each
    6. all gold regexes hit (home run)         +20
    7. context noise                           -0.05 * log(token_count)
    8. space usage                             -0.05 * log(total_space)

Phase-specific differences only affect *how* the gold question is
generated, not how the scorer signature looks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_gateway.rl.rl_env._models import TaskResult
    from llm_gateway.rl.rl_env.snapshot import Snapshot


@dataclass
class ScoreResult:
    """Structured reward output."""

    reward: float = 0.0
    main: float = 0.0
    supplementary: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class BaseScorer(ABC):
    """Abstract scorer hook.

    One scorer is associated with a single (pre_snapshot → action →
    post_snapshot) transition.
    """

    phase: str = "base"

    @abstractmethod
    def score(
        self,
        *,
        pre_snapshot: "Snapshot | None",
        post_snapshot: "Snapshot | None",
        task_result: "TaskResult",
        gold: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Compute reward for a single env step."""


# ---------------------------------------------------------------------------
# Placeholder subclasses — one per RL phase, per the design doc
# ---------------------------------------------------------------------------


class _NullScorer(BaseScorer):
    """Shared null implementation used until real rules are wired in."""

    def score(
        self,
        *,
        pre_snapshot: "Snapshot | None",
        post_snapshot: "Snapshot | None",
        task_result: "TaskResult",
        gold: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> ScoreResult:
        return ScoreResult(
            reward=0.0,
            main=0.0,
            supplementary=0.0,
            breakdown={},
            details={
                "phase": self.phase,
                "note": "scorer not implemented — returning zero reward",
            },
        )


class IngestScorer(_NullScorer):
    """Phase 2 — ingest-side scorer (placeholder)."""

    phase = "ingest"


class EvolveScorer(_NullScorer):
    """Phase 3 — consolidation/evolution scorer (placeholder)."""

    phase = "evolve"


class ConsumeScorer(_NullScorer):
    """Phase 1 — consumption (retrieval) scorer (placeholder)."""

    phase = "consume"


__all__ = [
    "BaseScorer",
    "ScoreResult",
    "IngestScorer",
    "EvolveScorer",
    "ConsumeScorer",
]
