from __future__ import annotations

from typing import Any


def get_tool_schemas(task: str) -> list[dict[str, Any]]:
    """Return native OpenAI tool schemas for one memory RL task."""
    if task == "ingest":
        from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_TOOLS

        return INGEST_T3_TOOLS
    if task == "consolidate":
        from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_TOOLS

        return CONSOLIDATE_T3_TOOLS
    return []
