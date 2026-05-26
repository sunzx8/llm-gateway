from __future__ import annotations

import os
from typing import Any


def get_tool_schemas(task: str) -> list[dict[str, Any]]:
    """Return native OpenAI tool schemas for one memory RL task."""
    task_version = os.environ.get("MEMORY_RL_TASK_VERSION", "")
    if task_version == "t2_agent_loop":
        if task == "ingest":
            from llm_gateway.atomic_t2_agent_loop.ingest_task import INGEST_T2_AGENT_TOOLS

            return INGEST_T2_AGENT_TOOLS
        if task == "consolidate":
            from llm_gateway.atomic_t2_agent_loop.consolidate_task import CONSOLIDATE_T2_AGENT_TOOLS

            return CONSOLIDATE_T2_AGENT_TOOLS
        if task == "retrieve":
            from llm_gateway.atomic_t2_agent_loop.retrieve_task import RETRIEVE_T2_AGENT_TOOLS

            return RETRIEVE_T2_AGENT_TOOLS

    if task == "ingest":
        from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_TOOLS

        return INGEST_T3_TOOLS
    if task == "consolidate":
        from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_TOOLS

        return CONSOLIDATE_T3_TOOLS
    return []
