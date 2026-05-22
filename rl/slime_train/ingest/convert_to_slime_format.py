#!/usr/bin/env python3
"""Convert ingest snapshot data to Slime JSONL format (task_loop mode).

When ``MEMORY_RL_APPLY_MODE=task_loop``, the custom_generate function delegates
to ``IngestT2AgentLoopTask.run()`` which constructs the system prompt, user prompt,
and tools INTERNALLY from the snapshot + metadata. Therefore the dataset only needs
to provide the MINIMAL input:
  - prompt: the pending_messages (conversation to ingest)
  - metadata: snapshot_id, traj_id, session_time, pending_messages, probes, etc.

The ``prompt`` field is a simple list of messages representing the conversation
to process. The task loop will build the full agent loop prompt at generate time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.atomic_t2_agent_loop.ingest_task import (
    INGEST_T2_AGENT_TOOLS,
)
from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a single ingest record to Slime training format.

    The prompt contains ONLY the pending_messages (the conversation to be ingested).
    The task loop (custom_generate → IngestT2AgentLoopTask.run()) will:
    1. Load the snapshot from metadata.snapshot_id
    2. Build the full system prompt (INGEST_T2_AGENT_SYSTEM_PROMPT)
    3. Build the user prompt with memory state + conversation
    4. Expose the full tool set (INGEST_T2_AGENT_TOOLS)
    """
    snapshot_id = record.get("snapshot_id", "")
    traj_id = record.get("trajectory_id", "")
    pending_messages = record.get("pending_messages", [])

    # prompt: just the conversation messages that need to be ingested
    # This is the key input for the ingest task — everything else comes from the snapshot
    prompt = [{"role": "user", "content": msg.get("content", "")} if msg.get("role") == "user"
              else {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}
              for msg in pending_messages]

    probes = select_task_probes(record, "ingest", fallback_all_when_untyped=True)
    return {
        "prompt": prompt,
        "label": None,
        "metadata": {
            "task": "ingest_t2_agent_loop",
            "snapshot_id": snapshot_id,
            "traj_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
            "session_time": record.get("generated_at", ""),
            "pending_messages": pending_messages,
            "probes": probes,
            "probe_source": "query_probes:ingest",
            "tools": [tool.get("function", {}).get("name", "") for tool in INGEST_T2_AGENT_TOOLS],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ingest RL data to Slime JSONL (task_loop mode)")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root (unused in task_loop mode, kept for CLI compat)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [
        convert_record(r)
        for r in records
        if r.get("rl_task_type", "ingest") == "ingest"
        and select_task_probes(r, "ingest", fallback_all_when_untyped=True)
    ]

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    eval_count = int(len(samples) * args.eval_ratio) if args.eval_output else 0
    eval_samples = samples[:eval_count]
    train_samples = samples[eval_count:]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if args.eval_output:
        Path(args.eval_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.eval_output, "w", encoding="utf-8") as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote train={len(train_samples)} eval={len(eval_samples)}")


if __name__ == "__main__":
    main()
