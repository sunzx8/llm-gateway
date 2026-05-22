#!/usr/bin/env python3
"""Convert consolidate snapshot data to Slime JSONL format (task_loop mode).

When ``MEMORY_RL_APPLY_MODE=task_loop``, the custom_generate function delegates
to ``ConsolidateT2AgentLoopTask.run()`` which constructs the system prompt, user
prompt, and tools INTERNALLY from the snapshot state. Therefore the dataset only
needs to provide MINIMAL metadata:
  - prompt: empty or a simple trigger (consolidate is triggered, not user-driven)
  - metadata: snapshot_id, traj_id, probes, etc.

The task loop will build the full agent loop prompt (with complete memory state,
C1-C11 checklist, etc.) at generate time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.atomic_t2_agent_loop.consolidate_task import (
    CONSOLIDATE_T2_AGENT_TOOLS,
)
from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes


def convert_record(record: dict[str, Any], probes: list[dict[str, Any]], probe_source: str) -> dict[str, Any]:
    """Convert a single consolidate record to Slime training format.

    The prompt is minimal — just a trigger signal. The task loop
    (custom_generate → ConsolidateT2AgentLoopTask.run()) will:
    1. Load the snapshot from metadata.snapshot_id
    2. Read the full memory state (FS files, Vec entries, Graph edges)
    3. Build the complete system + user prompt with C1-C11 checklist
    4. Expose the full consolidate tool set
    """
    snapshot_id = record.get("snapshot_id", "")
    traj_id = record.get("trajectory_id", "")

    # prompt: minimal trigger — consolidate doesn't have user-provided input
    # The task loop builds everything from the snapshot state
    prompt = [{"role": "user", "content": "Perform memory consolidation."}]

    return {
        "prompt": prompt,
        "label": None,
        "metadata": {
            "task": "consolidate_t2_agent_loop",
            "snapshot_id": snapshot_id,
            "traj_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
            "probes": probes,
            "probe_source": probe_source,
            "tools": [tool.get("function", {}).get("name", "") for tool in CONSOLIDATE_T2_AGENT_TOOLS],
        },
    }


def build_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build consolidate samples from explicit consolidate query probes first.

    ``rl_data_test_2`` provides ``probes_by_task.consolidate`` on the same
    snapshot record. Legacy ingest-only data has no consolidate probes; for that
    case we keep a fallback where record ``i + 1``'s snapshot approximates the
    state after record ``i`` was ingested, and record ``i``'s ingest probes are
    used as hidden query objectives.
    """
    samples: list[dict[str, Any]] = []
    for record in records:
        probes = select_task_probes(record, "consolidate")
        if probes:
            samples.append(convert_record(record, probes, "query_probes:consolidate"))

    if samples:
        return samples

    by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_traj[rec.get("trajectory_id", "")].append(rec)

    for traj_id, traj_records in by_traj.items():
        traj_records.sort(key=lambda r: (r.get("session_index", 0), r.get("batch_index", 0), r.get("step_index", 0)))
        for prev, current in zip(traj_records, traj_records[1:]):
            probes = select_task_probes(prev, "ingest", fallback_all_when_untyped=True)
            if not probes:
                continue
            current = {**current, "trajectory_id": traj_id}
            samples.append(convert_record(current, probes, "adjacent_ingest_query_probes:fallback"))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert consolidate RL data to Slime JSONL (task_loop mode)")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root (unused in task_loop mode, kept for CLI compat)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = build_samples(records)

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
