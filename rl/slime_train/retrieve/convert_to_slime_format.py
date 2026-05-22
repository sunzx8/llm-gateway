#!/usr/bin/env python3
"""Convert retrieve snapshot data to Slime JSONL format (task_loop mode).

When ``MEMORY_RL_APPLY_MODE=task_loop``, the custom_generate function delegates
to ``RetrieveT2AgentLoopTask.run()`` which constructs the system prompt, user
prompt, and tools INTERNALLY from the snapshot state. Therefore the dataset only
needs to provide the MINIMAL input:
  - prompt: the query (question to answer)
  - metadata: snapshot_id, traj_id, query, ground_truth, probes, etc.

The task loop will build the full agent loop prompt (with fs_structure,
vec_collections, graph_schema, etc.) at generate time.

Usage:
    python convert_to_slime_format.py \
        --input /path/to/rl_data.jsonl \
        --output data/rl_train.jsonl \
        --eval_output data/rl_val.jsonl \
        --eval_ratio 0.05
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

from llm_gateway.atomic_t2_agent_loop.retrieve_task import (
    RETRIEVE_T2_AGENT_TOOLS,
)
from llm_gateway.rl.slime_train.memory_rl.probes import (
    normalize_ground_truth,
    select_task_probes,
)


def convert_probe(probe: dict[str, Any], record_meta: dict[str, Any]) -> dict[str, Any]:
    """Convert a single retrieve probe to Slime training format.

    The prompt contains ONLY the query question. The task loop
    (custom_generate → RetrieveT2AgentLoopTask.run()) will:
    1. Load the snapshot from metadata.snapshot_id
    2. Build the full system prompt with fs_structure/vec_collections/graph_schema
    3. Build the user prompt (RETRIEVE_T2_AGENT_USER_TEMPLATE)
    4. Expose the full retrieve tool set (RETRIEVE_T2_AGENT_TOOLS)
    """
    question = probe.get("probe_query", "")

    # prompt: just the query — the only user-provided input for retrieve
    prompt = [{"role": "user", "content": question}]

    ground_truth = normalize_ground_truth(probe)

    return {
        "prompt": prompt,
        "label": None,
        "metadata": {
            "task": "retrieve_t2_agent_loop",
            "query": question,
            "ground_truth": ground_truth,
            "snapshot_id": record_meta.get("snapshot_id", ""),
            "traj_id": record_meta.get("trajectory_id", ""),
            "user_id": record_meta.get("user_id", ""),
            "session_id": record_meta.get("session_id", ""),
            "answerable": probe.get("answerable", True),
            "probe_type": probe.get("probe_type", ""),
            "generation_mode": probe.get("generation_mode", ""),
            "tools": [tool.get("function", {}).get("name", "") for tool in RETRIEVE_T2_AGENT_TOOLS],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert retrieve RL data to Slime JSONL (task_loop mode)")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root (unused in task_loop mode, kept for CLI compat)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Convert: expand each record's retrieve probes into individual samples
    samples: list[dict[str, Any]] = []
    for record in records:
        retrieve_probes = select_task_probes(record, "retrieve")
        if not retrieve_probes:
            continue

        snap_id = record.get("snapshot_id", "")
        traj_id = record.get("trajectory_id", "")
        if not snap_id or not traj_id:
            continue

        record_meta = {
            "snapshot_id": snap_id,
            "trajectory_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
        }

        for probe in retrieve_probes:
            question = probe.get("probe_query", "")
            if not question:
                continue
            samples.append(convert_probe(probe, record_meta))

    print(f"converted {len(samples)} retrieve samples from {len(records)} records")

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
