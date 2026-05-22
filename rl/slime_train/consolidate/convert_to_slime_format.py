#!/usr/bin/env python3
"""Convert T3 query-probe data into consolidate/evolve RL samples.

Preferred input is the ``rl_data_test_2`` style schema where every record has
``probes_by_task.consolidate``. For older ingest-only datasets, the converter can
fallback to adjacent ingest probes as a best-effort evolution objective.
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

from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_SYSTEM_PROMPT, CONSOLIDATE_T3_TOOLS
from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes

OUTPUT_INSTRUCTION = """

## RL Output Format

You cannot call native tools in this training environment. Output one JSON object:

{
  "tool_calls": [
    {"tool": "fs_write", "arguments": {"path": "...", "content": "..."}},
    {"tool": "fs_append", "arguments": {"path": "...", "content": "..."}},
    {"tool": "vec_add", "arguments": {"collection": "...", "items": [{"text": "...", "metadata": {...}}]}},
    {"tool": "graph_add_node", "arguments": {"node_id": "...", "label": "...", "properties": {...}}},
    {"tool": "graph_add_edge", "arguments": {"source": "...", "target": "...", "relation": "...", "properties": {...}}},
    {"tool": "finish", "arguments": {"summary": "..."}}
  ]
}

Your edits will be evaluated by hidden query probes before and after evolution.
Optimize retrieval correctness without adding unsupported facts or deleting source evidence.
"""


def summarize_env(loaded) -> str:
    fs_files = loaded.fs.list_files()
    parts = [
        "## Current Memory State",
        "",
        f"### File System Structure ({len(fs_files)} files):",
        loaded.fs.tree(max_depth=4),
        "",
        "### File Contents Preview:",
    ]
    for path in fs_files[:20]:
        content = loaded.fs.read_file(path)
        if len(content) > 1600:
            content = content[:1600] + "\n... (truncated)"
        parts.append(f"\n#### {path}\n{content}")
    if not fs_files:
        parts.append("(no files)")
    parts.extend([
        "",
        "### Vector DB Stats:",
        json.dumps(loaded.vec.get_stats(), ensure_ascii=False),
        "",
        "### Graph DB Stats:",
        json.dumps(loaded.graph.get_stats(), ensure_ascii=False),
    ])
    return "\n".join(parts)


def convert_record(record: dict[str, Any], session: SnapshotSession, probes: list[dict[str, Any]], probe_source: str) -> dict[str, Any]:
    snapshot_id = record.get("snapshot_id", "")
    traj_id = record.get("trajectory_id", "")
    with session.load(traj_id=traj_id, snapshot_id=snapshot_id) as loaded:
        state_summary = summarize_env(loaded)

    user_prompt = f"""{state_summary}

---

Review the memory state and emit conservative consolidate/evolution tool calls
that improve future query-probe retrieval. Focus on routing/index notes, vector
coverage, graph connectivity, deduplication, and preserving provenance.
"""
    return {
        "prompt": [
            {"role": "system", "content": CONSOLIDATE_T3_SYSTEM_PROMPT + OUTPUT_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        "label": None,
        "metadata": {
            "task": "consolidate_t3",
            "snapshot_id": snapshot_id,
            "traj_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
            "probes": probes,
            "probe_source": probe_source,
            "tools": [tool.get("function", {}).get("name", "") for tool in CONSOLIDATE_T3_TOOLS],
        },
    }


def build_samples(records: list[dict[str, Any]], session: SnapshotSession) -> list[dict[str, Any]]:
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
            samples.append(convert_record(record, session, probes, "query_probes:consolidate"))

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
            samples.append(convert_record(current, session, probes, "adjacent_ingest_query_probes:fallback"))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert T3 consolidate RL data to Slime JSONL")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root; defaults to input parent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    data_root = Path(args.data_root) if args.data_root else input_path.parent
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    session = SnapshotSession(str(data_root), enable_git=False)
    samples = build_samples(records, session)

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
