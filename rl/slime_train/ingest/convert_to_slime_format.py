#!/usr/bin/env python3
"""Convert T3 ingest snapshot data to Slime JSONL format.

Input is the ingest snapshot dataset layout produced by
``src/data_gen/ingest_snapshot/run_generate.py``. Each sample asks the policy
to emit JSON-serialized T3 ingest tool calls for ``pending_messages``. Reward
uses hidden query probes selected from ``probes_by_task.ingest`` when available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_SYSTEM_PROMPT, INGEST_T3_TOOLS
from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes

OUTPUT_INSTRUCTION = """

## RL Output Format

You cannot call native tools in this training environment. Instead, output one
JSON object only, after any private reasoning:

{
  "tool_calls": [
    {"tool": "fs_write", "arguments": {"action": "add_file|append|update:<line>|update_meta", "path": "...", "content": "..."}},
    {"tool": "vec_write", "arguments": {"action": "add|update:<id>", "collection": "...", "text": "...", "entry_type": "fact|event|preference|insight"}},
    {"tool": "graph_write", "arguments": {"action": "add_node", "node_id": "...", "label": "..."}},
    {"tool": "graph_write", "arguments": {"action": "add_edge", "source": "...", "target": "...", "relation": "..."}},
    {"tool": "finish", "arguments": {"summary": "..."}}
  ]
}

Your writes will be evaluated by hidden query probes after ingest. Store enough
faithful evidence from the conversation for future retrieval, and do not invent
facts beyond the provided messages.
"""


def summarize_env(loaded) -> str:
    """Build a compact current-memory summary from a loaded snapshot."""
    fs_files = loaded.fs.list_files()
    parts = ["## Current Memory State", "", "### File System Structure:", loaded.fs.tree(max_depth=3)]
    if fs_files:
        parts.extend(["", "### File Contents Preview:"])
        for path in fs_files[:12]:
            content = loaded.fs.read_file(path)
            if len(content) > 1200:
                content = content[:1200] + "\n... (truncated)"
            parts.append(f"\n#### {path}\n{content}")
    else:
        parts.extend(["", "### File Contents Preview:", "(no files)"])
    parts.extend([
        "",
        "### Vector DB Stats:",
        json.dumps(loaded.vec.get_stats(), ensure_ascii=False),
        "",
        "### Graph DB Stats:",
        json.dumps(loaded.graph.get_stats(), ensure_ascii=False),
    ])
    return "\n".join(parts)


def format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant" and msg.get("tool_calls"):
            calls = []
            for tc in msg.get("tool_calls", []):
                calls.append(f"{tc.get('name', 'tool')}({tc.get('arguments', '')})")
            content = (content + " " + " ".join(f"[tool_call: {c}]" for c in calls)).strip()
        elif role == "tool" and msg.get("tool_name"):
            role = f"tool({msg.get('tool_name')})"
        lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


def convert_record(record: dict[str, Any], session: SnapshotSession) -> dict[str, Any]:
    snapshot_id = record.get("snapshot_id", "")
    traj_id = record.get("trajectory_id", "")
    with session.load(traj_id=traj_id, snapshot_id=snapshot_id) as loaded:
        state_summary = summarize_env(loaded)

    user_prompt = f"""{state_summary}

---

## Conversation to Process (session: {record.get('session_id', '')})

{format_messages(record.get('pending_messages', []))}

---

Extract valuable information and write it into the memory system. Compare
against Current Memory State to decide ADD vs UPDATE.
"""

    probes = select_task_probes(record, "ingest", fallback_all_when_untyped=True)
    return {
        "prompt": [
            {"role": "system", "content": INGEST_T3_SYSTEM_PROMPT + OUTPUT_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        "label": None,
        "metadata": {
            "task": "ingest_t3",
            "snapshot_id": snapshot_id,
            "traj_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
            "session_time": record.get("generated_at", ""),
            "pending_messages": record.get("pending_messages", []),
            "probes": probes,
            "probe_source": "query_probes:ingest",
            "tools": [tool.get("function", {}).get("name", "") for tool in INGEST_T3_TOOLS],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert T3 ingest RL data to Slime JSONL")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root; defaults to input parent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    data_root = Path(args.data_root) if args.data_root else input_path.parent
    session = SnapshotSession(str(data_root), enable_git=False)

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [
        convert_record(r, session)
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
