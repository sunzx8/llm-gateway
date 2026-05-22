#!/usr/bin/env python3
"""Print compact JSONL dataset stats for RL train/eval logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def infer_task(metadata: dict[str, Any]) -> str:
    task = str(metadata.get("task", metadata.get("mixed_task", ""))).lower()
    if task.startswith("ingest"):
        return "ingest"
    if task.startswith("consolidate") or task.startswith("evolve"):
        return "consolidate"
    if task.startswith("retrieve") or task.startswith("query") or task.startswith("consume"):
        return "retrieve"
    if isinstance(metadata.get("ground_truth"), dict):
        return "retrieve"
    if metadata.get("probes"):
        return "ingest_or_consolidate"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize memory RL JSONL data")
    parser.add_argument("path")
    parser.add_argument("--label", default="data")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[data:{args.label}] missing path={path}")
        return

    task_counts: Counter[str] = Counter()
    prompt_roles: Counter[str] = Counter()
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                task_counts["<invalid_json>"] += 1
                continue
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            task_counts[infer_task(metadata)] += 1
            prompt = item.get("prompt") if isinstance(item, dict) else None
            if isinstance(prompt, list):
                for msg in prompt:
                    if isinstance(msg, dict):
                        prompt_roles[str(msg.get("role", "?"))] += 1

    tasks = ", ".join(f"{k}={v}" for k, v in sorted(task_counts.items())) or "none"
    roles = ", ".join(f"{k}={v}" for k, v in sorted(prompt_roles.items())) or "none"
    print(f"[data:{args.label}] path={path} samples={count} tasks={{ {tasks} }} prompt_roles={{ {roles} }}")


if __name__ == "__main__":
    main()
