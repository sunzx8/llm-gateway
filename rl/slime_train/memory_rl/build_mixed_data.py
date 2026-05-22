#!/usr/bin/env python3
"""Build mixed Slime JSONL data for memory RL tasks.

Supported task modes:
- ingest
- consolidate / evolve
- retrieve / query / consume
- combinations such as ingest+consolidate, ingest+consolidate+retrieve
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.consolidate.convert_to_slime_format import build_samples as build_consolidate_samples
from llm_gateway.rl.slime_train.ingest.convert_to_slime_format import convert_record as convert_ingest_record
from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes
from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import convert_sample as convert_retrieve_sample
from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import load_fs_from_git

_TASK_ALIASES = {
    "ingest": "ingest",
    "consolidate": "consolidate",
    "evolve": "consolidate",
    "retrieve": "retrieve",
    "query": "retrieve",
    "consume": "retrieve",
}


def parse_tasks(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in raw.replace(",", "+").split("+") if p.strip()]
    tasks: list[str] = []
    for part in parts:
        task = _TASK_ALIASES.get(part)
        if not task:
            raise ValueError(f"unsupported task in mode: {part}")
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def tag_sample(sample: dict[str, Any], task: str) -> dict[str, Any]:
    metadata = sample.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["task"] = task
        metadata["mixed_task"] = task
    return sample


def build_ingest(records: list[dict[str, Any]], session: SnapshotSession) -> list[dict[str, Any]]:
    samples = [
        convert_ingest_record(r, session)
        for r in records
        if r.get("rl_task_type", "ingest") == "ingest"
        and select_task_probes(r, "ingest", fallback_all_when_untyped=True)
    ]
    return [tag_sample(s, "ingest") for s in samples]


def build_consolidate(records: list[dict[str, Any]], session: SnapshotSession) -> list[dict[str, Any]]:
    return [tag_sample(s, "consolidate") for s in build_consolidate_samples(records, session)]


def build_retrieve(records: list[dict[str, Any]], data_root: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    snapshot_cache: dict[str, dict[str, Any]] = {}
    index_cache: dict[str, dict[str, Any]] = {}
    skipped = 0

    for record in records:
        snap_id = record.get("snapshot_id", "")
        traj_id = record.get("trajectory_id", "")
        retrieve_probes = record.get("probes_by_task", {}).get("retrieve", [])
        if not retrieve_probes:
            continue
        if not snap_id or not traj_id:
            skipped += len(retrieve_probes)
            continue
        if snap_id not in snapshot_cache:
            try:
                snapshot_cache[snap_id] = load_fs_from_git(data_root, traj_id, snap_id, index_cache)
            except Exception as exc:  # noqa: BLE001 - keep builder robust across partial data
                print(f"[warn] skip retrieve snapshot {traj_id}/{snap_id}: {exc}", file=sys.stderr)
                skipped += len(retrieve_probes)
                continue
        record_meta = {
            "snapshot_id": snap_id,
            "trajectory_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
        }
        for probe in retrieve_probes:
            try:
                converted.append(tag_sample(convert_retrieve_sample(snapshot_cache[snap_id], probe, record_meta), "retrieve"))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] skip retrieve probe: {exc}", file=sys.stderr)
                skipped += 1
    if skipped:
        print(f"[warn] skipped retrieve probes: {skipped}", file=sys.stderr)
    return converted


def apply_limits(samples_by_task: dict[str, list[dict[str, Any]]], limit_per_task: int, rng: random.Random) -> list[dict[str, Any]]:
    mixed: list[dict[str, Any]] = []
    for task, samples in samples_by_task.items():
        rng.shuffle(samples)
        if limit_per_task > 0:
            samples = samples[:limit_per_task]
        mixed.extend(samples)
        print(f"task={task} samples={len(samples)}")
    rng.shuffle(mixed)
    return mixed


def write_jsonl(path: str, samples: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed memory RL Slime JSONL")
    parser.add_argument("--input", required=True, help="Input rl_data.jsonl")
    parser.add_argument("--data-root", default=None, help="Dataset root; defaults to input parent")
    parser.add_argument("--tasks", default="ingest+consolidate+retrieve", help="Task mode, e.g. ingest+consolidate, ingest+consolidate+retrieve")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--limit-per-task", type=int, default=0, help="Optional cap per task before mixing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    data_root = str(Path(args.data_root) if args.data_root else input_path.parent)
    tasks = parse_tasks(args.tasks)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    session = SnapshotSession(data_root, enable_git=False)
    rng = random.Random(args.seed)

    samples_by_task: dict[str, list[dict[str, Any]]] = {}
    if "ingest" in tasks:
        samples_by_task["ingest"] = build_ingest(records, session)
    if "consolidate" in tasks:
        samples_by_task["consolidate"] = build_consolidate(records, session)
    if "retrieve" in tasks:
        samples_by_task["retrieve"] = build_retrieve(records, data_root)

    mixed = apply_limits(samples_by_task, args.limit_per_task, rng)
    if args.eval_output and args.eval_ratio > 0:
        n_eval = max(1, int(len(mixed) * args.eval_ratio)) if mixed else 0
        eval_samples = mixed[:n_eval]
        train_samples = mixed[n_eval:]
    else:
        train_samples = mixed
        eval_samples = []

    write_jsonl(args.output, train_samples)
    if args.eval_output:
        write_jsonl(args.eval_output, eval_samples)
    print(f"wrote train={len(train_samples)} eval={len(eval_samples)} tasks={'+'.join(tasks)}")


if __name__ == "__main__":
    main()
