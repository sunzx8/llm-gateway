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

_TASK_ALIASES = {
    "ingest": "ingest",
    "consolidate": "consolidate",
    "evolve": "consolidate",
    "retrieve": "retrieve",
    "query": "retrieve",
    "consume": "retrieve",
}


def shift_pre_ingest_snapshot(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shift each record's snapshot_id within a trajectory by -1.

    The dataset (e.g. ``merged_stage1_e2e``) stores ``snapshot_id`` as the
    POST-ingest snapshot for that step (verified by experiments/A_audit:
    next-snap dominates curr-snap 61-82% with curr-better only ~0.1%). Training
    code (ingest/consolidate reward + custom_generate task_loop) assumes
    ``snapshot_id`` is the PRE-ingest snapshot and applies tool calls on top of
    it. Without this shift the env already contains the memories that the
    pending_messages would produce, so r_probe is artificially inflated and
    gradient signal collapses.

    For each trajectory, sort by (session_index, batch_index, step_index) and
    replace ``snapshot_id`` with the *previous* record's ``snapshot_id``. The
    first record in each trajectory is mapped to an empty snapshot_id (""),
    which downstream loaders treat as "start from a freshly-reset env".

    The original POST snapshot id is preserved as ``post_ingest_snapshot_id``
    so retrieve / debug paths can still find it.
    """
    by_traj: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_traj.setdefault(r.get("trajectory_id", ""), []).append(r)

    out: list[dict[str, Any]] = []
    for tid, recs in by_traj.items():
        recs.sort(
            key=lambda r: (
                r.get("session_index", 0),
                r.get("batch_index", 0),
                r.get("step_index", 0),
            )
        )
        prev_snap = ""
        for r in recs:
            r2 = dict(r)
            r2["post_ingest_snapshot_id"] = r.get("snapshot_id", "")
            r2["snapshot_id"] = prev_snap
            prev_snap = r.get("snapshot_id", "")
            out.append(r2)
    return out


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
    # NOTE: convert_ingest_record is single-arg in current shawnxsun layout
    # (task_loop mode); session is unused but kept for forward compatibility.
    _ = session
    samples = [
        convert_ingest_record(r)
        for r in records
        if r.get("rl_task_type", "ingest") == "ingest"
        and select_task_probes(r, "ingest", fallback_all_when_untyped=True)
    ]
    # Carry post_ingest_snapshot_id (if shifting was applied) into metadata
    # so reward / debug paths can locate the original POST snapshot.
    by_id = {r.get("session_id", "") + "::" + r.get("trajectory_id", ""): r for r in records}
    for s in samples:
        meta = s.get("metadata") or {}
        key = (meta.get("session_id") or "") + "::" + (meta.get("traj_id") or "")
        rec = by_id.get(key)
        if rec is not None and "post_ingest_snapshot_id" in rec:
            meta["post_ingest_snapshot_id"] = rec["post_ingest_snapshot_id"]
    return [tag_sample(s, "ingest") for s in samples]


def build_consolidate(records: list[dict[str, Any]], session: SnapshotSession) -> list[dict[str, Any]]:
    _ = session  # unused; kept for backward-compatible signature
    return [tag_sample(s, "consolidate") for s in build_consolidate_samples(records)]


def build_retrieve(records: list[dict[str, Any]], data_root: str) -> list[dict[str, Any]]:
    # Lazy import: not all checkouts ship a current retrieve.convert_to_slime_format.
    try:
        from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import convert_sample as convert_retrieve_sample
        from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import load_fs_from_git
    except ImportError as exc:
        raise RuntimeError(
            "build_retrieve requires retrieve.convert_to_slime_format.{convert_sample,load_fs_from_git}; "
            "drop --tasks=retrieve or update the retrieve converter."
        ) from exc

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
    parser.add_argument(
        "--shift-snapshot",
        dest="shift_snapshot",
        action="store_true",
        default=True,
        help=(
            "Shift each ingest/consolidate record's snapshot_id to the previous "
            "step's snapshot within the same trajectory (recover true PRE-ingest "
            "state). Verified necessary on merged_stage1_e2e by "
            "experiments/A_audit. Default: ON."
        ),
    )
    parser.add_argument(
        "--no-shift-snapshot",
        dest="shift_snapshot",
        action="store_false",
        help="Disable snapshot_id shift; keep raw POST snapshot per record.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data_root = str(Path(args.data_root) if args.data_root else input_path.parent)
    tasks = parse_tasks(args.tasks)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ingest/consolidate apply tool calls on top of the snapshot, so they need
    # PRE-ingest snapshots. retrieve evaluates against an already-ingested env,
    # so its records keep the raw POST snapshot.
    if args.shift_snapshot:
        ingest_consolidate_records = shift_pre_ingest_snapshot(records)
        print(f"[shift] ingest/consolidate records shifted to PRE-ingest (n={len(ingest_consolidate_records)})")
    else:
        ingest_consolidate_records = records
        print("[shift] disabled; using raw POST snapshots for ingest/consolidate")
    retrieve_records = records  # always raw POST

    session = SnapshotSession(data_root, enable_git=False)
    rng = random.Random(args.seed)

    samples_by_task: dict[str, list[dict[str, Any]]] = {}
    if "ingest" in tasks:
        samples_by_task["ingest"] = build_ingest(ingest_consolidate_records, session)
    if "consolidate" in tasks:
        samples_by_task["consolidate"] = build_consolidate(ingest_consolidate_records, session)
    if "retrieve" in tasks:
        samples_by_task["retrieve"] = build_retrieve(retrieve_records, data_root)

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
