#!/usr/bin/env python3
"""一键转换三个阶段（ingest / consolidate / retrieve）的数据为 Slime JSONL 格式。

支持两种数据集布局:
1. 单文件模式: 指定一个 rl_data.jsonl，脚本自动按 eval_ratio 划分 train/eval
2. 预分割模式: 数据集目录下已有 train/ val/ test/ 子目录，每个内含 rl_data.jsonl

自动检测逻辑:
    如果 --input 指向一个目录，且目录下存在 train/ val/ test/ 中至少一个含有
    rl_data.jsonl，则进入预分割模式。否则 --input 应为具体的 rl_data.jsonl 文件。

输出命名:
    {task}_{dataset_name}_{split}.jsonl

示例:
    # 预分割模式 —— 数据集目录已含 train/val/test
    python /data/cloud_disk_1/erenpeng/llm-gateway/rl/slime_train/convert_all_stages.py \
        --input /data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e \
        --output-dir /data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output \
        --seed 42

    # 单文件模式 —— 传统单 rl_data.jsonl
    python convert_all_stages.py \
        --input /path/to/rl_data.jsonl \
        --output-dir /path/to/output/ \
        --eval_ratio 0.05 \
        --seed 42

    # 只转换指定阶段
    python convert_all_stages.py \
        --input /path/to/merged_stage1_e2e \
        --output-dir /path/to/output/ \
        --stages ingest retrieve
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage converters (lazy imports to avoid loading all dependencies upfront)
# ---------------------------------------------------------------------------

def run_ingest(input_path: Path, data_root: Path) -> list[dict]:
    """转换 ingest 阶段数据，返回样本列表。"""
    from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes
    from llm_gateway.rl.slime_train.ingest.convert_to_slime_format import (
        convert_record as ingest_convert_record,
    )

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    samples = [
        ingest_convert_record(r)
        for r in records
        if r.get("rl_task_type", "ingest") == "ingest"
        and select_task_probes(r, "ingest", fallback_all_when_untyped=True)
    ]
    return samples


def run_consolidate(input_path: Path, data_root: Path) -> list[dict]:
    """转换 consolidate 阶段数据，返回样本列表。"""
    from llm_gateway.rl.slime_train.consolidate.convert_to_slime_format import build_samples

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = build_samples(records)
    return samples


def run_retrieve(input_path: Path, data_root: Path) -> list[dict]:
    """转换 retrieve 阶段数据，返回样本列表。"""
    from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import convert_probe
    from llm_gateway.rl.slime_train.memory_rl.probes import select_task_probes

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(f"  加载了 {len(records)} 条 step records")

    converted = []
    skipped = 0
    for record in records:
        retrieve_probes = select_task_probes(record, "retrieve")
        if not retrieve_probes:
            continue

        snap_id = record.get("snapshot_id", "")
        traj_id = record.get("trajectory_id", "")
        if not snap_id or not traj_id:
            skipped += len(retrieve_probes)
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
                skipped += 1
                continue
            try:
                sample = convert_probe(probe, record_meta)
                converted.append(sample)
            except Exception as e:
                logger.warning(f"  转换 probe 失败: {e}")
                skipped += 1

    logger.info(f"  转换成功 {len(converted)} 条，跳过 {skipped} 条")
    return converted


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

STAGE_RUNNERS = {
    "ingest": run_ingest,
    "consolidate": run_consolidate,
    "retrieve": run_retrieve,
}

ALL_STAGES = list(STAGE_RUNNERS.keys())


def _detect_splits(input_path: Path) -> dict[str, Path] | None:
    """检测数据集目录下是否存在预分割的 train/val/test 子目录。

    Returns:
        dict mapping split_name -> rl_data.jsonl path, or None if not pre-split.
    """
    if not input_path.is_dir():
        return None

    splits = {}
    for split_name in ("train", "val", "test"):
        candidate = input_path / split_name / "rl_data.jsonl"
        if candidate.exists():
            splits[split_name] = candidate

    return splits if splits else None


def _write_samples(samples: list[dict], output_path: Path) -> None:
    """将样本列表写入 JSONL 文件。"""
    if not samples:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _write_split_by_ratio(
    samples: list[dict],
    output_dir: Path,
    task: str,
    dataset_name: str,
    eval_ratio: float,
    seed: int,
) -> None:
    """按比例划分 train/eval 并写入。用于单文件模式。"""
    if not samples:
        logger.warning(f"  [{task.upper()}] 没有生成任何样本，跳过写入")
        return

    rng = random.Random(seed)
    rng.shuffle(samples)

    eval_count = int(len(samples) * eval_ratio) if eval_ratio > 0 else 0
    eval_samples = samples[:eval_count]
    train_samples = samples[eval_count:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{task}_{dataset_name}_train.jsonl"
    _write_samples(train_samples, train_path)
    logger.info(f"    → {train_path} ({len(train_samples)} 条)")

    if eval_samples:
        eval_path = output_dir / f"{task}_{dataset_name}_eval.jsonl"
        _write_samples(eval_samples, eval_path)
        logger.info(f"    → {eval_path} ({len(eval_samples)} 条)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键转换 ingest / consolidate / retrieve 三阶段数据为 Slime JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入路径: 可以是 rl_data.jsonl 文件，或者包含 train/val/test 子目录的数据集目录")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据集根目录（包含 snapshots/, workdirs/ 等）。"
                             "单文件模式下默认与 --input 同级目录；预分割模式下默认为各 split 目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="数据集名称（用于输出文件名）。默认自动从输入路径推断")
    parser.add_argument("--stages", nargs="*", default=None,
                        choices=ALL_STAGES,
                        help=f"要转换的阶段（默认全部: {ALL_STAGES}）")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="评估集比例，仅在单文件模式下使用 (默认 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    stages = args.stages or ALL_STAGES

    # 推断数据集名称
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        # 从输入路径推断：如果是目录取目录名，如果是文件取父目录名
        if input_path.is_dir():
            dataset_name = input_path.name
        else:
            dataset_name = input_path.parent.name

    logger.info(f"输入路径: {input_path}")
    logger.info(f"数据集名称: {dataset_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"转换阶段: {stages}")

    # 检测是否为预分割模式
    splits = _detect_splits(input_path)

    if splits:
        # ===== 预分割模式 =====
        logger.info(f"检测到预分割数据集，splits: {list(splits.keys())}")

        for split_name, rl_data_path in sorted(splits.items()):
            logger.info("=" * 60)
            logger.info(f"处理 split: {split_name}")

            # data_root 为该 split 的目录（含 snapshots/, workdirs/ 等）
            split_data_root = Path(args.data_root).resolve() if args.data_root else rl_data_path.parent

            for stage in stages:
                runner = STAGE_RUNNERS[stage]
                logger.info(f"  [{stage.upper()}] 开始转换...")
                try:
                    samples = runner(rl_data_path, split_data_root)
                except Exception as e:
                    logger.error(f"  [{stage.upper()}] 转换失败: {e}", exc_info=True)
                    continue

                if not samples:
                    logger.warning(f"  [{stage.upper()}] 没有生成任何样本，跳过")
                    continue

                # 输出: {task}_{dataset_name}_{split}.jsonl
                out_path = output_dir / f"{stage}_{dataset_name}_{split_name}.jsonl"
                _write_samples(samples, out_path)
                logger.info(f"    → {out_path} ({len(samples)} 条)")

    else:
        # ===== 单文件模式 =====
        if not input_path.is_file():
            logger.error(f"输入路径既不是预分割目录，也不是有效文件: {input_path}")
            sys.exit(1)

        logger.info(f"单文件模式，eval_ratio={args.eval_ratio}, seed={args.seed}")

        data_root = Path(args.data_root).resolve() if args.data_root else input_path.parent

        for stage in stages:
            runner = STAGE_RUNNERS[stage]
            logger.info("=" * 60)
            logger.info(f"[{stage.upper()}] 开始转换...")
            try:
                samples = runner(input_path, data_root)
            except Exception as e:
                logger.error(f"[{stage.upper()}] 转换失败: {e}", exc_info=True)
                continue

            _write_split_by_ratio(samples, output_dir, stage, dataset_name, args.eval_ratio, args.seed)

    logger.info("=" * 60)
    logger.info("全部完成!")


if __name__ == "__main__":
    main()
