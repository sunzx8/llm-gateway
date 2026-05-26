#!/usr/bin/env python3
"""离线工具：基于 *.detail.jsonl 从原始 jsonl 抽取过滤后的训练数据。

适用于：
  1. 全量任务还在跑、想看当前已完成部分的过滤结果（不打断主任务）。
  2. 全量跑完后，重新调阈值（例如把 pass=1/N 也去掉，只留 2/N..N-2/N）而不重跑 rollout。
  3. 主进程被 kill / OOM 后，从 detail 恢复 final filtered。

detail.jsonl 每行结构（由 rejection_sampling_filter.py 产出）：
  {
    "index": <int>,           # 在原始 jsonl 中的 0-based 行号
    "task": "ingest|retrieve|consolidate",
    "n_rollouts": <int>,
    "pass_count": <int>,
    "pass_rate": <float>,
    "avg_score": <float>,
    "keep": <bool>,
    "rollouts": [...],
    "metadata_brief": {...},
  }

用法示例：
  # 默认规则（与主脚本一致：0 < pass_count < n_rollouts 才保留）
  python3 build_filtered_from_detail.py \\
      --input  filtered_clean_data/ingest_merged_stage1_e2e_train.clean.jsonl \\
      --detail filtered_clean_data/rejection_filtered/ingest_full.jsonl.detail.jsonl \\
      --output filtered_clean_data/rejection_filtered/ingest_partial.jsonl

  # 自定义保留区间（例如只保留 pass_count in [2, 6]）
  python3 build_filtered_from_detail.py \\
      --input  ... --detail ... --output ... \\
      --min-pass 2 --max-pass 6

  # 三个 task 一次跑（auto 推断）
  for task in ingest retrieve consolidate; do
      python3 build_filtered_from_detail.py \\
          --input  filtered_clean_data/${task}_merged_stage1_e2e_train.clean.jsonl \\
          --detail filtered_clean_data/rejection_filtered/${task}_full.jsonl.detail.jsonl \\
          --output filtered_clean_data/rejection_filtered/${task}_partial.jsonl
  done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_detail(path: str) -> dict[int, dict[str, Any]]:
    """读 detail.jsonl，返回 {index: detail_record}。

    若同一 index 出现多次（例如二次重跑追加），保留最后一次。
    """
    out: dict[int, dict[str, Any]] = {}
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            idx = rec.get("index")
            if not isinstance(idx, int):
                bad_lines += 1
                continue
            out[idx] = rec
    if bad_lines:
        print(f"[WARN] detail 文件有 {bad_lines} 行无法解析 (可能是流式写入截断)", file=sys.stderr)
    return out


def _iter_input_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def _decide_keep(rec: dict[str, Any], min_pass: int | None, max_pass: int | None,
                 min_score: float | None, max_score: float | None) -> bool:
    """根据阈值判定是否保留。

    若所有阈值参数都为 None，等价于主脚本默认规则：0 < pass_count < n_rollouts。
    """
    n = int(rec.get("n_rollouts", 0))
    if n <= 0:
        return False
    pc = int(rec.get("pass_count", 0))
    avg = float(rec.get("avg_score", 0.0))

    # 默认：0 < pc < n
    if min_pass is None and max_pass is None and min_score is None and max_score is None:
        return 0 < pc < n

    # 显式区间
    if min_pass is not None and pc < min_pass:
        return False
    if max_pass is not None and pc > max_pass:
        return False
    if min_score is not None and avg < min_score:
        return False
    if max_score is not None and avg > max_score:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="从 detail.jsonl 离线构建过滤后的训练集")
    p.add_argument("--input", required=True, help="原始全量 jsonl（与主脚本 --input 相同）")
    p.add_argument("--detail", required=True, help="主脚本产出的 *.detail.jsonl")
    p.add_argument("--output", required=True, help="过滤后输出 jsonl")
    # 阈值
    p.add_argument("--min-pass", type=int, default=None,
                   help="保留 pass_count >= min_pass（默认 1，即排除 0/N）")
    p.add_argument("--max-pass", type=int, default=None,
                   help="保留 pass_count <= max_pass（默认 n_rollouts-1，即排除 N/N）")
    p.add_argument("--min-score", type=float, default=None,
                   help="保留 avg_score >= min_score（可选）")
    p.add_argument("--max-score", type=float, default=None,
                   help="保留 avg_score <= max_score（可选）")
    # 行为
    p.add_argument("--require-no-error", action="store_true",
                   help="若该 sample 任意 rollout 有 error 则丢弃")
    p.add_argument("--summary-only", action="store_true",
                   help="只打印统计，不写出文件")
    args = p.parse_args()

    detail = _load_detail(args.detail)
    print(f"[INFO] detail samples loaded: {len(detail)}")

    n_total = 0
    n_have_detail = 0
    n_keep = 0
    n_no_pass = 0    # pc == 0
    n_all_pass = 0   # pc == n
    n_dropped_error = 0
    n_dropped_threshold = 0
    pass_hist: dict[int, int] = {}  # pass_count -> count

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = None if args.summary_only else open(out_path, "w", encoding="utf-8")
    try:
        for i, sample in enumerate(_iter_input_samples(args.input)):
            n_total += 1
            if sample is None:
                continue
            rec = detail.get(i)
            if rec is None:
                continue  # 还没跑到这条
            n_have_detail += 1

            n = int(rec.get("n_rollouts", 0))
            pc = int(rec.get("pass_count", 0))
            pass_hist[pc] = pass_hist.get(pc, 0) + 1
            if pc == 0:
                n_no_pass += 1
            elif pc == n and n > 0:
                n_all_pass += 1

            # error 过滤
            if args.require_no_error:
                rollouts = rec.get("rollouts", []) or []
                has_err = any((isinstance(r, dict) and isinstance(r.get("info"), dict)
                               and r["info"].get("error")) for r in rollouts)
                if has_err:
                    n_dropped_error += 1
                    continue

            if not _decide_keep(rec, args.min_pass, args.max_pass,
                                args.min_score, args.max_score):
                n_dropped_threshold += 1
                continue

            # 写出（保持原 sample，附加 __rejection_sampling__）
            if out_f is not None:
                enriched = dict(sample)
                meta = dict(enriched.get("metadata", {}) or {})
                meta["__rejection_sampling__"] = {
                    "pass_count": pc,
                    "n_rollouts": n,
                    "pass_rate": rec.get("pass_rate"),
                    "avg_score": rec.get("avg_score"),
                }
                enriched["metadata"] = meta
                out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            n_keep += 1
    finally:
        if out_f is not None:
            out_f.close()

    n_pending = n_total - n_have_detail
    print("=" * 60)
    print(f"input total samples:      {n_total}")
    print(f"with detail (completed):  {n_have_detail}")
    print(f"pending (not yet run):    {n_pending}")
    print(f"  no_pass(0/N):           {n_no_pass}")
    print(f"  all_pass(N/N):          {n_all_pass}")
    if args.require_no_error:
        print(f"  dropped(has_error):     {n_dropped_error}")
    print(f"  dropped(threshold):     {n_dropped_threshold}")
    print(f"  KEPT:                   {n_keep}")
    print("pass_count histogram:")
    for k in sorted(pass_hist.keys()):
        print(f"  pc={k}: {pass_hist[k]}")
    if not args.summary_only:
        print(f"output: {out_path}")


if __name__ == "__main__":
    main()
