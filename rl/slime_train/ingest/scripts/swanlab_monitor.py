#!/usr/bin/env python3
"""
SwanLab 实时训练监控脚本 — Ingest GRPO

同时解析：
1. slime 训练日志 (train.log / run*.log) → entropy, kl_loss, pg_loss, grad_norm, lr
2. reward 细分日志 (reward_metrics.jsonl) → r_probe, r_format, r_total
3. rollout perf 日志 → step_time, train_time, wait_time_ratio

用法:
    python scripts/swanlab_monitor.py --log-dir <日志目录> [--run-name <运行名称>]
"""

import argparse
import json
import os
import re
import time
import logging
from pathlib import Path

import swanlab

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# SwanLab API Key
SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", "BVQaRTEEZKWC9p3iF5MMp")


def strip_ansi(text: str) -> str:
    """移除 ANSI 颜色码和 Ray 前缀"""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'\([^)]*pid=\d+[^)]*\)\s*', '', text)
    return text


def parse_train_metric_line(line: str) -> dict | None:
    """解析训练指标行 (model.py 输出)"""
    if "model.py" not in line or "step" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"step\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    try:
        step = int(match.group(1))
        data = eval(match.group(2))
        data["_step"] = step
        return data
    except Exception:
        return None


def parse_rollout_perf_line(line: str) -> dict | None:
    """解析 perf 输出行 (train_metric_utils.py 输出)"""
    if "perf" not in line:
        return None
    if "train_metric_utils" not in line and "rollout.py" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"perf\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    try:
        data = eval(match.group(2))
        data["_rollout_id"] = int(match.group(1))
        return data
    except Exception:
        return None


def parse_rollout_metric_line(line: str) -> dict | None:
    """解析 rollout 指标行 (data.py 输出)"""
    if "data.py" not in line or "rollout" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"rollout\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    try:
        data = eval(match.group(2))
        data["_rollout_id"] = int(match.group(1))
        return data
    except Exception:
        return None


def monitor(
    train_log: Path,
    reward_log: Path,
    poll_interval: float = 5.0,
):
    """实时监控日志并上传到 SwanLab"""
    logger.info(f"监控训练日志: {train_log}")
    logger.info(f"监控 reward 日志: {reward_log}")

    train_pos = 0
    reward_pos = 0
    global_step = 0
    reward_batch_count = 0
    rollout_group_rewards = []
    rollout_group_count = 0

    while True:
        # ============================================================
        # 1. 解析训练日志
        # ============================================================
        if train_log.exists():
            with open(train_log, "r", errors="replace") as f:
                f.seek(train_pos)
                new_lines = f.readlines()
                train_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue

                train_data = parse_train_metric_line(line)
                if train_data:
                    step = train_data.get("_step", global_step)
                    global_step = step
                    metrics = {}

                    key_mapping = {
                        "train/entropy_loss": "train/entropy",
                        "train/kl_loss": "train/kl_loss",
                        "train/pg_loss": "train/pg_loss",
                        "train/pg_clipfrac": "train/pg_clipfrac",
                        "train/ppo_kl": "train/ppo_kl",
                        "train/loss": "train/total_loss",
                        "train/grad_norm": "train/grad_norm",
                    }
                    for src, dst in key_mapping.items():
                        if src in train_data and isinstance(train_data[src], (int, float)):
                            metrics[dst] = train_data[src]

                    for key, val in train_data.items():
                        if "lr-pg" in key and isinstance(val, (int, float)):
                            metrics["train/lr"] = val
                            break

                    if metrics:
                        swanlab.log(metrics, step=step)
                    continue

                rollout_data = parse_rollout_metric_line(line)
                if rollout_data:
                    rid = rollout_data.get("_rollout_id", 0)
                    metrics = {}
                    for key in ("rollout/rewards", "rollout/raw_reward", "rollout/response_lengths",
                                "rollout/advantages", "rollout/log_probs", "rollout/ref_log_probs"):
                        if key in rollout_data and isinstance(rollout_data[key], (int, float)):
                            metrics[key] = rollout_data[key]
                    if metrics:
                        swanlab.log(metrics, step=rid)
                    continue

                perf_data = parse_rollout_perf_line(line)
                if perf_data:
                    rid = perf_data.get("_rollout_id", 0)
                    metrics = {}
                    perf_keys = {
                        "perf/step_time": "perf/step_time",
                        "perf/update_weights_time": "perf/update_weights_time",
                        "perf/train_time": "perf/train_time",
                        "perf/train_wait_time": "perf/train_wait_time",
                        "perf/actor_train_time": "perf/actor_train_time",
                        "perf/actor_train_tok_per_s": "perf/actor_train_tok_per_s",
                        "perf/wait_time_ratio": "perf/wait_time_ratio",
                        "perf/actor_train_tflops": "perf/actor_train_tflops",
                    }
                    for src, dst in perf_keys.items():
                        if src in perf_data and isinstance(perf_data[src], (int, float)):
                            metrics[dst] = perf_data[src]
                    if metrics:
                        swanlab.log(metrics, step=rid)
                    continue

        # ============================================================
        # 2. 解析 reward 细分指标日志
        # ============================================================
        if reward_log.exists():
            with open(reward_log, "r", errors="replace") as f:
                f.seek(reward_pos)
                new_lines = f.readlines()
                reward_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                reward_batch_count += 1
                # 动态读取 reward_metrics.jsonl 中所有数值字段
                metrics = {}
                for key, val in record.items():
                    if isinstance(val, (int, float)):
                        metrics[f"reward/{key}"] = val
                if metrics:
                    swanlab.log(metrics, step=reward_batch_count)

                # 每 8 个 sample 一组，计算组内 reward std 和 advantage
                r_total = record.get("r_total", record.get("reward", 0))
                rollout_group_rewards.append(float(r_total))
                if len(rollout_group_rewards) >= 8:
                    rollout_group_count += 1
                    import statistics
                    group_mean = statistics.mean(rollout_group_rewards)
                    group_std = statistics.pstdev(rollout_group_rewards)
                    if group_std > 0:
                        advantages = [(r - group_mean) / (group_std + 1e-6) for r in rollout_group_rewards]
                    else:
                        advantages = [0.0] * len(rollout_group_rewards)
                    swanlab.log({
                        "reward_group/mean": group_mean,
                        "reward_group/std": group_std,
                        "reward_group/advantage_max": max(advantages),
                        "reward_group/advantage_min": min(advantages),
                    }, step=rollout_group_count)
                    rollout_group_rewards = []

                if reward_batch_count % 20 == 0:
                    logger.info(
                        f"[Reward] batch={reward_batch_count}, "
                        f"r_total={record.get('r_total', 0):.3f}"
                    )

        time.sleep(poll_interval)


def main():
    import datetime

    parser = argparse.ArgumentParser(description="SwanLab 实时训练监控 — Ingest GRPO")
    parser.add_argument("--log-dir", type=str, required=True, help="训练日志目录")
    parser.add_argument("--run-name", type=str, default=None, help="SwanLab 运行名称")
    parser.add_argument("--task-version", type=str, default=None, help="task version (如 atomic_code_t2, agent_loop)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="轮询间隔(秒)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    train_log = log_dir / "train.log"
    if not train_log.exists():
        output_dir = log_dir.parent
        run_logs = sorted(output_dir.parent.glob("run*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_logs:
            train_log = run_logs[0]

    reward_log = log_dir / "reward_metrics.jsonl"

    # experiment_name 格式: {task_version}_ingest_{时间戳}
    task_version = args.task_version or os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{task_version}_ingest_{timestamp}"

    os.environ["SWANLAB_API_KEY"] = SWANLAB_API_KEY
    swanlab.init(
        project="memory_ai_rl",
        experiment_name=run_name,
        mode="cloud",
        config={
            "model": "Qwen3.5-35B-A3B",
            "task": "ingest",
            "task_version": task_version,
            "log_dir": str(log_dir),
            "reward_weights": {"r_probe": 0.9, "r_format": 0.1},
        },
    )
    logger.info(f"SwanLab 初始化: project=memory_ai_rl, run={run_name}")
    logger.info(f"SwanLab URL: {swanlab.get_url()}")

    try:
        monitor(train_log, reward_log, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        logger.info("监控终止")
    finally:
        swanlab.finish()
        logger.info("SwanLab 已关闭")


if __name__ == "__main__":
    main()
