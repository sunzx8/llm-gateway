#!/usr/bin/env python3
"""
SwanLab 实时训练监控脚本 — Consolidate GRPO/GSPO

同时解析：
1. slime 训练日志 (train.log) → entropy/kl/pg_loss/grad_norm/lr/clipfrac/ppo_kl
2. slime 验证日志 (eval step / tqdm progress) → val/* / val_progress/*
3. reward 细分日志 (reward_metrics.jsonl) → train reward/* 与 eval val_reward/* 分离
       r_total / r_acc_delta(/_signed) / r_score_delta(/_signed) /
       r_after_score / r_after_headroom / r_context_diff / r_format /
       n_tool_calls / action_cost
4. rollout 详细记录 (rollout_records.jsonl) →
       success_rate, regression_rate (after_acc < before_acc),
       finish_rate / max_turns_rate / no_tool_calls_rate,
       per-turn tool_calls / llm_errors / tool_errors,
       trajectory latency, response 长度分布与截断率
5. 组内统计：N_SAMPLES_PER_PROMPT 个 sample 形成一组 GRPO/GSPO 比较组,
       计算 group reward std / advantage spread / **effective sample ratio**
       (group_std > 阈值的组占比, 反映"有效训练信号"占比，对稀疏 reward 任务关键)
6. rollout perf 日志 → step_time/train_time/wait_time_ratio/tok_per_s/tflops
7. GPU 系统指标
8. 错误率 (parse_function_call/timeout/oom 等)

用法:
    python swanlab_monitor.py --log-dir <日志目录> [--run-name <运行名称>]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path

import swanlab

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# SwanLab API Key
SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", "BVQaRTEEZKWC9p3iF5MMp")


# ===========================================================================
# 工具函数
# ===========================================================================

def strip_ansi(text: str) -> str:
    """移除 ANSI 颜色码和 Ray 前缀"""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"\([^)]*pid=\d+[^)]*\)\s*", "", text)
    return text


def _safe_eval_dict(text: str) -> dict | None:
    """slime 日志里的 dict literal 经常含 numpy/torch 标量字符串，宽松解析。"""
    try:
        data = eval(text, {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")})
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ===========================================================================
# 训练日志解析器
# ===========================================================================

def parse_train_metric_line(line: str) -> dict | None:
    """解析训练指标行 (model.py 输出, 格式 'step X: {...}')"""
    if "model.py" not in line or "step" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"step\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    data = _safe_eval_dict(match.group(2))
    if data is None:
        return None
    data["_step"] = int(match.group(1))
    return data


def parse_eval_metric_line(line: str) -> dict | None:
    """解析验证指标行（多种格式兼容）"""
    line_stripped = strip_ansi(line)

    # "eval step X: {…}" 或 "validation step X: {…}"
    match = re.search(r"(?:eval|validation)[_\s]+(?:step\s*)?(\d+):\s*(\{.*\})", line_stripped, re.IGNORECASE)
    if match:
        data = _safe_eval_dict(match.group(2))
        if data is not None:
            data["_eval_step"] = int(match.group(1))
            return data

    # model.py 中包含 eval 标记的 step 行
    if "eval" in line_stripped.lower() and "step" in line_stripped.lower():
        match = re.search(r"step\s+(\d+):\s*(\{.*\})", line_stripped)
        if match:
            data = _safe_eval_dict(match.group(2))
            if data is not None and (
                any("eval" in str(k).lower() for k in data.keys())
                or "eval" in line_stripped.lower().split("step")[0]
            ):
                data["_eval_step"] = int(match.group(1))
                return data

    # "(eval|val) X: {...}"
    match = re.search(r"(eval|val)\s+(\d+):\s*(\{.*\})", line_stripped, re.IGNORECASE)
    if match:
        data = _safe_eval_dict(match.group(3))
        if data is not None:
            data["_eval_step"] = int(match.group(2))
            return data
    return None



def _parse_hms_to_seconds(text: str) -> float:
    parts = [int(p) for p in text.strip().split(":") if p.isdigit()]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 1:
        return float(parts[0])
    return 0.0


def parse_eval_progress_line(line: str) -> dict | None:
    """解析 tqdm eval 进度行：Eval consolidate:  93%|...| 40/43 [18:27<00:58, 19.35s/it]"""
    line = strip_ansi(line)
    if "Eval " not in line or "/" not in line:
        return None
    match = re.search(
        r"Eval\s+([^:]+):.*?\|\s*(\d+)/(\d+)\s+\[([^<\]]+)<([^,\]]+),\s*([0-9.]+)s/it",
        line,
    )
    if not match:
        return None
    name = re.sub(r"[^A-Za-z0-9_]+", "_", match.group(1).strip()) or "default"
    done = int(match.group(2))
    total = int(match.group(3))
    elapsed_s = _parse_hms_to_seconds(match.group(4))
    eta_s = _parse_hms_to_seconds(match.group(5))
    sec_per_item = float(match.group(6))
    return {
        f"val_progress/{name}/done": float(done),
        f"val_progress/{name}/total": float(total),
        f"val_progress/{name}/pct": done / total if total else 0.0,
        f"val_progress/{name}/elapsed_min": elapsed_s / 60.0,
        f"val_progress/{name}/eta_min": eta_s / 60.0,
        f"val_progress/{name}/sec_per_item": sec_per_item,
    }

def parse_rollout_perf_line(line: str) -> dict | None:
    """解析 perf 输出行 (train_metric_utils.py / rollout.py 输出)"""
    if "perf" not in line:
        return None
    if "train_metric_utils" not in line and "rollout.py" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"perf\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    data = _safe_eval_dict(match.group(2))
    if data is None:
        return None
    data["_rollout_id"] = int(match.group(1))
    return data


def parse_rollout_metric_line(line: str) -> dict | None:
    """解析 rollout 指标行 (data.py 输出)"""
    if "data.py" not in line or "rollout" not in line:
        return None
    line = strip_ansi(line)
    match = re.search(r"rollout\s+(\d+):\s*(\{.*\})", line)
    if not match:
        return None
    data = _safe_eval_dict(match.group(2))
    if data is None:
        return None
    data["_rollout_id"] = int(match.group(1))
    return data


def parse_error_line(line: str) -> str | None:
    """提取常见错误类型，用于错误率统计.

    只识别 loguru/python logger 真正的 ERROR/CRITICAL 级别，避免把以下情况误判：
      * tool 返回字符串里含 'ERROR: File not found: ...'（DEBUG 级别 result 字段）
      * 普通 INFO 日志中提到了 "error" 单词
    """
    line_stripped = strip_ansi(line)
    # 必须在行首附近出现真正的级别标记（loguru: '| ERROR    |'；python logging: 'ERROR:' / '- ERROR -'）
    is_real_error_level = bool(
        re.search(r"\|\s*(ERROR|CRITICAL)\s*\|", line_stripped)
        or re.search(r"\b(ERROR|CRITICAL)\s*[:\-]\s", line_stripped[:120])
    )
    has_traceback = "Traceback (most recent call last)" in line_stripped
    if not is_real_error_level and not has_traceback:
        return None
    if "parse_function_call" in line_stripped:
        return "parse_function_call"
    if "Too many open files" in line_stripped:
        return "too_many_open_files"
    if "TimeoutError" in line_stripped or "timeout" in line_stripped.lower():
        return "timeout"
    if "ConnectionError" in line_stripped or "connection refused" in line_stripped.lower():
        return "connection_error"
    if "OOM" in line_stripped or "out of memory" in line_stripped.lower():
        return "oom"
    if "CUDA" in line_stripped:
        return "cuda_error"
    # HTTP 状态码：用单词边界匹配真正的 5xx/4xx，避免 " 5" 误伤
    if re.search(r"\b5\d{2}\b", line_stripped) and "http" in line_stripped.lower():
        return "http_5xx"
    if re.search(r"\b4\d{2}\b", line_stripped) and "http" in line_stripped.lower():
        return "http_4xx"
    return "other_error"


# ===========================================================================
# GPU 系统指标采集
# ===========================================================================

def get_gpu_metrics() -> dict | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        gpu_utils: list[float] = []
        mem_used_total = 0.0
        mem_total_total = 0.0
        temps: list[float] = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpu_utils.append(float(parts[0]))
                    mem_used_total += float(parts[1])
                    mem_total_total += float(parts[2])
                    temps.append(float(parts[3]))
                except ValueError:
                    continue
        if not gpu_utils:
            return None
        return {
            "system/gpu_util_mean": statistics.mean(gpu_utils),
            "system/gpu_util_max": max(gpu_utils),
            "system/gpu_util_min": min(gpu_utils),
            "system/gpu_mem_used_gb": mem_used_total / 1024,
            "system/gpu_mem_total_gb": mem_total_total / 1024,
            "system/gpu_mem_util_pct": (mem_used_total / mem_total_total * 100) if mem_total_total > 0 else 0,
            "system/gpu_temp_mean": statistics.mean(temps),
            "system/gpu_temp_max": max(temps),
            "system/num_gpus": len(gpu_utils),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


# ===========================================================================
# Rollout 记录 (consolidate-specific) 汇总
# ===========================================================================

# 与 consolidate_reward.reward 落盘的 sub_rewards 字段名对齐
_SUB_REWARD_KEYS_NUMERIC = (
    "r_total",
    "r_format",
    "r_before_score",
    "r_after_score",
    "r_after_headroom",
    "r_before_acc",
    "r_after_acc",
    "r_acc_delta",
    "r_acc_delta_signed",
    "r_score_delta",
    "r_score_delta_signed",
    "r_context_diff",
    "n_tool_calls",
    "action_cost",
    "used_post_snapshot",
)

# task_result.stats 里的字段（来自 BaseContextTask.execute / agentic_trace 末尾）
_TASK_STATS_NUMERIC = (
    "total_latency_s",
    "llm_calls",
    "llm_total_input_tokens_k",
    "llm_total_output_tokens_k",
    "llm_total_tokens_k",
    "llm_total_latency_s",
    "llm_errors",
    "tool_calls",
    "tool_total_latency_s",
    "tool_errors",
    "fs_access_count",
    "vec_access_count",
    "graph_access_count",
)


def _try_parse_response_payload(response: str) -> dict | None:
    """rollout_records.response 是 JSON 字符串（task_loop 模式），尽力解析。"""
    if not response or not isinstance(response, str):
        return None
    s = response.strip()
    if not s.startswith("{"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _bucketize_response_len(lengths: list[int]) -> dict[str, float]:
    """response 长度按几个 bucket 上报命中比例，便于看分布漂移。"""
    if not lengths:
        return {}
    buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, 1 << 30)]
    out: dict[str, float] = {}
    n = len(lengths)
    for lo, hi in buckets:
        cnt = sum(1 for x in lengths if lo <= x < hi)
        label = f"{lo}_{hi}" if hi < (1 << 30) else f"{lo}plus"
        out[f"rollout_stats/resp_len_bucket_{label}_pct"] = cnt / n
    return out


def _log_rollout_record_stats(records: list[dict], batch_id: int) -> None:
    """汇总一批 rollout 详细记录并上报"""
    rewards: list[float] = []
    response_lens: list[int] = []
    n_tool_calls_arr: list[int] = []

    # sub_rewards
    sub_arrs: dict[str, list[float]] = {k: [] for k in _SUB_REWARD_KEYS_NUMERIC}

    # task_result.stats
    task_stat_arrs: dict[str, list[float]] = {k: [] for k in _TASK_STATS_NUMERIC}

    success_count = 0
    error_count = 0
    finish_reason_counter: dict[str, int] = {}
    stop_reason_counter: dict[str, int] = {}
    truncated_count = 0
    regression_count = 0  # after_acc < before_acc
    no_change_count = 0  # after_acc == before_acc
    improve_count = 0  # after_acc > before_acc

    parse_errors = 0  # response 不是合法 JSON / 没 task_result

    for rec in records:
        # 顶层 reward
        rewards.append(float(rec.get("reward", 0.0)))

        # sub_rewards
        sub = rec.get("sub_rewards") or {}
        if isinstance(sub, dict):
            for k in sub_arrs:
                v = sub.get(k)
                if isinstance(v, (int, float)):
                    sub_arrs[k].append(float(v))
            if "error" in sub:
                error_count += 1
            # 进步/退步统计 (basis on sub_rewards before/after acc)
            before_acc = sub.get("r_before_acc")
            after_acc = sub.get("r_after_acc")
            if isinstance(before_acc, (int, float)) and isinstance(after_acc, (int, float)):
                if after_acc > before_acc:
                    improve_count += 1
                elif after_acc < before_acc:
                    regression_count += 1
                else:
                    no_change_count += 1

        # response_len
        rlen = rec.get("response_len", 0)
        if isinstance(rlen, (int, float)):
            response_lens.append(int(rlen))

        # 解析 response payload (task_loop 模式)
        payload = _try_parse_response_payload(rec.get("response", ""))
        if payload is None:
            parse_errors += 1
            continue
        # stop_reason
        sr = str(payload.get("stop_reason", "") or "")
        if sr:
            stop_reason_counter[sr] = stop_reason_counter.get(sr, 0) + 1
        # tool_calls 数
        tcs = payload.get("tool_calls") or []
        if isinstance(tcs, list):
            n_tool_calls_arr.append(len(tcs))
        # task_result
        tr = payload.get("task_result") or {}
        if isinstance(tr, dict):
            stats = tr.get("stats") or {}
            if isinstance(stats, dict):
                ok = stats.get("success")
                if ok is True:
                    success_count += 1
                elif ok is False:
                    pass  # already error counted via sub.error
                fr = str(tr.get("finish_reason", "") or "")
                if fr:
                    finish_reason_counter[fr] = finish_reason_counter.get(fr, 0) + 1
                for k in task_stat_arrs:
                    v = stats.get(k)
                    if isinstance(v, (int, float)):
                        task_stat_arrs[k].append(float(v))
        # 截断标记
        if rec.get("is_truncated") in (1, True):
            truncated_count += 1

    metrics: dict[str, float] = {}
    n = len(records) or 1

    # --- Reward 分布 ---
    if rewards:
        metrics["rollout_stats/reward_mean"] = statistics.mean(rewards)
        metrics["rollout_stats/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0
        metrics["rollout_stats/reward_max"] = max(rewards)
        metrics["rollout_stats/reward_min"] = min(rewards)
        metrics["rollout_stats/reward_median"] = statistics.median(rewards)
        metrics["rollout_stats/positive_reward_rate"] = sum(1 for r in rewards if r > 0) / len(rewards)
        metrics["rollout_stats/negative_reward_rate"] = sum(1 for r in rewards if r < 0) / len(rewards)
        metrics["rollout_stats/zero_reward_rate"] = sum(1 for r in rewards if r == 0) / len(rewards)

    # --- Sub-reward 均值（用 sub/<k>/mean 命名，便于和 reward/<k> 单点曲线对照）---
    for k, arr in sub_arrs.items():
        if not arr:
            continue
        metrics[f"sub/{k}_mean"] = statistics.mean(arr)
        metrics[f"sub/{k}_nonzero_pct"] = sum(1 for x in arr if x != 0) / len(arr)
        # 对 signed delta 单独看正/负比例
        if k.endswith("_signed"):
            metrics[f"sub/{k}_positive_pct"] = sum(1 for x in arr if x > 0) / len(arr)
            metrics[f"sub/{k}_negative_pct"] = sum(1 for x in arr if x < 0) / len(arr)

    # --- 进步/退步比例 (基于 r_after_acc vs r_before_acc) ---
    delta_total = improve_count + regression_count + no_change_count
    if delta_total > 0:
        metrics["rollout_stats/acc_improve_rate"] = improve_count / delta_total
        metrics["rollout_stats/acc_regression_rate"] = regression_count / delta_total
        metrics["rollout_stats/acc_no_change_rate"] = no_change_count / delta_total

    # --- Tool call 数量分布 ---
    if n_tool_calls_arr:
        metrics["rollout_stats/n_tool_calls_mean"] = statistics.mean(n_tool_calls_arr)
        metrics["rollout_stats/n_tool_calls_median"] = statistics.median(n_tool_calls_arr)
        metrics["rollout_stats/n_tool_calls_max"] = max(n_tool_calls_arr)
        metrics["rollout_stats/zero_tool_calls_pct"] = sum(1 for x in n_tool_calls_arr if x == 0) / len(n_tool_calls_arr)

    # --- task_result.stats 均值 ---
    for k, arr in task_stat_arrs.items():
        if not arr:
            continue
        metrics[f"task_stats/{k}_mean"] = statistics.mean(arr)
        if k in ("llm_errors", "tool_errors"):
            metrics[f"task_stats/{k}_rate"] = sum(1 for x in arr if x > 0) / len(arr)

    # --- finish_reason / stop_reason 分布 ---
    for fr, c in finish_reason_counter.items():
        # 防止 fr 含特殊字符
        key = re.sub(r"[^A-Za-z0-9_]+", "_", fr)[:32] or "unknown"
        metrics[f"finish_reason/{key}_pct"] = c / n
    for sr, c in stop_reason_counter.items():
        key = re.sub(r"[^A-Za-z0-9_]+", "_", sr)[:32] or "unknown"
        metrics[f"stop_reason/{key}_pct"] = c / n

    # --- 响应长度分布 ---
    if response_lens:
        metrics["rollout_stats/response_len_mean"] = statistics.mean(response_lens)
        metrics["rollout_stats/response_len_max"] = max(response_lens)
        metrics["rollout_stats/response_len_min"] = min(response_lens)
        metrics["rollout_stats/response_len_p50"] = statistics.median(response_lens)
        try:
            quantiles = statistics.quantiles(response_lens, n=10)
            metrics["rollout_stats/response_len_p90"] = quantiles[8]
            metrics["rollout_stats/response_len_p10"] = quantiles[0]
        except Exception:
            pass
        metrics.update(_bucketize_response_len(response_lens))

    metrics["rollout_stats/success_rate"] = success_count / n
    metrics["rollout_stats/error_rate"] = error_count / n
    metrics["rollout_stats/truncated_rate"] = truncated_count / n
    metrics["rollout_stats/payload_parse_error_rate"] = parse_errors / n
    metrics["rollout_stats/batch_size"] = float(n)

    swanlab.log(metrics, step=batch_id)
    logger.info(
        "[RolloutStats] batch=%d, reward=%.3f±%.3f, success=%.2f, regress=%.2f, fmt=%.2f, "
        "n_tool_calls=%.1f, resp_len=%.0f",
        batch_id,
        metrics.get("rollout_stats/reward_mean", 0.0),
        metrics.get("rollout_stats/reward_std", 0.0),
        metrics.get("rollout_stats/success_rate", 0.0),
        metrics.get("rollout_stats/acc_regression_rate", 0.0),
        metrics.get("sub/r_format_mean", 0.0),
        metrics.get("rollout_stats/n_tool_calls_mean", 0.0),
        metrics.get("rollout_stats/response_len_mean", 0.0),
    )


# ===========================================================================
# 核心监控循环
# ===========================================================================

# Reward 聚合时 sub-key 列表（与 reward.py 的 metrics_record 字段对齐）
_REWARD_NUMERIC_KEYS = (
    "r_total",
    "r_format",
    "r_after_score",
    "r_after_headroom",
    "r_acc_delta",
    "r_acc_delta_signed",
    "r_score_delta",
    "r_score_delta_signed",
    "r_context_diff",
    "n_tool_calls",
    "action_cost",
    "used_post_snapshot",
    "response_len",
)



def _record_split(record: dict) -> str:
    """Return train/eval split for records written by reward.py; old logs default to train."""
    split = str(record.get("split") or record.get("phase") or "").strip().lower()
    if split in {"eval", "val", "validation"}:
        return "eval"
    if record.get("is_eval") in (1, 1.0, True, "1", "true", "True"):
        return "eval"
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    split = str(metadata.get("split") or metadata.get("_memory_rl_split") or "").strip().lower()
    if split in {"eval", "val", "validation"}:
        return "eval"
    return "train"


def _reward_distribution_metrics(records: list[dict], prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in _REWARD_NUMERIC_KEYS:
        arr = [float(r[k]) for r in records if isinstance(r.get(k), (int, float))]
        if not arr:
            continue
        metrics[f"{prefix}/{k}_mean"] = statistics.mean(arr)
        metrics[f"{prefix}/{k}_std"] = statistics.pstdev(arr) if len(arr) > 1 else 0.0
        metrics[f"{prefix}/{k}_min"] = min(arr)
        metrics[f"{prefix}/{k}_max"] = max(arr)
    totals = [float(r.get("r_total", r.get("reward", 0.0))) for r in records]
    if totals:
        metrics[f"{prefix}/n_samples"] = float(len(totals))
        metrics[f"{prefix}/positive_rate"] = sum(1 for x in totals if x > 0) / len(totals)
        metrics[f"{prefix}/negative_rate"] = sum(1 for x in totals if x < 0) / len(totals)
        metrics[f"{prefix}/zero_rate"] = sum(1 for x in totals if x == 0) / len(totals)
        metrics[f"{prefix}/r_total_median"] = statistics.median(totals)
    return metrics


def _flush_eval_reward_records(records: list[dict], step: int, eval_idx: int) -> None:
    """Upload per-eval detailed reward decomposition under val_reward/* without polluting train reward/*."""
    if not records:
        return
    metrics = _reward_distribution_metrics(records, "val_reward")
    metrics["val_reward/eval_idx"] = float(eval_idx)
    metrics["val_reward/unique_prompts"] = float(len({str(r.get("prompt_uid", "")) for r in records if r.get("prompt_uid")}))
    swanlab.log(metrics, step=step)
    logger.info(
        "[ValReward step %d] n=%d, r_total mean=%.3f std=%.3f, pos=%.2f, tool_calls=%.1f",
        step,
        len(records),
        metrics.get("val_reward/r_total_mean", 0.0),
        metrics.get("val_reward/r_total_std", 0.0),
        metrics.get("val_reward/positive_rate", 0.0),
        metrics.get("val_reward/n_tool_calls_mean", 0.0),
    )


def _flush_eval_rollout_records(records: list[dict], step: int, eval_idx: int) -> None:
    """Upload eval rollout_records summary (task stats, finish reasons, response length) under val_rollout/*."""
    if not records:
        return
    rewards: list[float] = []
    response_lens: list[int] = []
    sub_arrs: dict[str, list[float]] = {k: [] for k in _SUB_REWARD_KEYS_NUMERIC}
    task_stat_arrs: dict[str, list[float]] = {k: [] for k in _TASK_STATS_NUMERIC}
    finish_reason_counter: dict[str, int] = {}
    stop_reason_counter: dict[str, int] = {}
    parse_errors = 0
    success_count = 0
    for rec in records:
        if isinstance(rec.get("reward"), (int, float)):
            rewards.append(float(rec["reward"]))
        if isinstance(rec.get("response_len"), (int, float)):
            response_lens.append(int(rec["response_len"]))
        sub = rec.get("sub_rewards") or {}
        if isinstance(sub, dict):
            for k, arr in sub_arrs.items():
                v = sub.get(k)
                if isinstance(v, (int, float)):
                    arr.append(float(v))
        payload = _try_parse_response_payload(rec.get("response", ""))
        if payload is None:
            parse_errors += 1
            continue
        sr = str(payload.get("stop_reason", "") or "")
        if sr:
            stop_reason_counter[sr] = stop_reason_counter.get(sr, 0) + 1
        tr = payload.get("task_result") or {}
        if isinstance(tr, dict):
            fr = str(tr.get("finish_reason", "") or "")
            if fr:
                finish_reason_counter[fr] = finish_reason_counter.get(fr, 0) + 1
            stats = tr.get("stats") or {}
            if isinstance(stats, dict):
                if stats.get("success") is True:
                    success_count += 1
                for k, arr in task_stat_arrs.items():
                    v = stats.get(k)
                    if isinstance(v, (int, float)):
                        arr.append(float(v))
    n = len(records) or 1
    metrics: dict[str, float] = {"val_rollout/eval_idx": float(eval_idx), "val_rollout/n_samples": float(n)}
    if rewards:
        metrics["val_rollout/reward_mean"] = statistics.mean(rewards)
        metrics["val_rollout/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        metrics["val_rollout/reward_median"] = statistics.median(rewards)
    if response_lens:
        metrics["val_rollout/response_len_mean"] = statistics.mean(response_lens)
        metrics["val_rollout/response_len_p50"] = statistics.median(response_lens)
        metrics["val_rollout/response_len_max"] = max(response_lens)
        metrics["val_rollout/response_len_min"] = min(response_lens)
    for k, arr in sub_arrs.items():
        if arr:
            metrics[f"val_sub/{k}_mean"] = statistics.mean(arr)
            metrics[f"val_sub/{k}_nonzero_pct"] = sum(1 for x in arr if x != 0) / len(arr)
    for k, arr in task_stat_arrs.items():
        if arr:
            metrics[f"val_task_stats/{k}_mean"] = statistics.mean(arr)
    for fr, c in finish_reason_counter.items():
        key = re.sub(r"[^A-Za-z0-9_]+", "_", fr)[:32] or "unknown"
        metrics[f"val_finish_reason/{key}_pct"] = c / n
    for sr, c in stop_reason_counter.items():
        key = re.sub(r"[^A-Za-z0-9_]+", "_", sr)[:32] or "unknown"
        metrics[f"val_stop_reason/{key}_pct"] = c / n
    metrics["val_rollout/success_rate"] = success_count / n
    metrics["val_rollout/payload_parse_error_rate"] = parse_errors / n
    swanlab.log(metrics, step=step)
    logger.info(
        "[ValRollout step %d] n=%d, reward=%.3f, success=%.2f, resp_len=%.0f",
        step,
        n,
        metrics.get("val_rollout/reward_mean", 0.0),
        metrics.get("val_rollout/success_rate", 0.0),
        metrics.get("val_rollout/response_len_mean", 0.0),
    )

def _flush_reward_step(
    records: list[dict],
    step: int,
    n_samples_per_prompt: int,
    effective_threshold: float,
    effective_window: deque,
    streaming_state: dict,
) -> None:
    """把一个训练 step 内的所有 reward 样本聚合成一组指标上报。

    * reward/<k>：每个 sub-key 上报 mean / std / min / max
    * reward_group/*：按 prompt_uid 精确分组（同一 prompt N 个 sample），算组内 std / advantage spread / effective ratio
    """
    if not records:
        return

    # ---------- per-step 标量统计 ----------
    metrics: dict[str, float] = _reward_distribution_metrics(records, "reward")
    totals = [float(r.get("r_total", r.get("reward", 0.0))) for r in records]

    # ---------- reward_group：按 prompt_uid 精确分组 ----------
    prompt_groups: dict[str, list[float]] = {}
    missing_uid = 0
    for r in records:
        uid = r.get("prompt_uid")
        rt = r.get("r_total", r.get("reward", 0.0))
        if not isinstance(rt, (int, float)):
            continue
        if uid:
            prompt_groups.setdefault(str(uid), []).append(float(rt))
        else:
            missing_uid += 1

    if prompt_groups:
        # 标准 GRPO/GSPO 路径：每个 prompt 的 N 个 sample 形成一个比较组。
        group_stds: list[float] = []
        group_means: list[float] = []
        group_spreads: list[float] = []
        adv_spreads: list[float] = []
        effective_count = 0
        # 只统计"完整"的组（凑齐 N 个 sample），避免被截断的组拉低 std
        complete_groups = [g for g in prompt_groups.values() if len(g) == n_samples_per_prompt]
        partial_groups = [g for g in prompt_groups.values() if len(g) != n_samples_per_prompt]

        target_groups = complete_groups if complete_groups else list(prompt_groups.values())

        for grp in target_groups:
            if len(grp) < 2:
                continue
            mean = statistics.mean(grp)
            std = statistics.pstdev(grp)
            group_stds.append(std)
            group_means.append(mean)
            group_spreads.append(max(grp) - min(grp))
            if std > effective_threshold:
                effective_count += 1
                advs = [(r - mean) / (std + 1e-6) for r in grp]
                adv_spreads.append(max(advs) - min(advs))
            else:
                adv_spreads.append(0.0)

        n_groups = len(target_groups)
        if n_groups > 0:
            eff_ratio = effective_count / n_groups
            effective_window.append(eff_ratio)
            metrics["reward_group/n_groups"] = float(n_groups)
            metrics["reward_group/n_partial_groups"] = float(len(partial_groups))
            metrics["reward_group/missing_prompt_uid"] = float(missing_uid)
            metrics["reward_group/effective_ratio"] = eff_ratio
            metrics["reward_group/effective_ratio_window"] = (
                statistics.mean(effective_window) if effective_window else 0.0
            )
            if group_stds:
                metrics["reward_group/std_mean"] = statistics.mean(group_stds)
                metrics["reward_group/std_max"] = max(group_stds)
                metrics["reward_group/std_min"] = min(group_stds)
            if group_spreads:
                metrics["reward_group/spread_mean"] = statistics.mean(group_spreads)
                metrics["reward_group/spread_max"] = max(group_spreads)
            if adv_spreads:
                metrics["reward_group/advantage_spread_mean"] = statistics.mean(adv_spreads)
                metrics["reward_group/advantage_spread_max"] = max(adv_spreads)
            if group_means:
                metrics["reward_group/mean_of_means"] = statistics.mean(group_means)
    else:
        # 兜底：拿不到 prompt_uid → 仍然按"流式凑 N 条"算，但显式打 warning，
        # 让看曲线的人知道这条曲线只是近似（顺序乱时不准）。
        if not streaming_state["warned"]:
            logger.warning(
                "[reward_group] reward_metrics 缺少 prompt_uid 字段，退化为按落盘顺序凑 %d 条算 group "
                "(GRPO 真实组方差不准)。请确保使用了带 prompt_uid 的 reward.py。",
                n_samples_per_prompt,
            )
            streaming_state["warned"] = True
        for rt in totals:
            streaming_state["buffer"].append(rt)
            if len(streaming_state["buffer"]) >= n_samples_per_prompt:
                grp = streaming_state["buffer"][:n_samples_per_prompt]
                streaming_state["buffer"] = streaming_state["buffer"][n_samples_per_prompt:]
                if len(grp) >= 2:
                    mean = statistics.mean(grp)
                    std = statistics.pstdev(grp)
                    metrics.setdefault("reward_group/std_mean_streaming", std)
                    metrics.setdefault("reward_group/spread_mean_streaming", max(grp) - min(grp))
                    metrics["reward_group/streaming_warning"] = 1.0

    swanlab.log(metrics, step=step)
    logger.info(
        "[Reward step %d] n=%d, r_total mean=%.3f std=%.3f, group_std_mean=%.3f, eff_ratio=%.2f",
        step,
        len(records),
        metrics.get("reward/r_total_mean", 0.0),
        metrics.get("reward/r_total_std", 0.0),
        metrics.get("reward_group/std_mean", 0.0),
        metrics.get("reward_group/effective_ratio", 0.0),
    )


def monitor(
    train_log: Path,
    reward_log: Path,
    rollout_records_log: Path,
    poll_interval: float = 5.0,
    gpu_poll_interval: float = 30.0,
    rollout_batch_size: int = 0,
    samples_per_step: int = 0,
    n_samples_per_prompt: int = 8,
):
    """实时监控日志并上传到 SwanLab.

    step 对齐说明
    --------------
    * train/* / val/* / val_progress/* / perf/* / rollout/*：直接采用 slime 输出的 step（=global_step / rollout_id）。
    * reward/* / sub/* / rollout_stats/*：原本按"每条 reward 行 +1"或"每 50 条 +1"，X 轴和 train/* 完全
      对不齐。现在改成按 ``samples_per_step = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`` 攒满一个
      训练 step 才聚合上报，X 轴 = 训练 step（reward_step / rollout_step），与 train/global_step 一致；eval reward 单独进入 val_reward/*。
    * reward_group/*：按 ``prompt_uid`` 精确分组（同一 prompt 的 N_SAMPLES_PER_PROMPT 个 rollout 落盘
      顺序不一定连续，按计数流式凑数算出来的 std 不是真的 GRPO 组方差）。同时输出"每个 step 内 effective
      ratio"和"滑窗 effective ratio"代替原来的累积比例（累积只能涨不能跌，看不出训练塌陷）。
    """
    logger.info(f"监控训练日志: {train_log}")
    logger.info(f"监控 reward 日志: {reward_log}")
    logger.info(f"监控 rollout 记录: {rollout_records_log}")
    logger.info(
        "聚合参数: samples_per_step=%d, n_samples_per_prompt=%d, rollout_records_batch=%s",
        samples_per_step,
        n_samples_per_prompt,
        rollout_batch_size or samples_per_step or "auto",
    )

    train_pos = 0
    reward_pos = 0
    rollout_records_pos = 0

    global_step = 0
    eval_step_count = 0

    # ----- reward_metrics.jsonl 聚合到训练 step -----
    # 按 samples_per_step 攒一个 step 的所有样本，然后整体上报。
    reward_step_buffer: list[dict] = []
    eval_reward_buffer: list[dict] = []
    pending_eval_steps: deque[int] = deque()
    reward_step_idx = 0  # 当前累计满了多少个训练 step
    eval_reward_flush_idx = 0
    ready_eval_rollout_step = 0
    total_reward_records = 0  # 累计读到的 reward 行数（仅用于日志）
    total_eval_reward_records = 0

    # ----- reward_group 按 prompt 精确分组 -----
    # 同一 step 内：prompt_uid -> [r_total, ...]
    EFFECTIVE_GROUP_STD_THRESHOLD = float(os.environ.get("EFFECTIVE_GROUP_STD", "0.01"))
    effective_ratio_window: deque = deque(maxlen=int(os.environ.get("EFFECTIVE_RATIO_WINDOW", "20")))
    # 兜底：拿不到 prompt_uid 时退化成"按 N 条流式凑组"，但会标 streaming 警告
    streaming_group_buffer: list[float] = []
    streaming_group_idx = 0
    streaming_warning_logged = False

    error_window: deque = deque(maxlen=2000)
    error_counts: dict[str, int] = {}

    # ----- rollout_records.jsonl 聚合到训练 step -----
    rollout_record_buffer: list[dict] = []
    eval_rollout_record_buffer: list[dict] = []
    rollout_step_idx = 0
    # 当 rollout_batch_size 显式传入时优先用它，否则按 samples_per_step（标准做法）。
    record_flush_size = rollout_batch_size if rollout_batch_size > 0 else max(samples_per_step, 1)

    last_gpu_poll = 0.0
    train_start_time = time.time()

    while True:
        now = time.time()

        # ============================================================
        # 1. 训练日志: train / eval / rollout / perf / errors
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

                # --- 训练指标 ---
                train_data = parse_train_metric_line(line)
                if train_data:
                    step = train_data.get("_step", global_step)
                    global_step = step
                    metrics: dict[str, float] = {}
                    key_mapping = {
                        "train/entropy_loss": "train/entropy",
                        "train/kl_loss": "train/kl_loss",
                        "train/pg_loss": "train/pg_loss",
                        "train/pg_clipfrac": "train/pg_clipfrac",
                        "train/ppo_kl": "train/ppo_kl",
                        "train/loss": "train/total_loss",
                        "train/grad_norm": "train/grad_norm",
                        "train/value_loss": "train/value_loss",
                        "train/approx_kl": "train/approx_kl",
                        "train/policy_kl": "train/policy_kl",
                        "train/clip_fraction": "train/clip_fraction",
                    }
                    for src, dst in key_mapping.items():
                        if src in train_data and isinstance(train_data[src], (int, float)):
                            metrics[dst] = float(train_data[src])

                    # 学习率
                    for key, val in train_data.items():
                        if "lr-pg" in key and isinstance(val, (int, float)):
                            metrics["train/lr"] = float(val)
                            break
                        if key == "lr" and isinstance(val, (int, float)):
                            metrics["train/lr"] = float(val)

                    # 动态捕获其他 train/ 数值
                    for key, val in train_data.items():
                        if key.startswith("train/") and isinstance(val, (int, float)) and key not in key_mapping:
                            metrics[key] = float(val)

                    metrics["progress/global_step"] = float(step)
                    metrics["progress/elapsed_minutes"] = (now - train_start_time) / 60.0

                    if metrics:
                        swanlab.log(metrics, step=step)
                    continue

                # --- 验证 ---
                eval_data = parse_eval_metric_line(line)
                if eval_data:
                    eval_step_count += 1
                    step = eval_data.get("_eval_step", eval_step_count)
                    metrics = {}
                    eval_key_mapping = {
                        "eval/loss": "val/loss",
                        "eval/entropy_loss": "val/entropy",
                        "eval/kl_loss": "val/kl_loss",
                        "eval/pg_loss": "val/pg_loss",
                        "eval/reward": "val/reward",
                        "eval/reward_mean": "val/reward_mean",
                        "eval/reward_std": "val/reward_std",
                        "eval/response_length": "val/response_length",
                        "eval/advantages": "val/advantages",
                        "eval/log_probs": "val/log_probs",
                        "eval/ref_log_probs": "val/ref_log_probs",
                        "eval/rewards": "val/rewards",
                        "eval/raw_reward": "val/raw_reward",
                        "loss": "val/loss",
                        "reward": "val/reward",
                        "rewards": "val/rewards",
                        "raw_reward": "val/raw_reward",
                        "response_lengths": "val/response_length",
                    }
                    for src, dst in eval_key_mapping.items():
                        if src in eval_data and isinstance(eval_data[src], (int, float)):
                            metrics[dst] = float(eval_data[src])
                    for key, val in eval_data.items():
                        if key.startswith(("eval/", "val/")) and isinstance(val, (int, float)):
                            dst = key.replace("eval/", "val/")
                            if dst not in metrics:
                                metrics[dst] = float(val)
                    if metrics:
                        log_step = global_step or step
                        swanlab.log(metrics, step=log_step)
                        pending_eval_steps.append(log_step)
                        logger.info(f"[Val] step={step}, metrics={list(metrics.keys())}")
                    continue

                # --- 验证进度（tqdm）---
                eval_progress = parse_eval_progress_line(line)
                if eval_progress:
                    swanlab.log(eval_progress, step=global_step or reward_step_idx or 0)
                    continue

                # --- Rollout 指标 ---
                rollout_data = parse_rollout_metric_line(line)
                if rollout_data:
                    rid = rollout_data.get("_rollout_id", 0)
                    metrics = {}
                    # slime 的 rollout/{rewards, response_lengths, advantages, ...} 经常是 list/array
                    # 之前直接 isinstance(...,(int,float)) 会全部丢弃 → 这里对 list 单独处理：
                    # 数值标量直接用；list 则 mean / std / min / max / count 都出一份。
                    list_only_keys = (
                        "rollout/rewards",
                        "rollout/raw_reward",
                        "rollout/response_lengths",
                        "rollout/advantages",
                        "rollout/log_probs",
                        "rollout/ref_log_probs",
                        "rollout/kl",
                        "rollout/entropy",
                    )
                    for key, val in rollout_data.items():
                        if not key.startswith("rollout/"):
                            continue
                        if isinstance(val, bool):
                            continue
                        if isinstance(val, (int, float)):
                            metrics[key] = float(val)
                        elif isinstance(val, (list, tuple)) and val and key in list_only_keys:
                            try:
                                arr = [float(x) for x in val if isinstance(x, (int, float))]
                            except Exception:
                                arr = []
                            if arr:
                                metrics[f"{key}_mean"] = statistics.mean(arr)
                                metrics[f"{key}_std"] = statistics.pstdev(arr) if len(arr) > 1 else 0.0
                                metrics[f"{key}_min"] = min(arr)
                                metrics[f"{key}_max"] = max(arr)
                                metrics[f"{key}_count"] = float(len(arr))
                    if metrics:
                        # rid 是 slime 内部 rollout_id，与 global_step 一致（一个 rollout = 一个训练 step）
                        swanlab.log(metrics, step=rid)
                    continue

                # --- Perf ---
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
                        "perf/rollout_time": "perf/rollout_time",
                        "perf/reward_time": "perf/reward_time",
                        "perf/generate_time": "perf/generate_time",
                        "perf/samples_per_sec": "perf/samples_per_sec",
                    }
                    for src, dst in perf_keys.items():
                        if src in perf_data and isinstance(perf_data[src], (int, float)):
                            metrics[dst] = float(perf_data[src])
                    for key, val in perf_data.items():
                        if (
                            key.startswith("perf/")
                            and isinstance(val, (int, float))
                            and key not in perf_keys
                        ):
                            metrics[key] = float(val)
                    if metrics:
                        swanlab.log(metrics, step=rid)
                    continue

                # --- 错误统计 ---
                etype = parse_error_line(line)
                if etype:
                    error_window.append((now, etype))
                    error_counts[etype] = error_counts.get(etype, 0) + 1

        # ============================================================
        # 2. reward_metrics.jsonl: 攒满一个训练 step 再聚合上报
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

                split = _record_split(record)
                if split == "eval":
                    eval_reward_buffer.append(record)
                    total_eval_reward_records += 1
                    continue

                reward_step_buffer.append(record)
                total_reward_records += 1

                # 攒满 samples_per_step 就 flush 成一个训练 step 的指标
                if samples_per_step > 0 and len(reward_step_buffer) >= samples_per_step:
                    reward_step_idx += 1
                    streaming_state = {
                        "buffer": streaming_group_buffer,
                        "warned": streaming_warning_logged,
                        "idx": streaming_group_idx,
                    }
                    _flush_reward_step(
                        reward_step_buffer[:samples_per_step],
                        reward_step_idx,
                        n_samples_per_prompt=n_samples_per_prompt,
                        effective_threshold=EFFECTIVE_GROUP_STD_THRESHOLD,
                        effective_window=effective_ratio_window,
                        streaming_state=streaming_state,
                    )
                    streaming_warning_logged = streaming_state["warned"]
                    streaming_group_buffer = streaming_state["buffer"]
                    reward_step_buffer = reward_step_buffer[samples_per_step:]

        while pending_eval_steps and eval_reward_buffer:
            eval_reward_flush_idx += 1
            eval_step = pending_eval_steps.popleft()
            ready_eval_rollout_step = eval_step
            _flush_eval_reward_records(eval_reward_buffer, eval_step, eval_reward_flush_idx)
            eval_reward_buffer = []

        # ============================================================
        # 3. rollout_records.jsonl: 攒满一个训练 step 后汇总
        # ============================================================
        if rollout_records_log.exists():
            with open(rollout_records_log, "r", errors="replace") as f:
                f.seek(rollout_records_pos)
                new_lines = f.readlines()
                rollout_records_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                split = _record_split(record)
                if split == "eval":
                    eval_rollout_record_buffer.append(record)
                    continue

                rollout_record_buffer.append(record)
                if record_flush_size > 0 and len(rollout_record_buffer) >= record_flush_size:
                    rollout_step_idx += 1
                    _log_rollout_record_stats(rollout_record_buffer[:record_flush_size], rollout_step_idx)
                    rollout_record_buffer = rollout_record_buffer[record_flush_size:]

        if eval_rollout_record_buffer and ready_eval_rollout_step:
            _flush_eval_rollout_records(eval_rollout_record_buffer, ready_eval_rollout_step, max(eval_reward_flush_idx, 1))
            eval_rollout_record_buffer = []
            ready_eval_rollout_step = 0

        # ============================================================
        # 4. GPU
        # ============================================================
        if now - last_gpu_poll >= gpu_poll_interval:
            last_gpu_poll = now
            gpu_metrics = get_gpu_metrics()
            if gpu_metrics:
                swanlab.log(gpu_metrics, step=global_step or reward_step_idx or 1)

        # ============================================================
        # 5. 错误率（最近 5 分钟滑窗）
        # ============================================================
        if error_window:
            cutoff = now - 300
            recent_errors = [e for e in error_window if e[0] >= cutoff]
            if recent_errors:
                error_metrics: dict[str, float] = {"errors/total_5min": float(len(recent_errors))}
                type_counts: dict[str, int] = {}
                for _, etype in recent_errors:
                    type_counts[etype] = type_counts.get(etype, 0) + 1
                for etype, count in type_counts.items():
                    error_metrics[f"errors/{etype}_5min"] = float(count)
                # 累计计数
                for etype, count in error_counts.items():
                    error_metrics[f"errors/{etype}_total"] = float(count)
                swanlab.log(error_metrics, step=global_step or reward_step_idx or 1)

        time.sleep(poll_interval)


# ===========================================================================
# 入口
# ===========================================================================

def _read_env_float(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


def main():
    parser = argparse.ArgumentParser(description="SwanLab 实时训练监控 — Consolidate GRPO/GSPO")
    parser.add_argument("--log-dir", type=str, required=True, help="训练日志目录")
    parser.add_argument("--run-name", type=str, default=None, help="SwanLab 运行名称")
    parser.add_argument("--task-version", type=str, default=None, help="task version")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="日志轮询间隔(秒)")
    parser.add_argument("--gpu-poll-interval", type=float, default=30.0, help="GPU 指标采集间隔(秒)")
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=0,
        help=(
            "rollout_records 每多少条聚合一次。建议设成 ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT "
            "（即一个训练 step 的样本总数），这样 rollout_stats/* 的 X 轴就和 train/global_step 对齐。"
            "默认 0=自动跟随 samples_per_step。"
        ),
    )
    parser.add_argument(
        "--samples-per-step",
        type=int,
        default=0,
        help=(
            "一个训练 step 包含多少个 reward 样本，等于 ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT。"
            "默认 0 时从环境变量自动推断。"
        ),
    )
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=0,
        help="GRPO/GSPO 每个 prompt 采样多少 rollout（用于 reward_group 分组），默认从 N_SAMPLES_PER_PROMPT env 读取。",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    train_log = log_dir / "train.log"
    if not train_log.exists():
        output_dir = log_dir.parent
        run_logs = sorted(output_dir.parent.glob("run*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_logs:
            train_log = run_logs[0]

    reward_log = log_dir / "reward_metrics.jsonl"
    rollout_records_log = log_dir / "rollout_records.jsonl"

    task_version = args.task_version or os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{task_version}_consolidate_{timestamp}"

    os.environ["SWANLAB_API_KEY"] = SWANLAB_API_KEY

    # ---- 推导 samples_per_step / n_samples_per_prompt ----
    n_samples_per_prompt = args.n_samples_per_prompt or int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))
    rollout_batch_size_train = int(os.environ.get("ROLLOUT_BATCH_SIZE", "0"))
    samples_per_step = args.samples_per_step
    if samples_per_step <= 0:
        if rollout_batch_size_train > 0:
            samples_per_step = rollout_batch_size_train * n_samples_per_prompt
        else:
            # 兜底：用 launch 传给 monitor 的 --rollout-batch-size（=launch 计算好的 samples_per_step）
            samples_per_step = max(args.rollout_batch_size, n_samples_per_prompt)
    logger.info(
        "step 对齐: samples_per_step=%d (= ROLLOUT_BATCH_SIZE %d × N_SAMPLES_PER_PROMPT %d)",
        samples_per_step,
        rollout_batch_size_train,
        n_samples_per_prompt,
    )

    # 把 launch 脚本注入的关键超参一起记到 swanlab config
    cfg = {
        "model": os.environ.get("HF_CHECKPOINT", "Qwen3.6-27B"),
        "task": "consolidate",
        "task_version": task_version,
        "log_dir": str(log_dir),
        "train_log": str(train_log),
        "reward_log": str(reward_log),
        "rollout_records_log": str(rollout_records_log),
        "poll_interval": args.poll_interval,
        "gpu_poll_interval": args.gpu_poll_interval,
        "rollout_batch_size_monitor": args.rollout_batch_size,
        "rollout_batch_size_train": rollout_batch_size_train,
        "samples_per_step": samples_per_step,
        "n_samples_per_prompt": n_samples_per_prompt,
        "advantage_estimator": os.environ.get("ADVANTAGE_ESTIMATOR", ""),
        "kl_loss_coef": _read_env_float("KL_LOSS_COEF", 0.0),
        "entropy_coef": _read_env_float("ENTROPY_COEF", 0.0),
        "eps_clip": _read_env_float("EPS_CLIP", 0.0),
        "eps_clip_high": _read_env_float("EPS_CLIP_HIGH", 0.0),
        "lr": _read_env_float("LR", 0.0),
        "lr_warmup_iters": int(_read_env_float("LR_WARMUP_ITERS", 0)),
        "rollout_temperature": _read_env_float("ROLLOUT_TEMPERATURE", 0.0),
        "reward_weights": {
            "r_acc_delta": _read_env_float("CONSOLIDATE_REWARD_W_ACC_DELTA", 0.50),
            "r_score_delta": _read_env_float("CONSOLIDATE_REWARD_W_SCORE_DELTA", 0.25),
            "r_after": _read_env_float("CONSOLIDATE_REWARD_W_AFTER", 0.15),
            "r_context_diff": _read_env_float("CONSOLIDATE_REWARD_W_CONTEXT_DIFF", 0.05),
            "r_format": _read_env_float("CONSOLIDATE_REWARD_W_FORMAT", 0.05),
        },
        "reward_signed": os.environ.get("CONSOLIDATE_REWARD_LEGACY_POSITIVE", "0") != "1",
        "reward_use_headroom": os.environ.get("CONSOLIDATE_REWARD_USE_HEADROOM", "1") == "1",
        "reward_action_cost_per_call": _read_env_float("CONSOLIDATE_REWARD_ACTION_COST", 0.0),
        "reward_action_free_calls": int(_read_env_float("CONSOLIDATE_REWARD_FREE_CALLS", 8)),
        "effective_group_std": _read_env_float("EFFECTIVE_GROUP_STD", 0.01),
        "effective_ratio_window": int(_read_env_float("EFFECTIVE_RATIO_WINDOW", 20)),
        "eval_interval": int(_read_env_float("EVAL_INTERVAL", 0)),
        "n_samples_per_eval_prompt": int(_read_env_float("N_SAMPLES_PER_EVAL_PROMPT", 1)),
        "monitored_metrics": [
            "train/*", "val/*", "val_progress/*", "rollout/*", "perf/*",
            "reward/*", "reward_group/*", "val_reward/*", "val_rollout/*", "rollout_stats/*",
            "sub/*", "val_sub/*", "task_stats/*", "val_task_stats/*",
            "finish_reason/*", "stop_reason/*", "val_finish_reason/*", "val_stop_reason/*",
            "system/*", "errors/*", "progress/*",
        ],
    }

    swanlab.init(project="memory_ai_rl", experiment_name=run_name, mode="cloud", config=cfg)
    logger.info(f"SwanLab 初始化: project=memory_ai_rl, run={run_name}")
    try:
        logger.info(f"SwanLab URL: {swanlab.get_url()}")
    except Exception:
        pass
    logger.info(
        "监控指标组: train, val, val_progress, rollout, perf, reward, reward_group, val_reward, val_rollout, rollout_stats, "
        "sub, val_sub, task_stats, val_task_stats, finish_reason, stop_reason, system, errors, progress"
    )

    try:
        monitor(
            train_log,
            reward_log,
            rollout_records_log,
            poll_interval=args.poll_interval,
            gpu_poll_interval=args.gpu_poll_interval,
            rollout_batch_size=args.rollout_batch_size,
            samples_per_step=samples_per_step,
            n_samples_per_prompt=n_samples_per_prompt,
        )
    except KeyboardInterrupt:
        logger.info("监控终止")
    finally:
        swanlab.finish()
        logger.info("SwanLab 已关闭")


if __name__ == "__main__":
    main()
