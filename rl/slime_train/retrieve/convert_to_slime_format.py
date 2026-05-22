#!/usr/bin/env python3
"""
将快照数据 (.cbsnap) + query 探针问题转换为 slime 框架所需的 JSONL 格式。

slime 数据格式:
{
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "label": null,
    "metadata": {"ground_truth": {...}}
}

工作流:
1. 加载 .cbsnap 快照 → 反序列化为 MemoryEnv 可识别的三后端状态
2. 从快照中提取当前 memory 内容摘要（FS tree、Vec collections、Graph stats）
3. 结合 query 探针问题，组装为 RetrieveT3Task 的 system prompt + user prompt
4. 输出 slime 格式的训练数据

使用:
    python convert_to_slime_format.py \
        --input /path/to/slime_train/data/rl_data_test_2/rl_data.jsonl \
        --output data/rl_train.jsonl \
        --eval_output data/rl_val.jsonl \
        --eval_ratio 0.05

    # --data-root 默认为 --input 同级目录（包含 snapshots/, workdirs/ 等）
    # 如需指定不同目录:
    python convert_to_slime_format.py \
        --input rl_data.jsonl \
        --data-root /path/to/rl_data_test_2/ \
        --output data/rl_train.jsonl
"""

import argparse
import json
import os
import re
import sys
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta


from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths
ensure_workspace_paths(__file__)

from llm_gateway.rl.slime_train._t3_assets import build_fs_structure_from_files
from llm_gateway.rl.slime_train._t3_assets import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def load_fs_from_git(data_root: str, traj_id: str, snapshot_id: str, index_cache: dict) -> dict:
    """通过 git show 从 workdirs 中获取指定 commit 的 FS 文件内容。

    不需要 git worktree（避免磁盘写入），直接用 git show {commit}:{path} 读取。

    Returns:
        {"files": {"path/to/file.md": "content", ...}}
    """
    import subprocess

    # 从 ingest_snapshots.jsonl 索引获取 commit_hash
    if not index_cache:
        _load_index(data_root, index_cache)

    record = index_cache.get(snapshot_id)
    if not record:
        raise KeyError(f"索引中找不到 snapshot_id={snapshot_id}")

    commit_hash = record.get("commit_hash", "")
    if not commit_hash:
        raise ValueError(f"snapshot_id={snapshot_id} 没有 commit_hash")

    git_dir = os.path.join(data_root, "workdirs", traj_id)
    if not os.path.isdir(os.path.join(git_dir, ".git")):
        raise FileNotFoundError(f"git repo 不存在: {git_dir}")

    # 用 git ls-tree 列出该 commit 下的所有文件
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit_hash],
        cwd=git_dir,
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-tree 失败: {result.stderr.strip()}")

    files = {}
    for filepath in result.stdout.strip().split("\n"):
        if not filepath:
            continue
        # 用 git show 获取文件内容
        show_result = subprocess.run(
            ["git", "show", f"{commit_hash}:{filepath}"],
            cwd=git_dir,
            capture_output=True, text=True, timeout=10,
        )
        if show_result.returncode == 0:
            files[filepath] = show_result.stdout

    return {"files": files}


def _load_index(data_root: str, index_cache: dict) -> None:
    """加载 ingest_snapshots.jsonl 到 cache。"""
    index_path = os.path.join(data_root, "ingest_snapshots.jsonl")
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("snapshot_id", "")
            if sid:
                index_cache[sid] = record


def build_fs_structure_from_state(fs_state: dict) -> str:
    """从 FS 状态中构建文件系统结构摘要（供 system prompt 注入）。"""
    return build_fs_structure_from_files(
        fs_state.get("files", {}),
        max_files=50,
        include_truncation_note=True,
    )


def _infer_session_time_from_snapshot(snapshot_state: dict) -> str:
    """从快照状态中推断 session_time：取所有后端中最新的 timestamp，再加一段偏移。

    扫描来源：
    1. FS 文件内容中的行级时间戳 [session_time | event_time]
    2. Vec entries 的 metadata.ingest_time / metadata.occurred_at
    3. Graph edges 的 properties.ingest_time / properties.occurred_at
    4. 快照 meta 中的时间信息

    取最大时间戳后 +1 小时，确保在所有已有记忆之后。
    """
    timestamps: list[str] = []

    # 1. 从 FS 文件内容中提取时间戳
    fs_state = snapshot_state.get("fs", {})
    files = fs_state.get("files", {})
    for content in files.values():
        if not content:
            continue
        # 匹配 [YYYY-MM-DD HH:MM:SS ... | ...] 格式
        for m in re.finditer(r'\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', content):
            timestamps.append(m.group(1))
        # 匹配 [YYYY-MM-DD] 格式
        for m in re.finditer(r'\[(\d{4}-\d{2}-\d{2})\]', content):
            timestamps.append(m.group(1))

    # 2. 从 Vec entries 提取时间
    vec_state = snapshot_state.get("vec", {})
    collections = vec_state.get("collections", {}) if isinstance(vec_state, dict) else {}
    for entries in collections.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            meta = entry.get("metadata", {}) or {}
            for field in ("ingest_time", "occurred_at"):
                val = meta.get(field, "")
                if isinstance(val, str) and val and val != "/":
                    timestamps.append(val)
                elif isinstance(val, list):
                    timestamps.extend(v for v in val if isinstance(v, str) and v)

    # 3. 从 Graph edges 提取时间
    graph_state = snapshot_state.get("graph", {})
    edges = graph_state.get("edges", []) if isinstance(graph_state, dict) else []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        props = edge.get("properties", {}) or {}
        for field in ("ingest_time", "occurred_at"):
            val = props.get(field, "")
            if isinstance(val, str) and val and val != "/":
                timestamps.append(val)

    # 4. 从快照 meta 提取
    meta = snapshot_state.get("meta", {})
    if meta.get("timestamp"):
        timestamps.append(str(meta["timestamp"]))

    # 解析所有时间戳，取最大值
    max_dt: datetime | None = None
    for ts in timestamps:
        dt = _try_parse_timestamp(ts)
        if dt and (max_dt is None or dt > max_dt):
            max_dt = dt

    # 默认 fallback
    if max_dt is None:
        max_dt = datetime(2026, 1, 1, 12, 0, 0)

    # +1 小时，确保在所有已有记忆之后
    session_dt = max_dt + timedelta(hours=1)
    # 格式与 RetrieveT3Task 一致：YYYY-MM-DD HH:MM:SS, Weekday
    weekday = session_dt.strftime("%a")
    return session_dt.strftime(f"%Y-%m-%d %H:%M:%S, {weekday}")


def _try_parse_timestamp(ts: str) -> datetime | None:
    """尝试将时间字符串解析为 datetime 对象。"""
    ts = ts.strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S, %a",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y-%m",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts[:len(datetime.now().strftime(fmt))], fmt)
        except (ValueError, IndexError):
            continue
    # 尝试只取年月日部分
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", ts)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def build_prompt_for_sample(
    snapshot_state: dict,
    probe: dict,
) -> list[dict[str, str]]:
    """构建单条训练样本的 prompt（system + user）。

    Args:
        snapshot_state: 解码后的快照状态
        probe: raw data 中的单条探针:
            - probe_query: 问题文本
            - alt_queries: 备选查询

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    # snapshot_state = {"files": {...}} 或 {"fs": {"files": {...}}}
    fs_state = snapshot_state if "files" in snapshot_state else snapshot_state.get("fs", {})
    fs_structure = build_fs_structure_from_state(fs_state)

    # 构建 system prompt（与 RetrieveT3Task._rewrite_queries_llm 一致）
    system_prompt = QUERY_REWRITE_SYSTEM_PROMPT.replace("{fs_structure}", fs_structure)

    # 构建 user prompt
    question = probe.get("probe_query", "")
    # session_time: 从快照中推断（取最新 timestamp 之后）
    session_time = _infer_session_time_from_snapshot(snapshot_state)
    user_prompt = QUERY_REWRITE_USER_TEMPLATE.replace(
        "{conversation}", question
    ).replace("{current_time}", session_time)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def convert_sample(snapshot_state: dict, probe: dict, record_meta: dict) -> dict:
    """将单条快照 + 探针问题转为 slime 格式。

    Args:
        snapshot_state: 解码后的快照状态
        probe: 来自 probes_by_task.retrieve 的单条探针:
            - answerable: bool
            - question_type: "fill_in_the_blank" | "multiple_choice"
            - probe_query: 问题文本
            - alt_queries: 备选查询列表
            - ground_truth: 答案（列表或字符串）
            - options: {"A": "...", "B": "...", ...}  (选择题)
            - source_evidence: str
            - probe_type: "atomic" | "entity" | "numerical"
            - task_target: "retrieve"
            - generation_mode: str
        record_meta: 外层记录元信息:
            - snapshot_id, trajectory_id, user_id, session_id
    """
    prompt = build_prompt_for_sample(snapshot_state, probe)

    # 映射 question_type → answer_type
    question_type = probe.get("question_type", "fill_in_the_blank")
    if question_type == "multiple_choice":
        answer_type = "mcq"
    else:
        answer_type = "fill"

    # 构建 ground_truth
    raw_gt = probe.get("ground_truth", "")
    if answer_type == "mcq":
        # 选择题：ground_truth 是正确选项字母（如 "A"）
        correct_option = raw_gt if isinstance(raw_gt, str) else str(raw_gt)
        answer = correct_option
    else:
        # 填空题：ground_truth 是可接受答案列表
        if isinstance(raw_gt, list):
            answer = raw_gt[0] if raw_gt else ""
        else:
            answer = str(raw_gt)

    ground_truth = {
        "question": probe.get("probe_query", ""),
        "answer": answer,
        "answer_type": answer_type,
        "alt_queries": probe.get("alt_queries", []),
        "source_evidence": probe.get("source_evidence", ""),
        "probe_type": probe.get("probe_type", ""),
    }

    # 填空题：将所有可接受答案保存（供 reward 函数做更灵活的匹配）
    if answer_type == "fill" and isinstance(raw_gt, list):
        ground_truth["acceptable_answers"] = raw_gt

    # 选择题：保存选项和正确答案
    if answer_type == "mcq":
        options = probe.get("options", {})
        # 将 dict {"A": "...", "B": "..."} 转为列表 ["A. ...", "B. ..."]
        if isinstance(options, dict):
            ground_truth["options"] = [f"{k}. {v}" for k, v in sorted(options.items())]
        else:
            ground_truth["options"] = options
        ground_truth["correct_option"] = correct_option

    return {
        "prompt": prompt,
        "label": None,
        "metadata": {
            "task": "retrieve",
            "query": probe.get("probe_query", ""),
            "ground_truth": ground_truth,
            "snapshot_id": record_meta.get("snapshot_id", ""),
            "traj_id": record_meta.get("trajectory_id", ""),
            "user_id": record_meta.get("user_id", ""),
            "session_id": record_meta.get("session_id", ""),
            "answerable": probe.get("answerable", True),
            "probe_type": probe.get("probe_type", ""),
            "generation_mode": probe.get("generation_mode", ""),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="转换 raw data (rl_data.jsonl) 为 slime 格式 — retrieve 任务")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件 (rl_data.jsonl)，每行一个 step record")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据集根目录（包含 snapshots/, workdirs/ 等）。"
                             "默认与 --input 同级目录")
    parser.add_argument("--output", type=str, required=True,
                        help="训练数据输出路径")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="评估数据输出路径")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="评估集比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 数据集根目录：默认与 input 文件同级
    data_root = args.data_root or os.path.dirname(os.path.abspath(args.input))

    rng = random.Random(args.seed)

    # 加载 raw data（每行一个 step record）
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info(f"加载了 {len(records)} 条 step records")

    # 统计 retrieve 探针总数
    total_probes = sum(
        len(r.get("probes_by_task", {}).get("retrieve", []))
        for r in records
    )
    logger.info(f"共 {total_probes} 条 retrieve 探针问题")

    # 按 snapshot_id 缓存 FS 状态
    snapshot_cache: dict[str, dict] = {}
    # ingest_snapshots.jsonl 索引缓存
    index_cache: dict[str, dict] = {}

    # 转换：遍历每个 record，从 probes_by_task.retrieve 提取探针
    converted = []
    skipped = 0
    for record in records:
        snap_id = record.get("snapshot_id", "")
        traj_id = record.get("trajectory_id", "")

        # 从 probes_by_task.retrieve 获取探针
        retrieve_probes = record.get("probes_by_task", {}).get("retrieve", [])
        if not retrieve_probes:
            continue

        if not snap_id or not traj_id:
            skipped += len(retrieve_probes)
            continue

        # 通过 git show 获取 FS 文件内容（无磁盘写入）
        if snap_id not in snapshot_cache:
            try:
                fs_state = load_fs_from_git(data_root, traj_id, snap_id, index_cache)
                snapshot_cache[snap_id] = fs_state
            except Exception as e:
                logger.warning(f"加载 FS 失败 {traj_id}/{snap_id}: {e}")
                skipped += len(retrieve_probes)
                continue

        state = snapshot_cache[snap_id]

        # 外层元信息
        record_meta = {
            "snapshot_id": snap_id,
            "trajectory_id": traj_id,
            "user_id": record.get("user_id", ""),
            "session_id": record.get("session_id", ""),
        }

        # 展开 retrieve probes
        for probe in retrieve_probes:
            try:
                sample = convert_sample(state, probe, record_meta)
                converted.append(sample)
            except Exception as e:
                logger.warning(f"转换 probe 失败: {e}")
                skipped += 1

    logger.info(f"转换成功 {len(converted)} 条，跳过 {skipped} 条")

    # 拆分 train/eval
    rng.shuffle(converted)
    if args.eval_output and args.eval_ratio > 0:
        n_eval = max(1, int(len(converted) * args.eval_ratio))
        eval_data = converted[:n_eval]
        train_data = converted[n_eval:]
    else:
        train_data = converted
        eval_data = []

    # 写入
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"训练数据: {len(train_data)} 条 → {args.output}")

    if eval_data and args.eval_output:
        os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)
        with open(args.eval_output, "w", encoding="utf-8") as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"评估数据: {len(eval_data)} 条 → {args.eval_output}")


if __name__ == "__main__":
    main()
