# `llm_gateway/rl/data_gen/` — 数据生成（占位）

## 现状

数据生成（trajectory 摄入 + `.cbsnap` 落盘）目前仍在老 workspace 运行：

```
/data/home/trevzhang/projects/memory-ai-agent-workspace/src/data_gen/
└── ingest_snapshot/
    ├── run_generate.py         # 主入口：trajectories.jsonl → .cbsnap + ingest_snapshots.jsonl
    └── write_ops_logger.py
```

## 数据复用方式

老 workspace 产出的 `.cbsnap` 数据集（如 `rl_data_test_2/`、
`ingest_full_natural_probes_test_1/`）通过路径方式被本工程的训练侧消费，**不入库**：

- 通过环境变量 `SNAPSHOT_DATA_ROOT=/path/to/rl_data_test_2` 指定数据集根
- `llm_gateway.rl.rl_env.snapshot_session.SnapshotSession(data_root=SNAPSHOT_DATA_ROOT)`
  即可加载

## 后续计划

需要把 `src/data_gen/` 吸收进来时，再按下方目标布局迁入：

```
llm_gateway/rl/data_gen/
├── ingest_snapshot/
│   ├── run_generate.py
│   └── write_ops_logger.py
├── conflict_injection/
└── mvcc_pipeline/
```
