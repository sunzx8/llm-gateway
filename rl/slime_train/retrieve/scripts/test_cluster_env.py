#!/usr/bin/env python3
"""在集群容器内验证 reward 链路可用性。"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
RETRIEVE_ROOT = PROJECT_ROOT / "slime_train" / "retrieve"
WORKSPACE_SRC = Path(os.environ.get("WORKSPACE_SRC", PROJECT_ROOT / "src"))
for path in (str(PROJECT_ROOT), str(RETRIEVE_ROOT), str(WORKSPACE_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Test 1: import
print("=== Test 1: import ===")
from llm_gateway.rl.slime_train.tasks.retrieve_reward.reward import compute_r_format, compute_r_query_quality
print(f"  r_format('{{}}') = {compute_r_format('{}')}")

from snapshot_loader import SnapshotSession
print("  SnapshotSession imported OK")

# Test 2: snapshot load
print("\n=== Test 2: snapshot load ===")
DATA_ROOT = os.environ.get("RETRIEVE_SNAPSHOT_DATA_ROOT")
TRAJ_ID = os.environ.get("RETRIEVE_TEST_TRAJ_ID")
SNAPSHOT_ID = os.environ.get("RETRIEVE_TEST_SNAPSHOT_ID")
if not DATA_ROOT or not TRAJ_ID or not SNAPSHOT_ID:
    raise RuntimeError(
        "请设置 RETRIEVE_SNAPSHOT_DATA_ROOT、RETRIEVE_TEST_TRAJ_ID、RETRIEVE_TEST_SNAPSHOT_ID"
    )
session = SnapshotSession(data_root=DATA_ROOT)
loaded = session.load(traj_id=TRAJ_ID, snapshot_id=SNAPSHOT_ID)
print(f"  FS files: {loaded.env.fs.list_files()}")
print(f"  Vec: {loaded.env.vec.get_stats()}")
print(f"  retrieve_task: {type(loaded.env.retrieve_task).__name__}")
session.release(loaded)
session.cleanup_all()
print("  OK")

# Test 3: sglang available
print("\n=== Test 3: sglang ===")
import sglang
print(f"  sglang version: {sglang.__version__}")

print("\n✅ 集群容器内验证全部通过!")
