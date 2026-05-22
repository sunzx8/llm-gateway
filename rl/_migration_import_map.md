# RL 管线迁移 import 替换清单

来源工作区: `/data/home/trevzhang/projects/memory-ai-agent-workspace`
目标工作区: `/data/home/trevzhang/projects/llm_gateway`

约定:
- `from rl_env...` → `from llm_gateway.rl.rl_env...`
- `from rl_env import X` → `from llm_gateway.rl.rl_env import X`
- `from slime_train...` → `from llm_gateway.rl.slime_train...`
- `from agent_memory.tasks.{ingest_t3,consolidate_t3,retrieve_t3,retrieve_prompt} import ...` → `from llm_gateway.rl.slime_train._t3_assets import ...`
- `from agent_memory.core.models import Event, EventType, TaskResult, ToolCall` → `from llm_gateway.rl.rl_env._models import Event, EventType, TaskResult, ToolCall`
- `from agent_memory.core.stores import FileSystemStore, GraphStore, VectorStore` → 拆三行：
  - `from llm_gateway.storage.file_system_store import FileSystemStore`
  - `from llm_gateway.storage.vector_stores import VectorStore`
  - `from llm_gateway.storage.graph_stores import GraphStore`
  注：`llm_gateway/storage/__init__.py` 当前为空文件，必须按子模块路径直 import
- `from agent_memory.core.stores_factory import make_stores` → `from llm_gateway.storage.stores_factory import make_stores`，同时在 `MemoryEnv.reset()` 内部把老签名 `(base_dir, embedder, namespace, backend, enable_git)` 翻译成新签名 `(StorageConfig, user_id, embedder)`（构造一个临时 `StorageConfig`）
- `from agent_memory.tasks.{ingest,consolidate,retrieve}_t{2,3} import {Ingest,Consolidate,Retrieve}T{2,3}Task` → 全部由 `llm_gateway.rl.rl_env.task_factory.build_dispatcher(task_version, ...)` 间接绑定到 llm_gateway.{atomic_code_t2, context_task} 中的对应实现，env.py 内不再直接 import 任何 task 类

---

## 表 A. `src/rl_env/*.py` 全量 rl_env 自引用 + agent_memory 引用

| 文件 | 行号 | 旧 import | 替换为 |
|---|---|---|---|
| src/rl_env/__init__.py | 27 | `from rl_env.env import MemoryEnv, StepResult` | `from llm_gateway.rl.rl_env.env import MemoryEnv, StepResult` |
| src/rl_env/__init__.py | 28 | `from rl_env.git_backend import GitFsSnapshotBackend` | `from llm_gateway.rl.rl_env.git_backend import GitFsSnapshotBackend` |
| src/rl_env/__init__.py | 29 | `from rl_env.scorer import BaseScorer, ConsumeScorer, EvolveScorer, IngestScorer` | `from llm_gateway.rl.rl_env.scorer import BaseScorer, ConsumeScorer, EvolveScorer, IngestScorer` |
| src/rl_env/__init__.py | 30 | `from rl_env.serialize import CASBlobStore, EncodedSnapshot` | `from llm_gateway.rl.rl_env.serialize import CASBlobStore, EncodedSnapshot` |
| src/rl_env/__init__.py | 31 | `from rl_env.snapshot_session import LoadedEnv, SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession` |
| src/rl_env/__init__.py | 32-40 | `from rl_env.snapshot import (EncodedSnapshotBackend, FilesystemSnapshotBackend, InMemorySnapshotBackend, Snapshot, SnapshotBackend, dump_snapshot, load_snapshot)` | 同名符号，包名改为 `llm_gateway.rl.rl_env.snapshot` |
| src/rl_env/env.py | 43 | `from agent_memory.core.models import Event, EventType, TaskResult, ToolCall` | `from llm_gateway.rl.rl_env._models import Event, EventType, TaskResult, ToolCall` |
| src/rl_env/env.py | 44 | `from agent_memory.core.stores_factory import make_stores` | 删除该顶层 import；在 `MemoryEnv.reset()` 内部按需 `from llm_gateway.storage.stores_factory import make_stores` 并构造 `StorageConfig` |
| src/rl_env/env.py | 45 | `from agent_memory.tasks.consolidate_t2 import ConsolidateT2Task` | 删除；改由 `task_factory.build_dispatcher` 内按 `task_version` 绑定 |
| src/rl_env/env.py | 46 | `from agent_memory.tasks.consolidate_t3 import ConsolidateT3Task` | 同上删除 |
| src/rl_env/env.py | 47 | `from agent_memory.tasks.ingest_t2 import IngestT2Task` | 同上删除 |
| src/rl_env/env.py | 48 | `from agent_memory.tasks.ingest_t3 import IngestT3Task` | 同上删除 |
| src/rl_env/env.py | 49 | `from agent_memory.tasks.retrieve_t2 import RetrieveT2Task` | 同上删除 |
| src/rl_env/env.py | 50 | `from agent_memory.tasks.retrieve_t3 import RetrieveT3Task` | 同上删除 |
| src/rl_env/env.py | 52 | `from rl_env.scorer import BaseScorer, ScoreResult` | `from llm_gateway.rl.rl_env.scorer import BaseScorer, ScoreResult` |
| src/rl_env/env.py | 53-57 | `from rl_env.snapshot import (InMemorySnapshotBackend, Snapshot, SnapshotBackend)` | `from llm_gateway.rl.rl_env.snapshot import (InMemorySnapshotBackend, Snapshot, SnapshotBackend)` |
| src/rl_env/env.py | 60-61 (TYPE_CHECKING) | `from agent_memory.core.llm_interface import EmbeddingInterface, LLMInterface` / `from agent_memory.core.stores import FileSystemStore, GraphStore, VectorStore` | `from llm_gateway.utils.memory_llm_interface import EmbeddingInterface, LLMInterface` / 三行拆 storage 子模块 |
| src/rl_env/git_backend.py | 60 | `from rl_env.serialize import EncodedSnapshot, dump_encoded, load_encoded` | `from llm_gateway.rl.rl_env.serialize import EncodedSnapshot, dump_encoded, load_encoded` |
| src/rl_env/git_backend.py | 61-65 | `from rl_env.snapshot import (InMemorySnapshotBackend, Snapshot, SnapshotBackend)` | 包路径改为 `llm_gateway.rl.rl_env.snapshot` |
| src/rl_env/snapshot.py | 45-52 | `from rl_env.serialize import (...)` | 包路径改为 `llm_gateway.rl.rl_env.serialize` |
| src/rl_env/snapshot_session.py | 26 | `from agent_memory.core.stores import FileSystemStore, GraphStore, VectorStore` | 三行拆 `llm_gateway.storage.{file_system_store,vector_stores,graph_stores}` |
| src/rl_env/snapshot_session.py | 27 | `from rl_env.env import MemoryEnv` | `from llm_gateway.rl.rl_env.env import MemoryEnv` |
| src/rl_env/snapshot_session.py | 28 | `from rl_env.serialize import decode_snapshot, load_encoded` | `from llm_gateway.rl.rl_env.serialize import decode_snapshot, load_encoded` |
| src/rl_env/snapshot_session.py | 29 | `from rl_env.snapshot import InMemorySnapshotBackend` | `from llm_gateway.rl.rl_env.snapshot import InMemorySnapshotBackend` |

`src/rl_env/` 内总命中点: **rl_env 自引用 14 处 + agent_memory 9 处 = 23 处**

---

## 表 B. `slime_train/**/*.py` 内 agent_memory 引用 (3 处)

| 文件 | 行号 | 旧 import | 替换为 |
|---|---|---|---|
| slime_train/ingest/convert_to_slime_format.py | 27 | `from agent_memory.tasks.ingest_t3 import INGEST_T3_SYSTEM_PROMPT, INGEST_T3_TOOLS` | `from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_SYSTEM_PROMPT, INGEST_T3_TOOLS` |
| slime_train/consolidate/convert_to_slime_format.py | 27 | `from agent_memory.tasks.consolidate_t3 import CONSOLIDATE_T3_SYSTEM_PROMPT, CONSOLIDATE_T3_TOOLS` | `from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_SYSTEM_PROMPT, CONSOLIDATE_T3_TOOLS` |
| slime_train/retrieve/convert_to_slime_format.py | 57-58 | `from agent_memory.tasks.retrieve_prompt import build_fs_structure_from_files` / `from agent_memory.tasks.retrieve_t3 import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE` | `from llm_gateway.rl.slime_train._t3_assets import build_fs_structure_from_files, QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE` |
| slime_train/memory_rl/tool_schemas.py | 9 | `from agent_memory.tasks.ingest_t3 import INGEST_T3_TOOLS` (函数内 lazy import) | `from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_TOOLS` |
| slime_train/memory_rl/tool_schemas.py | 13 | `from agent_memory.tasks.consolidate_t3 import CONSOLIDATE_T3_TOOLS` | `from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_TOOLS` |

agent_memory 命中点: **5 处**

---

## 表 C. `slime_train/**/*.py` 内 rl_env / slime_train 自引用

| 文件 | 行号 | 旧 import | 替换为 |
|---|---|---|---|
| slime_train/tasks/ingest_reward/reward.py | 12 | `from slime_train.memory_rl.paths import ensure_workspace_paths` | `from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths` |
| slime_train/tasks/ingest_reward/reward.py | 16 | `from rl_env.snapshot_session import SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession` |
| slime_train/tasks/ingest_reward/reward.py | 17-18 | `from slime_train.memory_rl.{probes,response_parser} import ...` | 包路径改为 `llm_gateway.rl.slime_train.memory_rl.*` |
| slime_train/tasks/consolidate_reward/reward.py | 13 | `from slime_train.memory_rl.paths import ...` | `llm_gateway.rl.slime_train.memory_rl.paths` |
| slime_train/tasks/consolidate_reward/reward.py | 17 | `from rl_env.snapshot_session import SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession` |
| slime_train/tasks/consolidate_reward/reward.py | 18-26 | `from slime_train.memory_rl.{probes,response_parser} import ...` | 同 |
| slime_train/tasks/retrieve_reward/snapshot_cache.py | 9 | `from rl_env import SnapshotSession` | `from llm_gateway.rl.rl_env import SnapshotSession` |
| slime_train/tasks/retrieve_reward/reward.py | 14-25 | `_find_workspace_src()` + `sys.path.insert(0, _WORKSPACE_SRC)` | 整体删除并替换为 `from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths` + 调用 |
| slime_train/retrieve/snapshot_loader.py | 26 | `from rl_env.snapshot_session import LoadedEnv, SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession` |
| slime_train/retrieve/tasks/retrieve_reward/snapshot_cache.py | 9 | `from rl_env import SnapshotSession` | `from llm_gateway.rl.rl_env import SnapshotSession` (该副本是 retrieve/ 下的镜像；保留即可) |
| slime_train/ingest/convert_to_slime_format.py | 23 | `from slime_train.memory_rl.paths import ensure_workspace_paths` | `llm_gateway.rl.slime_train.memory_rl.paths` |
| slime_train/ingest/convert_to_slime_format.py | 28 | `from rl_env.snapshot_session import SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession` |
| slime_train/ingest/convert_to_slime_format.py | 29 | `from slime_train.memory_rl.probes import select_task_probes` | `llm_gateway.rl.slime_train.memory_rl.probes` |
| slime_train/consolidate/convert_to_slime_format.py | 23,28,29 | 同 ingest 模式 | 同 |
| slime_train/retrieve/convert_to_slime_format.py | (memory_rl.paths / rl_env / probes) | 同上模式 | 同 |
| slime_train/retrieve/scripts/test_cluster_env.py | 16 | `from slime_train.tasks.retrieve_reward.reward import ...` | `from llm_gateway.rl.slime_train.tasks.retrieve_reward.reward import ...` |
| slime_train/memory_rl/smoke_vllm_rollout_reward.py | 29-35 | `from slime_train.memory_rl.{agentic_rollout,paths,tool_executors,tool_schemas} import ...` | 全部改为 `llm_gateway.rl.slime_train.memory_rl.*` |
| slime_train/memory_rl/tool_executors.py | 7 | `from rl_env.snapshot_session import LoadedEnv, SnapshotSession` | `from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession` |
| slime_train/memory_rl/custom_generate.py | 11-16 | `from slime_train.memory_rl.{paths,response_parser,tool_executors} import ...` | `llm_gateway.rl.slime_train.memory_rl.*` |
| slime_train/memory_rl/agentic_rollout.py | 7 | `from slime_train.memory_rl.response_parser import parse_tool_calls_response` | 同 |

rl_env / slime_train 自引用命中点: **共 ~19 处**(批量 sed 即可)

---

## 表 D. PYTHONPATH / 寻根逻辑

| 文件 | 现状 | 迁移后改造建议 |
|---|---|---|
| slime_train/memory_rl/paths.py | `find_workspace_root` 寻找包含 `src/agent_memory` 的目录 | 改为寻找包含 `gateway/` + `storage/` 的 `llm_gateway` 根；不再追加 `src` 与 `slime_train` 到 sys.path（因为 import 已使用绝对包名），仅在被脚本直接 `python -m` 调用时把 llm_gateway 父目录加入 sys.path |
| slime_train/tasks/retrieve_reward/reward.py L14-25 | `_find_workspace_src` 寻 `src/agent_memory` | 整个函数和顶层 `sys.path.insert` 删除；改为 `from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths` + 函数内调用 |

## 表 E. shell 脚本

| 文件 | 现状 | 迁移后改造建议 |
|---|---|---|
| slime_train/ingest/scripts/train_ingest_grpo.sh L18 | `CUSTOM_RM_PATH=slime_train.tasks.ingest_reward.reward.reward_func` | `llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func` |
| 同 L19 | `CUSTOM_GENERATE_FUNCTION_PATH=slime_train.memory_rl.custom_generate.custom_generate` | `llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate` |
| 同 L41 | `PYTHONPATH=...:${WORKSPACE_ROOT}:${WORKSPACE_SRC}:${SLIME_TRAIN_ROOT}:${PROJECT_ROOT}:` | `PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${PYTHONPATH:-}` 其中 `LLM_GATEWAY_PARENT` 是 llm_gateway 的父目录(以使 `import llm_gateway` 成功)；同时删除 `WORKSPACE_SRC/SLIME_TRAIN_ROOT/PROJECT_ROOT` 等老变量 |
| slime_train/consolidate/scripts/train_consolidate_grpo.sh | 同 ingest 结构 | 同 |
| slime_train/retrieve/scripts/train_retrieve_grpo.sh | 同 | 同 |
| slime_train/retrieve/scripts/test_train_mini.sh | 调用上面 train 脚本 | 检查内部 PYTHONPATH 是否一致 |

## 表 F. agent_memory vendor 符号清单 (合并去重)

| 符号 | 来源模块 | 类型 | 引用位置 |
|---|---|---|---|
| `INGEST_T3_TOOLS` | agent_memory.tasks.ingest_t3 | list[dict] OpenAI tool schemas | slime_train/memory_rl/tool_schemas.py L9; slime_train/ingest/convert_to_slime_format.py L27 |
| `INGEST_T3_SYSTEM_PROMPT` | agent_memory.tasks.ingest_t3 | str | slime_train/ingest/convert_to_slime_format.py L27 |
| `CONSOLIDATE_T3_TOOLS` | agent_memory.tasks.consolidate_t3 | list[dict] | slime_train/memory_rl/tool_schemas.py L13; slime_train/consolidate/convert_to_slime_format.py L27 |
| `CONSOLIDATE_T3_SYSTEM_PROMPT` | agent_memory.tasks.consolidate_t3 | str | slime_train/consolidate/convert_to_slime_format.py L27 |
| `QUERY_REWRITE_SYSTEM_PROMPT` | agent_memory.tasks.retrieve_t3 | str | slime_train/retrieve/convert_to_slime_format.py L58 |
| `QUERY_REWRITE_USER_TEMPLATE` | agent_memory.tasks.retrieve_t3 | str | 同 |
| `build_fs_structure_from_files` | agent_memory.tasks.retrieve_prompt | function(list[str]) -> str | slime_train/retrieve/convert_to_slime_format.py L57 |

vendor 去重符号总数: **7**

---

## 总结

- rl_env 内总命中点数: **23** (rl_env 自引用 14 + agent_memory 9)
- slime_train 内 agent_memory 命中点数: **5**
- slime_train 内 rl_env / slime_train 自引用命中点数: **~19**
- shell 脚本命中点数: **3 个脚本 × ~3 行/脚本**
- vendor 符号去重后总数: **7**
