"""context_task — Context Task 任务模块。

提供 BaseContextTask 基类、IngestContextTask、ConsolidateContextTask、RetrieveContextTask 及相关统计数据结构。
配置项统一通过全局 config/config.yaml 管理，可通过 config 模块访问。
"""

from .base_context_task import (
    BaseContextTask,
    LLMCallStat,
    TaskStats,
    ToolCallStat,
)
from .consolidate_context_task import (
    ConsolidateContextTask,
    ConsolidateContextTaskStats,
)
from .generate_code_context_code_t2_task import (
    GenerateCodeContextCodeT2Task,
    GenerateCodeContextCodeT2TaskStats,
)
from .generate_code_context_multi_code_t1_task import (
    GenerateCodeContextMultiCodeT1Task,
    GenerateCodeContextMultiCodeT1TaskStats,
)
from .generate_code_context_multi_code_t2_task import (
    GenerateCodeContextMultiCodeT2Task,
    GenerateCodeContextMultiCodeT2TaskStats,
)
from .ingest_context_task import (
    IngestBatchStats,
    IngestContextTask,
    IngestContextTaskStats,
)
from .ingest_context_code_task import (
    IngestContextCodeTask,
    IngestContextCodeTaskStats,
)
from .retrieve_context_task import (
    RetrieveContextTask,
    RetrieveContextTaskStats,
)
from .retrieve_context_code_task import (
    RetrieveContextCodeTask,
    RetrieveContextCodeTaskStats,
)
from .retrieve_context_multi_code_task import (
    RetrieveContextMultiCodeTask,
    RetrieveContextMultiCodeTaskStats,
)
from .context_task_factory import (
    get_consolidate_context_task,
    get_retrieve_context_task,
    get_ingest_context_task,
    get_generate_code_context_task,
)

__all__ = [
    "BaseContextTask",
    "ConsolidateContextTask",
    "ConsolidateContextTaskStats",
    "GenerateCodeContextCodeT2Task",
    "GenerateCodeContextCodeT2TaskStats",
    "GenerateCodeContextMultiCodeT1Task",
    "GenerateCodeContextMultiCodeT1TaskStats",
    "GenerateCodeContextMultiCodeT2Task",
    "GenerateCodeContextMultiCodeT2TaskStats",
    "IngestBatchStats",
    "IngestContextTask",
    "IngestContextTaskStats",
    "IngestContextCodeTask",
    "IngestContextCodeTaskStats",
    "LLMCallStat",
    "RetrieveContextTask",
    "RetrieveContextTaskStats",
    "RetrieveContextCodeTask",
    "RetrieveContextCodeTaskStats",
    "RetrieveContextMultiCodeTask",
    "RetrieveContextMultiCodeTaskStats",
    "TaskStats",
    "ToolCallStat",
    "get_consolidate_context_task",
    "get_retrieve_context_task",
    "get_ingest_context_task",
    "get_generate_code_context_task",
]
