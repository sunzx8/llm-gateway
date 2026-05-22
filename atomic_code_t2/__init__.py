"""Atomic_Code_T2 方案根模块。

该方案与 T2 / Code_T2 隔离开发，本目录下包含本方案专属的：
- 三个 ContextTask：ingest / retrieve / consolidate
- 工具：tool/
- 提示词：prompt/

外部仅复用：gateway、config、storage、interceptor、logger、eval 等通用框架。
"""

from .ingest_task import IngestContextAtomicCodeT2Task
from .retrieve_task import RetrieveContextAtomicCodeT2Task
from .consolidate_task import ConsolidateContextAtomicCodeT2Task

__all__ = [
    "IngestContextAtomicCodeT2Task",
    "RetrieveContextAtomicCodeT2Task",
    "ConsolidateContextAtomicCodeT2Task",
]
