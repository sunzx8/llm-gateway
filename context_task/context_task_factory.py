"""ContextTask 工厂模块。

根据全局配置中的 context_task_mode 返回对应的 Task 实例。

映射关系：
- T2:             ConsolidateContextTask / RetrieveContextTask / IngestContextTask
- Code_T2:        ConsolidateContextTask / RetrieveContextCodeTask / IngestContextCodeTask
- Multi_Code_T1:  ConsolidateContextTask / RetrieveContextMultiCodeTask / IngestContextTask
- Multi_Code_T2:  ConsolidateContextTask / RetrieveContextMultiCodeTask / IngestContextMultiCodeTask
- Atomic_Code_T2: ConsolidateContextTask / RetrieveContextAtomicCodeT2Task / IngestContextAtomicCodeT2Task
                  （演进未实现，由 consolidate_trigger_threshold 极大值禁用触发）

代码生成任务映射（独立于演进任务触发）：
- Code_T2:        GenerateCodeContextCodeT2Task
- Multi_Code_T1:  GenerateCodeContextMultiCodeT1Task
- Multi_Code_T2:  GenerateCodeContextMultiCodeT2Task
- Atomic_Code_T2: 不支持（plan 内置，未引入代码生成路径）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.loader import get_config
from config.models import ContextTaskMode

if TYPE_CHECKING:
    from context_task.base_context_task import BaseContextTask
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase, VectorStoreBase

import logger.logger as logger


def _get_context_task_mode() -> ContextTaskMode:
    """从全局配置中获取当前的 ContextTask 运行模式。"""
    config = get_config()
    return config.memory_config.context_task.context_task_mode


def get_consolidate_context_task(
    llm: "LLMInterface",
    fs_store: "FileSystemStore",
    vec_store: "VectorStoreBase",
    graph_store: "GraphStoreBase",
    **kwargs,
) -> "BaseContextTask":
    """根据全局配置返回对应的 Consolidate Context Task 实例。

    所有模式均返回基础的 ConsolidateContextTask（纯演进）。
    代码生成已拆分至独立的 GenerateCode*Task，通过 get_generate_code_context_task 获取。

    Args:
        llm: LLM 接口实例。
        fs_store: 文件系统存储后端。
        vec_store: 向量数据库存储后端。
        graph_store: 图数据库存储后端。
        **kwargs: 传递给具体 Task 构造函数的额外参数。

    Returns:
        ConsolidateContextTask 实例。
    """
    from context_task.consolidate_context_task import ConsolidateContextTask
    mode = _get_context_task_mode()

    if mode == ContextTaskMode.Atomic_Code_T2:
        from atomic_code_t2 import ConsolidateContextAtomicCodeT2Task
        logger.info("ContextTaskFactory: 创建 ConsolidateContextAtomicCodeT2Task (mode=Atomic_Code_T2)")
        return ConsolidateContextAtomicCodeT2Task(llm, fs_store, vec_store, graph_store, **kwargs)

    logger.info("ContextTaskFactory: 创建 ConsolidateContextTask (mode=%s)", mode.value)
    return ConsolidateContextTask(llm, fs_store, vec_store, graph_store, **kwargs)


def get_retrieve_context_task(
    llm: "LLMInterface",
    fs_store: "FileSystemStore",
    vec_store: "VectorStoreBase",
    graph_store: "GraphStoreBase",
    **kwargs,
) -> "BaseContextTask":
    """根据全局配置返回对应的 Retrieve Context Task 实例。

    映射关系：
    - T2:           RetrieveContextTask
    - Code_T2:      RetrieveContextCodeTask
    - Multi_Code_T1: RetrieveContextMultiCodeTask
    - Multi_Code_T2: RetrieveContextMultiCodeTask

    Args:
        llm: LLM 接口实例。
        fs_store: 文件系统存储后端。
        vec_store: 向量数据库存储后端。
        graph_store: 图数据库存储后端。
        **kwargs: 传递给具体 Task 构造函数的额外参数。

    Returns:
        对应模式的 Retrieve Context Task 实例。
    """
    mode = _get_context_task_mode()

    if mode == ContextTaskMode.T2:
        from context_task.retrieve_context_task import RetrieveContextTask
        logger.info("ContextTaskFactory: 创建 RetrieveContextTask (mode=T2)")
        return RetrieveContextTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Code_T2:
        from context_task.retrieve_context_code_task import RetrieveContextCodeTask
        logger.info("ContextTaskFactory: 创建 RetrieveContextCodeTask (mode=Code_T2)")
        return RetrieveContextCodeTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T1:
        from context_task.retrieve_context_multi_code_task import RetrieveContextMultiCodeTask
        logger.info("ContextTaskFactory: 创建 RetrieveContextMultiCodeTask (mode=Multi_Code_T1)")
        return RetrieveContextMultiCodeTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T2:
        from context_task.retrieve_context_multi_code_task import RetrieveContextMultiCodeTask
        logger.info("ContextTaskFactory: 创建 RetrieveContextMultiCodeTask (mode=Multi_Code_T2)")
        return RetrieveContextMultiCodeTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Atomic_Code_T2:
        from atomic_code_t2 import RetrieveContextAtomicCodeT2Task
        logger.info("ContextTaskFactory: 创建 RetrieveContextAtomicCodeT2Task (mode=Atomic_Code_T2)")
        return RetrieveContextAtomicCodeT2Task(llm, fs_store, vec_store, graph_store, **kwargs)

    else:
        raise ValueError(f"未知的 ContextTaskMode: {mode}")


def get_ingest_context_task(
    llm: "LLMInterface",
    fs_store: "FileSystemStore",
    vec_store: "VectorStoreBase",
    graph_store: "GraphStoreBase",
    **kwargs,
) -> "BaseContextTask":
    """根据全局配置返回对应的 Ingest Context Task 实例。

    映射关系：
    - T2:           IngestContextTask
    - Code_T2:      IngestContextCodeTask
    - Multi_Code_T1: IngestContextTask（复用 T2 的摄入任务）
    - Multi_Code_T2: IngestContextCodeTask（复用 Code_T2 的摄入任务）

    Args:
        llm: LLM 接口实例。
        fs_store: 文件系统存储后端。
        vec_store: 向量数据库存储后端。
        graph_store: 图数据库存储后端。
        **kwargs: 传递给具体 Task 构造函数的额外参数。

    Returns:
        对应模式的 Ingest Context Task 实例。
    """
    mode = _get_context_task_mode()

    if mode == ContextTaskMode.T2:
        from context_task.ingest_context_task import IngestContextTask
        logger.info("ContextTaskFactory: 创建 IngestContextTask (mode=T2)")
        return IngestContextTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Code_T2:
        from context_task.ingest_context_code_task import IngestContextCodeTask
        logger.info("ContextTaskFactory: 创建 IngestContextCodeTask (mode=Code_T2)")
        return IngestContextCodeTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T1:
        from context_task.ingest_context_task import IngestContextTask
        logger.info("ContextTaskFactory: 创建 IngestContextTask (mode=Multi_Code_T1)")
        return IngestContextTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T2:
        from context_task.ingest_context_multi_code_task import IngestContextMultiCodeTask
        logger.info("ContextTaskFactory: 创建 IngestContextMultiCodeTask (mode=Multi_Code_T2)")
        return IngestContextMultiCodeTask(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Atomic_Code_T2:
        from atomic_code_t2 import IngestContextAtomicCodeT2Task
        logger.info("ContextTaskFactory: 创建 IngestContextAtomicCodeT2Task (mode=Atomic_Code_T2)")
        return IngestContextAtomicCodeT2Task(llm, fs_store, vec_store, graph_store, **kwargs)

    else:
        raise ValueError(f"未知的 ContextTaskMode: {mode}")


def get_generate_code_context_task(
    llm: "LLMInterface",
    fs_store: "FileSystemStore",
    vec_store: "VectorStoreBase",
    graph_store: "GraphStoreBase",
    **kwargs,
) -> "BaseContextTask":
    """根据全局配置返回对应的 Generate Code Context Task 实例。

    纯代码生成任务（不含演进），与 Consolidate Task 的触发逻辑保持一致，
    但独立触发，完全依赖上层调度。

    映射关系：
    - T2:            不支持（抛 ValueError）
    - Code_T2:       GenerateCodeContextCodeT2Task
    - Multi_Code_T1: GenerateCodeContextMultiCodeT1Task
    - Multi_Code_T2: GenerateCodeContextMultiCodeT2Task
    - Atomic_Code_T2: 当前阶段不支持代码生成（plan 是内置的，未引入演进），抛 ValueError

    Args:
        llm: LLM 接口实例。
        fs_store: 文件系统存储后端。
        vec_store: 向量数据库存储后端。
        graph_store: 图数据库存储后端。
        **kwargs: 传递给具体 Task 构造函数的额外参数。

    Returns:
        对应模式的 Generate Code Context Task 实例。

    Raises:
        ValueError: 当 mode 为 T2 / Atomic_Code_T2 时（不支持代码生成）。
    """
    mode = _get_context_task_mode()

    if mode == ContextTaskMode.T2:
        raise ValueError(
            "ContextTaskMode.T2 不支持代码生成任务，"
            "请使用 Code_T2 / Multi_Code_T1 / Multi_Code_T2 模式"
        )

    elif mode == ContextTaskMode.Code_T2:
        from context_task.generate_code_context_code_t2_task import GenerateCodeContextCodeT2Task
        logger.info("ContextTaskFactory: 创建 GenerateCodeContextCodeT2Task (mode=Code_T2)")
        return GenerateCodeContextCodeT2Task(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T1:
        from context_task.generate_code_context_multi_code_t1_task import GenerateCodeContextMultiCodeT1Task
        logger.info("ContextTaskFactory: 创建 GenerateCodeContextMultiCodeT1Task (mode=Multi_Code_T1)")
        return GenerateCodeContextMultiCodeT1Task(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Multi_Code_T2:
        from context_task.generate_code_context_multi_code_t2_task import GenerateCodeContextMultiCodeT2Task
        logger.info("ContextTaskFactory: 创建 GenerateCodeContextMultiCodeT2Task (mode=Multi_Code_T2)")
        return GenerateCodeContextMultiCodeT2Task(llm, fs_store, vec_store, graph_store, **kwargs)

    elif mode == ContextTaskMode.Atomic_Code_T2:
        raise ValueError(
            "ContextTaskMode.Atomic_Code_T2 当前阶段不支持代码生成任务（plan 是内置的，"
            "未引入演进路径）。如需通过该入口被调用，请检查上层调度是否对该模式做了豁免。"
        )

    else:
        raise ValueError(f"未知的 ContextTaskMode: {mode}")
