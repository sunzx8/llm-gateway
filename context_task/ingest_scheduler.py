"""
定时记忆摄入调度器 — 在独立进程中定时扫描所有 session 并触发记忆摄入。

架构：
    - 调度进程：运行独立的 asyncio event loop，定时扫描 SessionManager 中所有有 pending messages 的 session
    - 并发协程：使用 asyncio.gather + Semaphore 控制并发度，并发处理不同 session 的记忆摄入任务
    - 安全回收：服务关闭时通过 shutdown event 通知调度进程退出，等待当前正在执行的任务完成

为什么用进程：
    - 记忆摄入任务涉及大量 LLM API 调用和计算，独立进程可避免 GIL 限制
    - 进程隔离性更好，调度器崩溃不会影响主服务进程
    - 通过 multiprocessing.Event 实现跨进程的安全关闭信号
"""

from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import multiprocessing.managers
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from context_task import get_consolidate_context_task, get_ingest_context_task
from logger.logger import logger

# ============================================================
# 全局变量：记录各 user_id 的 ingest 累积次数
# 用于判断累积超过阈值后触发 consolidate
# 使用 multiprocessing.Manager 实现跨进程安全共享
# ============================================================
_manager: Optional[multiprocessing.managers.SyncManager] = None
_ingest_counter: Optional[Any] = None  # Manager.dict()
_ingest_counter_lock: Optional[Any] = None  # Manager.Lock()

# ============================================================
# 全局变量：记录各 user_id 的 retrieve 累积次数
# 用于判断累积超过阈值后异步触发 consolidate
# retrieve 在主进程执行，无需跨进程共享，使用 threading.Lock 即可
# ============================================================
_retrieve_counter: dict[str, int] = {}
_retrieve_counter_lock: threading.Lock = threading.Lock()


# ============================================================
# 跨进程文件锁：防止同一 user_id 的 consolidate 任务并发执行
# 使用 fcntl.flock 实现，主进程和调度子进程均可感知
# ============================================================

def _get_consolidate_lock_dir(memory_config_dict: dict[str, Any] | None = None) -> str:
    """获取 consolidate 文件锁目录。

    使用与存储根目录同级的 .locks 目录，确保跨进程可见。

    Args:
        memory_config_dict: 序列化的 MemoryConfig 字典，用于提取 root_dir。
    """
    root_dir = "/tmp/llm_gateway"
    if memory_config_dict:
        try:
            root_dir = memory_config_dict["storage"]["memory_fs"]["root_dir"]
        except (KeyError, TypeError):
            pass
    lock_dir = os.path.join(root_dir, ".locks", "consolidate")
    os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def _safe_lock_filename(user_id: str) -> str:
    """将 user_id 转换为安全的文件名。"""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", user_id)


def _try_acquire_consolidate_lock(
    user_id: str, memory_config_dict: dict[str, Any] | None = None,
) -> int | None:
    """尝试获取指定 user_id 的 consolidate 文件锁（非阻塞）。

    使用 fcntl.flock 的 LOCK_EX | LOCK_NB 模式：
    - 获取成功：返回文件描述符（调用方需在任务完成后调用 _release_consolidate_lock 释放）
    - 获取失败（已有其他进程/协程持有）：返回 None

    Args:
        user_id: 用户标识。
        memory_config_dict: 序列化的 MemoryConfig 字典，用于确定锁文件目录。

    Returns:
        文件描述符（成功）或 None（已被占用）。
    """
    lock_dir = _get_consolidate_lock_dir(memory_config_dict)
    lock_file = os.path.join(lock_dir, f"{_safe_lock_filename(user_id)}.lock")
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, IOError):
        # LOCK_NB 模式下获取失败会抛出 BlockingIOError (OSError 子类)
        try:
            os.close(fd)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return None


def _release_consolidate_lock(fd: int) -> None:
    """释放 consolidate 文件锁。

    Args:
        fd: 由 _try_acquire_consolidate_lock 返回的文件描述符。
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def _ensure_shared_state() -> None:
    """确保跨进程共享状态已初始化。

    使用 multiprocessing.Manager 创建共享的 dict 和 Lock，
    保证主进程和调度子进程之间的计数器状态一致。
    """
    global _manager, _ingest_counter, _ingest_counter_lock
    if _manager is None:
        _manager = multiprocessing.Manager()
        _ingest_counter = _manager.dict()
        _ingest_counter_lock = _manager.Lock()


def shutdown_shared_state() -> None:
    """关闭共享状态管理器，释放资源。

    应在服务关闭时调用。
    """
    global _manager, _ingest_counter, _ingest_counter_lock
    if _manager is not None:
        try:
            _manager.shutdown()
        except Exception:
            pass
        _manager = None
        _ingest_counter = None
        _ingest_counter_lock = None


def _scheduler_process_entry(
    memory_config_dict: dict[str, Any],
    interval: float,
    max_workers: int,
    shutdown_event: multiprocessing.synchronize.Event,
) -> None:
    """调度进程入口函数（模块级别，可被 pickle）。

    创建独立的 asyncio event loop 并运行定时调度逻辑。

    Args:
        memory_config_dict: 序列化的 MemoryConfig 字典。
        interval: 定时扫描间隔（秒）。
        max_workers: 最大并发 session 处理数。
        shutdown_event: 跨进程的关闭信号。
    """
    from logger.logger import logger as proc_logger

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _async_scheduler_loop(
                memory_config_dict, interval, max_workers, shutdown_event
            )
        )
    except Exception as e:
        if not shutdown_event.is_set():
            proc_logger.error(f"IngestScheduler: 调度循环异常退出: {e}", exc_info=True)
    finally:
        # 清理 event loop 中的剩余任务
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        loop.close()
        proc_logger.info("IngestScheduler: event loop 已关闭")


async def _async_scheduler_loop(
    memory_config_dict: dict[str, Any],
    interval: float,
    max_workers: int,
    shutdown_event: multiprocessing.synchronize.Event,
) -> None:
    """异步调度主循环：定时扫描并触发摄入。"""
    from logger.logger import logger as proc_logger

    proc_logger.info(
        "IngestScheduler: 异步调度循环已启动 (interval=%.1fs)",
        interval,
    )

    while not shutdown_event.is_set():
        try:
            await _run_one_cycle(memory_config_dict, max_workers)
        except Exception as e:
            proc_logger.error(f"IngestScheduler: 调度周期异常: {e}", exc_info=True)

        # 等待间隔时间，期间检查 shutdown_event
        # 使用短间隔轮询以便快速响应 shutdown
        elapsed = 0.0
        while elapsed < interval and not shutdown_event.is_set():
            await asyncio.sleep(min(1.0, interval - elapsed))
            elapsed += 1.0


async def _run_one_cycle(
    memory_config_dict: dict[str, Any],
    max_workers: int,
) -> None:
    """执行一次调度周期：扫描所有 session 并并发触发摄入任务。"""
    from logger.logger import logger as proc_logger
    from storage.session_store import session_manager

    # 收集所有有 pending messages 的 session
    pending_sessions = []
    users = await session_manager.list_users()

    for user_id in users:
        user_sessions = await session_manager.get_user_sessions(user_id)
        for session_id, store in user_sessions.items():
            messages, max_index = await store.get_pending_ingest_messages()
            if messages:
                pending_sessions.append({
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": messages,
                    "max_index": max_index,
                })

    if not pending_sessions:
        return

    proc_logger.info(
        "IngestScheduler: 本轮发现 %d 个待摄入 session",
        len(pending_sessions),
    )

    # 使用 Semaphore 控制并发度
    semaphore = asyncio.Semaphore(max_workers)

    async def _process_session(item: dict[str, Any]) -> None:
        async with semaphore:
            await _ingest_session(item, memory_config_dict)

    # 并发处理所有 pending session
    await asyncio.gather(
        *[_process_session(item) for item in pending_sessions],
        return_exceptions=True,
    )


def _split_messages_for_ingest(
    messages: list[dict[str, str]],
    max_messages_per_batch: int,
) -> list[tuple[int, list[dict[str, str]]]]:
    """按服务端逻辑摄入粒度切分消息。

    返回值中的 int 是该 batch 第一条消息相对于本次 messages 的 0-based 偏移，
    用于修正 agent loop 看到的全局 message_offset。
    """
    if not messages:
        return [(0, [])]

    batch_size = max(1, int(max_messages_per_batch))
    return [
        (start, messages[start:start + batch_size])
        for start in range(0, len(messages), batch_size)
    ]


def _is_successful_ingest_result(result: Any) -> bool:
    """判断一次逻辑 batch 是否应计入成功摄入次数。"""
    if not isinstance(result, dict):
        return True
    if result.get("success") is False:
        return False

    status = result.get("status")
    if status is not None and status != "success":
        return False

    return True


async def memory_ingest(
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
    memory_config_dict: dict[str, Any],
    memory_llm: Any,
    embedder: Any,
    message_offset: int = 1,
) -> list[dict[str, Any]]:
    """统一的记忆摄入函数。

    执行 ingest 任务并检查是否需要触发 consolidate。

    Args:
        user_id: 用户标识。
        session_id: 会话标识。
        messages: 待摄入的消息列表。
        memory_config_dict: 序列化的 MemoryConfig 字典。
        memory_llm: 已创建的 LLMInterface 实例。
        embedder: 已创建的 EmbeddingInterface 实例。
        message_offset: 消息编号起始值（用于 batch 切片场景，保证编号全局连续）。

    Returns:
        任务结果列表，包含 ingest 结果和可能的 consolidate 结果。
    """
    from config.models import MemoryConfig
    from storage.stores_factory import make_stores
    from context_task.ingest_context_task import IngestContextTask
    from logger.logger import logger as proc_logger

    memory_config = MemoryConfig(**memory_config_dict)
    task_results: list[dict[str, Any]] = []

    fs_store, vec_store, graph_store = make_stores(
        memory_config.storage, user_id, embedder,
    )

    ingest_task = get_ingest_context_task(
        memory_llm, fs_store, vec_store, graph_store,
        token_budget=memory_config.context_task.ingest_context_task.token_budget_per_ingest,
        max_messages=memory_config.context_task.ingest_context_task.max_messages_per_ingest,
        max_turns=memory_config.context_task.ingest_context_task.max_turns,
    )
    message_batches = _split_messages_for_ingest(
        messages,
        memory_config.context_task.ingest_context_task.max_messages_per_ingest,
    )

    for batch_index, (batch_start, batch_messages) in enumerate(message_batches, start=1):
        batch_message_offset = message_offset + batch_start
        result = await ingest_task.execute(
            session_id=session_id,
            messages=batch_messages,
            user_id=user_id or "default_user",
            message_offset=batch_message_offset,
        )
        task_results.append({
            "task_type": "ingest",
            "input_params": {
                "user_id": user_id,
                "session_id": session_id,
                "messages": batch_messages,
                "batch_index": batch_index,
                "batch_count": len(message_batches),
                "message_offset": batch_message_offset,
            },
            "result": result,
        })

        proc_logger.info(
            f"memory_ingest: session {session_id} (user={user_id}) "
            f"batch {batch_index}/{len(message_batches)} 摄入完成"
        )

        # 成功摄入一个逻辑 batch 后再累计计数：达到阈值时，演进刚写入的这一批结果
        if batch_messages and _is_successful_ingest_result(result):
            consolidate_result = await check_and_trigger_consolidate(user_id, session_id, memory_config_dict)
            if consolidate_result is not None:
                task_results.append({
                    "task_type": "consolidate",
                    "input_params": {
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                    "result": consolidate_result,
                })
        else:
            proc_logger.warning(
                f"memory_ingest: session {session_id} (user={user_id}) "
                f"batch {batch_index}/{len(message_batches)} 未成功摄入，跳过 consolidate 计数"
            )

    return task_results


async def _ingest_session(item: dict[str, Any], memory_config_dict: dict[str, Any]) -> None:
    """对单个 session 执行记忆摄入（定时调度器调用）。

    Args:
        item: 包含 user_id, session_id, messages, max_index 的字典。
        memory_config_dict: 序列化的 MemoryConfig 字典。
    """
    from config.models import MemoryConfig
    from storage.session_store import session_manager
    from utils.memory_llm_interface import LLMInterface, EmbeddingInterface
    from logger.logger import logger as proc_logger

    user_id = item["user_id"]
    session_id = item["session_id"]
    messages = item["messages"]
    max_index = item["max_index"]

    try:
        memory_config = MemoryConfig(**memory_config_dict)
        memory_llm = LLMInterface(memory_config.memory_llm.model_dump())
        embedder = EmbeddingInterface(memory_config.embedding.model_dump())

        await memory_ingest(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            memory_config_dict=memory_config_dict,
            memory_llm=memory_llm,
            embedder=embedder,
        )

        # 摄入成功后清理已处理的消息
        store = await session_manager.get_session(user_id, session_id)
        if store:
            await store.clear_ingested_messages(max_index)

    except Exception as e:
        proc_logger.error(
            f"IngestScheduler: session {session_id} (user={user_id}) 摄入失败: {e}",
            exc_info=True,
        )


# ============================================================
# Consolidate 相关逻辑
# ============================================================

async def check_and_trigger_consolidate(
    user_id: str, session_id: str, memory_config_dict: dict[str, Any]
) -> dict[str, Any] | None:
    """检查成功摄入累计次数是否达到阈值，达到则触发 consolidate。

    每次 ingest 成功完成后调用，对当前 user_id 的计数 +1，
    达到 consolidate_trigger_threshold 后重置计数器并触发 consolidate 任务。

    Args:
        user_id: 用户标识。
        session_id: 会话标识。
        memory_config_dict: 序列化的 MemoryConfig 字典。

    Returns:
        consolidate 结果字典，若未触发则返回 None。
    """
    from config.models import MemoryConfig
    from logger.logger import logger as proc_logger

    memory_config = MemoryConfig(**memory_config_dict)
    threshold = memory_config.context_task.consolidate_trigger_threshold

    _ensure_shared_state()
    assert _ingest_counter is not None
    assert _ingest_counter_lock is not None

    should_consolidate = False
    with _ingest_counter_lock:
        current = _ingest_counter.get(user_id, 0) + 1
        if current >= threshold:
            _ingest_counter[user_id] = 0  # 重置计数器
            should_consolidate = True
        else:
            _ingest_counter[user_id] = current

    if should_consolidate:
        proc_logger.info(
            f"用户 {user_id} 成功 ingest 累计次数达到阈值 {threshold}，触发 consolidate"
        )
        return await trigger_consolidate(user_id, session_id, memory_config_dict)

    return None


async def trigger_consolidate(
    user_id: str, session_id: str, memory_config_dict: dict[str, Any]
) -> dict[str, Any] | None:
    """触发 consolidate 任务（跨进程防并发）。

    使用文件锁确保同一 user_id 同一时刻只有一个 consolidate 在执行，
    无论从 ingest 路径还是 retrieve 路径触发，也无论在主进程还是调度子进程。

    演进完成后，会自动触发代码生成任务（如果当前模式支持）。

    Args:
        user_id: 用户标识。
        session_id: 会话标识。
        memory_config_dict: 序列化的 MemoryConfig 字典。

    Returns:
        consolidate 结果字典；若被跳过返回 None；失败时返回包含 error 的字典。
    """
    from config.models import MemoryConfig, ContextTaskMode
    from storage.stores_factory import make_stores
    from utils.memory_llm_interface import LLMInterface, EmbeddingInterface
    from context_task.consolidate_context_task import ConsolidateContextTask
    from context_task.context_task_factory import get_generate_code_context_task
    from logger.logger import logger as proc_logger

    # 尝试获取跨进程文件锁（非阻塞）
    lock_fd = _try_acquire_consolidate_lock(user_id, memory_config_dict)
    if lock_fd is None:
        proc_logger.info(
            f"用户 {user_id} consolidate 跳过（已有任务在执行，文件锁被占用）"
        )
        return None

    try:
        memory_config = MemoryConfig(**memory_config_dict)
        memory_llm = LLMInterface(memory_config.memory_llm.model_dump())
        embedder = EmbeddingInterface(memory_config.embedding.model_dump())

        fs_store, vec_store, graph_store = make_stores(
            memory_config.storage, user_id, embedder
        )

        consolidate_task = get_consolidate_context_task(
            memory_llm, fs_store, vec_store, graph_store,
            min_items=memory_config.context_task.consolidate_context_task.min_items_for_evolution,
            max_turns=memory_config.context_task.consolidate_context_task.max_turns,
        )
        result = await consolidate_task.execute(
            user_id=user_id,
            session_id=session_id,
        )
        proc_logger.info(f"用户 {user_id} consolidate 完成: {result}")

        # 演进完成后，触发代码生成任务（如果当前模式支持）。
        # T2 与 Atomic_Code_T2 均不支持 generate_code（详见 context_task_factory.get_generate_code_context_task），
        # 这里直接短路，避免在 try 里抛 ValueError 后只能记 ERROR 日志（噪音）。
        _codegen_unsupported_modes = {ContextTaskMode.T2, ContextTaskMode.Atomic_Code_T2}
        if memory_config.context_task.context_task_mode not in _codegen_unsupported_modes:
            try:
                generate_code_task = get_generate_code_context_task(
                    memory_llm, fs_store, vec_store, graph_store,
                    min_items=memory_config.context_task.consolidate_context_task.min_items_for_evolution,
                    max_turns=memory_config.context_task.consolidate_context_task.max_turns,
                )
                codegen_result = await generate_code_task.execute(
                    user_id=user_id,
                    session_id=session_id,
                )
                proc_logger.info(f"用户 {user_id} generate_code 完成: {codegen_result}")
                # 将代码生成结果合并到返回值
                if result is None:
                    result = {}
                result["codegen_result"] = codegen_result
            except Exception as codegen_err:
                proc_logger.error(
                    f"用户 {user_id} generate_code 失败: {codegen_err}",
                    exc_info=True,
                )
                if result is None:
                    result = {}
                result["codegen_error"] = str(codegen_err)

        return result
    except Exception as e:
        proc_logger.error(f"用户 {user_id} consolidate 失败: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        _release_consolidate_lock(lock_fd)


# ============================================================
# Retrieve 触发 Consolidate 相关逻辑
# ============================================================

def check_and_trigger_consolidate_by_retrieve(
    user_id: str, session_id: str, memory_config_dict: dict[str, Any]
) -> None:
    """检查累积 retrieve 次数是否达到阈值，达到则异步触发 consolidate。

    不阻塞调用方，通过 asyncio.create_task 在后台执行。
    跨进程防并发由 trigger_consolidate 内部的文件锁统一保证。

    Args:
        user_id: 用户标识。
        session_id: 会话标识。
        memory_config_dict: 序列化的 MemoryConfig 字典。
    """
    from config.models import MemoryConfig

    memory_config = MemoryConfig(**memory_config_dict)
    threshold = memory_config.context_task.consolidate_retrieve_trigger_threshold

    should_consolidate = False
    with _retrieve_counter_lock:
        current = _retrieve_counter.get(user_id, 0) + 1
        if current >= threshold:
            _retrieve_counter[user_id] = 0
            should_consolidate = True
        else:
            _retrieve_counter[user_id] = current

    if should_consolidate:
        logger.info(
            f"用户 {user_id} 累积 retrieve 次数达到阈值 {threshold}，异步触发 consolidate"
        )
        asyncio.create_task(
            _async_consolidate_by_retrieve(user_id, session_id, memory_config_dict)
        )


async def _async_consolidate_by_retrieve(
    user_id: str, session_id: str, memory_config_dict: dict[str, Any]
) -> None:
    """异步执行 retrieve 触发的 consolidate，捕获异常避免影响主流程。

    跨进程防并发由 trigger_consolidate 内部的文件锁统一保证，
    若该用户已有 consolidate 在执行，trigger_consolidate 会返回 None 并跳过。
    """
    try:
        result = await trigger_consolidate(user_id, session_id, memory_config_dict)
        if result is not None:
            logger.info(f"用户 {user_id} retrieve 触发的异步 consolidate 完成: {result}")
        else:
            logger.info(f"用户 {user_id} retrieve 触发的异步 consolidate 被跳过（文件锁被占用）")
    except Exception as e:
        logger.error(
            f"用户 {user_id} retrieve 触发的异步 consolidate 失败: {e}",
            exc_info=True,
        )


class IngestScheduler:
    """定时记忆摄入调度器。

    在服务启动时调用 start() 启动独立的调度进程，
    在服务关闭时调用 shutdown() 安全回收进程和正在执行的任务。

    Usage:
        scheduler = IngestScheduler(memory_config_dict)
        scheduler.start()
        ...
        scheduler.shutdown()
    """

    def __init__(
        self,
        memory_config_dict: dict[str, Any],
        interval: float = 30.0,
        max_workers: int = 4,
    ) -> None:
        """初始化调度器。

        Args:
            memory_config_dict: 序列化的 MemoryConfig 字典。
            interval: 定时扫描间隔（秒）。
            max_workers: 最大并发 session 处理数。
        """
        self._memory_config_dict = memory_config_dict
        self._interval = interval
        self._max_workers = max_workers
        self._shutdown_event: multiprocessing.Event = multiprocessing.Event()
        self._process: Optional[multiprocessing.Process] = None

        # 确保跨进程共享状态已初始化
        _ensure_shared_state()

    def start(self) -> None:
        """启动调度进程。"""
        if self._process is not None and self._process.is_alive():
            logger.warning("IngestScheduler: 调度进程已在运行中，跳过重复启动")
            return

        self._shutdown_event.clear()
        self._process = multiprocessing.Process(
            target=_scheduler_process_entry,
            args=(
                self._memory_config_dict,
                self._interval,
                self._max_workers,
                self._shutdown_event,
            ),
            name="ingest-scheduler",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "IngestScheduler: 调度进程已启动 (pid=%d, interval=%.1fs, max_workers=%d)",
            self._process.pid, self._interval, self._max_workers,
        )

    def shutdown(self, timeout: float = 30.0) -> None:
        """安全关闭调度进程。

        通知调度进程退出，等待当前正在执行的摄入任务完成后退出。

        Args:
            timeout: 等待调度进程退出的超时时间（秒）。
        """
        if self._process is None or not self._process.is_alive():
            logger.info("IngestScheduler: 调度进程未运行，无需关闭")
            return

        logger.info("IngestScheduler: 正在通知调度进程退出...")
        self._shutdown_event.set()

        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning(
                "IngestScheduler: 调度进程未在 %.1fs 内退出，强制终止",
                timeout,
            )
            self._process.terminate()
            self._process.join(timeout=5.0)
        else:
            logger.info("IngestScheduler: 调度进程已安全关闭")

        self._process = None

        # 关闭共享状态管理器
        shutdown_shared_state()

    @property
    def is_running(self) -> bool:
        """调度进程是否正在运行。"""
        return self._process is not None and self._process.is_alive()
