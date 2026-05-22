"""代码生成暂存区管理器。

提供原子性的代码文件提交机制：
- 代码生成任务的所有文件写入先写到暂存目录（.codegen_staging/）
- 全部完成后一次性提交（备份旧文件 + 覆盖新文件）
- 任务失败时丢弃暂存区，不影响现有代码

使用方式：
    staging = CodegenStagingArea(fs_store)
    staging.write_file("retrieve_base_memory.py", code)
    staging.write_file("retrieve_memory_v1.py", code_v1)
    ...
    staging.commit()  # 原子性提交
    # 或
    staging.discard()  # 丢弃暂存区
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

import logger.logger as logger

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore

from storage.file_system_store import (
    CODEGEN_DIR,
    CODEGEN_SNAPSHOT_DIR,
    CODEGEN_STAGING_DIR,
)


class CodegenStagingArea:
    """代码生成暂存区，保证代码文件修改的原子性。

    所有写入操作先落到 .codegen_staging/ 目录，commit() 时一次性：
    1. 将 .codegen/ 中即将被覆盖的文件备份到 .codegen/snapshots/
    2. 将暂存区文件移动到 .codegen/
    3. 清理暂存区

    这样在代码生成过程中，消费/摄入任务加载的始终是旧版本的完整代码，
    不会遇到半成品文件。
    """

    def __init__(self, fs_store: "FileSystemStore") -> None:
        self._fs = fs_store
        self._base_path = fs_store.base_path
        self._staging_dir = os.path.join(self._base_path, CODEGEN_STAGING_DIR)
        self._codegen_dir = os.path.join(self._base_path, CODEGEN_DIR)
        self._snapshot_dir = os.path.join(self._base_path, CODEGEN_SNAPSHOT_DIR)
        # 记录暂存区中写入的文件（相对于 CODEGEN_DIR 的路径）
        self._staged_files: list[str] = []
        # 确保暂存目录存在
        os.makedirs(self._staging_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入操作（写到暂存区）
    # ------------------------------------------------------------------

    def write_file(self, codegen_rel_name: str, content: str) -> str:
        """将文件写入暂存区。

        Args:
            codegen_rel_name: 相对于 .codegen/ 的文件名（如 "retrieve_base_memory.py"）。
                              也支持子目录路径（如 "sub/file.py"）。
            content: 文件内容。

        Returns:
            写入结果描述字符串。
        """
        staging_path = os.path.join(self._staging_dir, codegen_rel_name)
        os.makedirs(os.path.dirname(staging_path), exist_ok=True)
        with open(staging_path, "w", encoding="utf-8") as f:
            f.write(content)

        if codegen_rel_name not in self._staged_files:
            self._staged_files.append(codegen_rel_name)

        logger.info(
            "CodegenStagingArea: 文件已写入暂存区 %s (%d chars)",
            codegen_rel_name, len(content),
        )
        return f"Staged: {codegen_rel_name} ({len(content)} chars)"

    def read_file(self, codegen_rel_name: str) -> str:
        """从暂存区读取文件。

        Args:
            codegen_rel_name: 相对于 .codegen/ 的文件名。

        Returns:
            文件内容，不存在时返回 ERROR 前缀字符串。
        """
        staging_path = os.path.join(self._staging_dir, codegen_rel_name)
        if not os.path.exists(staging_path):
            return f"ERROR: File not found in staging: {codegen_rel_name}"
        with open(staging_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_staged_abs_path(self, codegen_rel_name: str) -> str:
        """获取暂存区文件的绝对路径（用于 load_consumer_class 等验证）。

        Args:
            codegen_rel_name: 相对于 .codegen/ 的文件名。

        Returns:
            暂存区中该文件的绝对路径。
        """
        return os.path.join(self._staging_dir, codegen_rel_name)

    @property
    def staged_files(self) -> list[str]:
        """返回暂存区中已写入的文件列表（相对于 .codegen/ 的路径）。"""
        return list(self._staged_files)

    # ------------------------------------------------------------------
    # 提交操作（原子性覆盖）
    # ------------------------------------------------------------------

    def commit(self) -> dict[str, Any]:
        """原子性提交：备份旧文件 + 覆盖新文件 + 清理暂存区。

        Returns:
            提交结果字典，包含：
            - committed: bool，是否成功提交
            - backed_up: list[str]，已备份的旧文件路径
            - deployed: list[str]，已部署的新文件路径
            - error: str，错误信息（如有）
        """
        if not self._staged_files:
            logger.info("CodegenStagingArea: 暂存区为空，无需提交")
            return {
                "committed": False,
                "backed_up": [],
                "deployed": [],
                "error": "暂存区为空",
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backed_up: list[str] = []
        deployed: list[str] = []

        try:
            # 确保快照目录存在
            os.makedirs(self._snapshot_dir, exist_ok=True)

            # Step 1: 备份即将被覆盖的旧文件
            for rel_name in self._staged_files:
                old_path = os.path.join(self._codegen_dir, rel_name)
                if os.path.exists(old_path):
                    # 生成备份文件名
                    name_no_ext, ext = os.path.splitext(rel_name)
                    # 将子目录分隔符替换为下划线，避免备份路径嵌套
                    safe_name = name_no_ext.replace("/", "_").replace("\\", "_")
                    backup_name = f"{safe_name}_{timestamp}{ext}"
                    backup_path = os.path.join(self._snapshot_dir, backup_name)

                    shutil.copy2(old_path, backup_path)
                    backed_up.append(f"{CODEGEN_SNAPSHOT_DIR}/{backup_name}")
                    logger.info(
                        "CodegenStagingArea: 旧文件已备份 %s -> %s",
                        rel_name, backup_name,
                    )

            # Step 2: 将暂存区文件覆盖到 .codegen/
            for rel_name in self._staged_files:
                staging_path = os.path.join(self._staging_dir, rel_name)
                target_path = os.path.join(self._codegen_dir, rel_name)

                if not os.path.exists(staging_path):
                    logger.warning(
                        "CodegenStagingArea: 暂存文件不存在，跳过: %s",
                        rel_name,
                    )
                    continue

                # 确保目标目录存在
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.move(staging_path, target_path)
                deployed.append(f"{CODEGEN_DIR}/{rel_name}")

                # 同步更新 FileSystemStore 的 BM25 索引
                codegen_rel_path = f"{CODEGEN_DIR}/{rel_name}"
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self._fs._bm25_index_update(codegen_rel_path, content)
                except Exception:
                    pass

            # Step 3: 清理暂存区
            self._cleanup_staging()

            logger.info(
                "CodegenStagingArea: 原子提交完成 (backed_up=%d, deployed=%d)",
                len(backed_up), len(deployed),
            )
            return {
                "committed": True,
                "backed_up": backed_up,
                "deployed": deployed,
                "error": "",
            }

        except Exception as e:
            logger.error("CodegenStagingArea: 提交失败: %s", e)
            return {
                "committed": False,
                "backed_up": backed_up,
                "deployed": deployed,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # 丢弃操作
    # ------------------------------------------------------------------

    def discard(self) -> None:
        """丢弃暂存区所有内容，不影响现有 .codegen/ 目录。"""
        self._cleanup_staging()
        logger.info("CodegenStagingArea: 暂存区已丢弃")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _cleanup_staging(self) -> None:
        """清理暂存目录。"""
        try:
            if os.path.exists(self._staging_dir):
                shutil.rmtree(self._staging_dir)
            self._staged_files.clear()
        except Exception as e:
            logger.warning("CodegenStagingArea: 清理暂存区失败: %s", e)
