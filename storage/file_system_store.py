"""文件系统存储后端。

提供基于本地文件系统的记忆存储，包含：
- FileSystemStore: 文件系统 + BM25 全文检索
- 相关常量（FS_ALLOWED_COMMANDS 等白名单）
- 通用数据结构（MemoryEntry / SearchResult）
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any
import logger.logger as logger
from debug import LoggedMethodsMixin


# ---------------------------------------------------------------------------
# 记忆库固定目录 & 文件路径常量
# ---------------------------------------------------------------------------

# 代码生成根目录
CODEGEN_DIR = ".codegen"
# 代码快照目录
CODEGEN_SNAPSHOT_DIR = f"{CODEGEN_DIR}/snapshots"
# 代码生成暂存目录（原子性提交前的临时存放区）
CODEGEN_STAGING_DIR = ".codegen_staging"
# 原始会话归档目录
SOURCE_SESSIONS_DIR = "source_sessions"

# ── 记忆库索引文件 ──
INDEX_DIR = ".meta"
INDEX_FILE_PATH = f"{INDEX_DIR}/index.md"
INDEX_FILE_PATH_FALLBACK = "index.md"

# ── 演进策略 ──
EVOLVE_STRATEGY_FILE_PATH = f"{CODEGEN_DIR}/evolve_strategy.md"
EVOLVE_STRATEGY_FILENAME = "evolve_strategy.md"

# ── 摄入相关 ──
INGEST_CODEGEN_FILENAME = "ingest_memory.py"
INGEST_BASE_CLASS_FILENAME = "ingest_base_memory.py"
INGEST_SUBCLASS_PATTERN = "ingest_memory_v{idx}.py"
INGEST_NEW_SUBCLASS_FILENAME = "ingest_memory_new.py"
MULTI_INGEST_STRATEGIES_FILENAME = "multi_ingest_strategies.md"
INGEST_ATOMIC_DESIGN_FILENAME = "ingest_atomic_design.md"
CODEGEN_INGEST_STRATEGY_FILENAME = "codegen_ingest_strategy.md"

# 摄入完整路径
INGEST_CODEGEN_FILE_PATH = f"{CODEGEN_DIR}/{INGEST_CODEGEN_FILENAME}"
INGEST_BASE_CLASS_FILE_PATH = f"{CODEGEN_DIR}/{INGEST_BASE_CLASS_FILENAME}"
INGEST_NEW_SUBCLASS_FILE_PATH = f"{CODEGEN_DIR}/{INGEST_NEW_SUBCLASS_FILENAME}"
MULTI_INGEST_STRATEGIES_FILE_PATH = f"{CODEGEN_DIR}/{MULTI_INGEST_STRATEGIES_FILENAME}"
INGEST_ATOMIC_DESIGN_FILE_PATH = f"{CODEGEN_DIR}/{INGEST_ATOMIC_DESIGN_FILENAME}"
CODEGEN_INGEST_STRATEGY_FILE_PATH = f"{CODEGEN_DIR}/{CODEGEN_INGEST_STRATEGY_FILENAME}"

# ── 消费相关 ──
RETRIEVE_CODEGEN_FILENAME = "retrieve_memory.py"
RETRIEVE_BASE_CLASS_FILENAME = "retrieve_base_memory.py"
RETRIEVE_SUBCLASS_PATTERN = "retrieve_memory_v{idx}.py"
RETRIEVE_NEW_SUBCLASS_FILENAME = "retrieve_memory_new.py"
MULTI_RETRIEVE_STRATEGIES_FILENAME = "multi_retrieve_strategies.md"
RETRIEVE_ATOMIC_DESIGN_FILENAME = "retrieve_atomic_design.md"
CODEGEN_RETRIEVE_STRATEGY_FILENAME = "codegen_retrieve_strategy.md"

# 消费完整路径
RETRIEVE_CODEGEN_FILE_PATH = f"{CODEGEN_DIR}/{RETRIEVE_CODEGEN_FILENAME}"
RETRIEVE_BASE_CLASS_FILE_PATH = f"{CODEGEN_DIR}/{RETRIEVE_BASE_CLASS_FILENAME}"
RETRIEVE_NEW_SUBCLASS_FILE_PATH = f"{CODEGEN_DIR}/{RETRIEVE_NEW_SUBCLASS_FILENAME}"
MULTI_RETRIEVE_STRATEGIES_FILE_PATH = f"{CODEGEN_DIR}/{MULTI_RETRIEVE_STRATEGIES_FILENAME}"
RETRIEVE_ATOMIC_DESIGN_FILE_PATH = f"{CODEGEN_DIR}/{RETRIEVE_ATOMIC_DESIGN_FILENAME}"
CODEGEN_RETRIEVE_STRATEGY_FILE_PATH = f"{CODEGEN_DIR}/{CODEGEN_RETRIEVE_STRATEGY_FILENAME}"


# ---------------------------------------------------------------------------
# Shell 沙箱白名单 —— 与 memory_repo.py 的 ALLOWED_COMMANDS 对齐
# ---------------------------------------------------------------------------

# FileSystemStore.execute_bash() 允许的命令集合。
# 消费阶段（retrieve）使用时，会额外叠加一层只读过滤，禁用写类命令。
FS_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    # 读取 / 浏览
    "cat", "ls", "head", "tail", "wc", "sort", "uniq", "tree",
    "find", "grep", "awk", "sed", "cut", "tr", "diff", "jq",
    "basename", "dirname", "realpath", "file", "stat",
    # 受控的写类命令（默认禁用，需显式开启 allow_write=True 才放行）
    "mkdir", "touch", "mv", "cp", "rm", "chmod",
    "tee", "echo", "printf",
    # 管道 / 变量处理工具
    "xargs", "yes",
    # Shell 内建（用于 pipeline / test 表达式等）
    "true", "false", "test", "type", "source", ".", "read",
    "export", "set", "unset", "local", "return", "exit", "cd",
    # Shell 控制流关键字
    "if", "then", "else", "elif", "fi", "for", "do", "done",
    "while", "until", "case", "esac", "in",
    # 解释器（允许调用 python3/bash 一行脚本来做 yaml / frontmatter 解析）
    "python3", "python", "bash", "sh",
    # git 只读操作（log / show / diff / blame / status 等；写类子命令被 blocked）
    "git",
    # 日期 / 时间
    "date",
})

# 写类命令（retrieve 等只读场景下强制禁用）
_FS_WRITE_COMMANDS: frozenset[str] = frozenset({
    "mkdir", "touch", "mv", "cp", "rm", "chmod",
    "tee", "printf",  # echo 不写盘则无害，但 `echo x > file` 通过 shell 重定向；下方单独过滤
})

# git 的写类子命令：分两级。
# _FS_GIT_ALWAYS_BLOCKED   —— 任何模式都禁止（会破坏分支 / 远端 / worktree）
# _FS_GIT_WRITE_SUBCOMMANDS —— 只读模式额外禁止（写工作树 / 改历史）
_FS_GIT_ALWAYS_BLOCKED: frozenset[str] = frozenset({
    "checkout", "branch", "merge", "rebase", "switch", "worktree",
    "push", "pull", "fetch", "remote", "clone",
})
_FS_GIT_WRITE_SUBCOMMANDS: frozenset[str] = frozenset({
    "add", "commit", "rm", "mv", "reset", "restore", "clean", "stash",
})

# 内部元数据目录 / 文件：不暴露给 BM25、list_files、tree、read_file 等下游通道。
# 这些路径可能包含二进制对象（如 .git/index）、无关噪声（__pycache__）等。
# 模型可以通过 `fs_execute_bash` 的 `git log` / `git show` 等命令间接访问 git 历史，
# 但不应在 fs.* 的"文件"抽象里看到这些路径。
_FS_INTERNAL_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".DS_Store",
})
_FS_INTERNAL_SKIP_FILES: frozenset[str] = frozenset({
    ".DS_Store",
})


# ---------------------------------------------------------------------------
# 通用数据结构
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """一条记忆条目（跨存储后端通用）。"""
    id: str
    content: str
    memory_type: str = "fact"  # fact | insight | preference | event | relation
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source_session: str = ""


@dataclass
class SearchResult:
    """检索结果。"""
    entry: MemoryEntry
    score: float = 0.0
    source: str = ""  # "filesystem" | "vector" | "graph"


# ---------------------------------------------------------------------------
# BM25 增量倒排索引
# ---------------------------------------------------------------------------


class _BM25IncrementalIndex:
    """增量维护的 BM25 倒排索引。

    核心思路：
    - 维护 docs: dict[str, tuple[str, list[str]]]  # path -> (raw_content, tokens)
    - 文件变更时只更新对应 path 的 entry，无需重建全量索引
    - 搜索时使用 rank_bm25.BM25Okapi 计算分数（惰性重建 BM25 对象）
    - BM25 对象在文档集变更后标记为 dirty，下次搜索时重建
    """

    def __init__(self) -> None:
        self._docs: dict[str, tuple[str, list[str]]] = {}  # path -> (content, tokens)
        self._bm25: Any = None  # BM25Okapi instance
        self._paths_ordered: list[str] = []  # 与 BM25 corpus 对齐的路径列表
        self._dirty: bool = True  # 标记是否需要重建 BM25 对象

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词：英文小写 split + CJK 字符 unigram/bigram。"""
        text_lower = text.lower()
        # 英文 token
        english = re.findall(r"[a-z0-9_]+", text_lower)
        # CJK bigram
        cjk_segments = re.findall(
            r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+", text_lower
        )
        cjk: list[str] = []
        for seg in cjk_segments:
            for i in range(len(seg)):
                cjk.append(seg[i])
                if i + 1 < len(seg):
                    cjk.append(seg[i : i + 2])
        return english + cjk

    def add_or_update(self, rel_path: str, content: str) -> None:
        """添加或更新一个文档的索引。"""
        tokens = self._tokenize(content)
        self._docs[rel_path] = (content, tokens)
        self._dirty = True

    def remove(self, rel_path: str) -> None:
        """从索引中移除一个文档。"""
        if rel_path in self._docs:
            del self._docs[rel_path]
            self._dirty = True

    def has(self, rel_path: str) -> bool:
        """检查路径是否已在索引中。"""
        return rel_path in self._docs

    def is_empty(self) -> bool:
        """索引是否为空。"""
        return len(self._docs) == 0

    def doc_count(self) -> int:
        """索引中的文档数量。"""
        return len(self._docs)

    def _rebuild_if_dirty(self) -> None:
        """惰性重建 BM25 对象（仅在 dirty 时）。"""
        if not self._dirty:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            self._paths_ordered = list(self._docs.keys())
            self._dirty = False
            return

        self._paths_ordered = sorted(self._docs.keys())
        if not self._paths_ordered:
            self._bm25 = None
            self._dirty = False
            return

        tokenized_corpus = [self._docs[p][1] for p in self._paths_ordered]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 10,
        scope: list[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """执行 BM25 搜索。

        Args:
            query: 搜索查询。
            top_k: 返回的最大结果数。
            scope: 可选的路径范围过滤（前缀匹配）。

        Returns:
            [(文件路径, BM25分数, 匹配片段)]
        """
        self._rebuild_if_dirty()

        if not self._paths_ordered:
            return []

        query_tokens = self._tokenize(query)

        if self._bm25 is None:
            # rank_bm25 不可用，回退到简单关键词匹配
            return self._fallback_search(query, top_k, scope)

        scores = self._bm25.get_scores(query_tokens)

        # 组合路径和分数
        scored: list[tuple[str, float, str]] = []
        for i, (path, score) in enumerate(zip(self._paths_ordered, scores)):
            if score <= 0:
                continue
            # scope 过滤
            if scope and not self._path_in_scope(path, scope):
                continue
            content = self._docs[path][0]
            # 提取匹配片段：优先展示包含查询关键词的行
            snippet = self._extract_snippet(content, query_tokens)
            scored.append((path, float(score), snippet))

        # BM25 在文档数量极少时可能因 IDF=0 导致全部返回 0 分，
        # 此时回退到关键词匹配
        if not scored:
            return self._fallback_search(query, top_k, scope)

        # 按分数降序排列
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _fallback_search(
        self, query: str, top_k: int, scope: list[str] | None
    ) -> list[tuple[str, float, str]]:
        """rank_bm25 不可用时的简单关键词匹配回退。"""
        keywords = query.lower().split()
        results: list[tuple[str, float, str]] = []
        for path, (content, _) in self._docs.items():
            if scope and not self._path_in_scope(path, scope):
                continue
            content_lower = content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                snippet = content[:200].replace("\n", " ")
                results.append((path, float(score), snippet))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    @staticmethod
    def _path_in_scope(path: str, scope: list[str]) -> bool:
        """检查路径是否在指定的 scope 范围内（前缀匹配）。"""
        for s in scope:
            if path == s or path.startswith(s):
                return True
        return False

    @staticmethod
    def _extract_snippet(content: str, query_tokens: list[str]) -> str:
        """提取包含查询关键词的匹配片段。"""
        lines = content.split("\n")
        matched_lines: list[str] = []
        for line in lines:
            line_lower = line.lower()
            if any(t in line_lower for t in query_tokens[:5]):
                matched_lines.append(line.strip())
                if len(matched_lines) >= 3:
                    break
        if matched_lines:
            return " | ".join(matched_lines)[:300]
        # 无精确行匹配时返回前200字符
        return content[:200].replace("\n", " ")


# ---------------------------------------------------------------------------
# 文件系统存储后端
# ---------------------------------------------------------------------------

class FileSystemStore(LoggedMethodsMixin):
    """文件系统存储后端 — 目录结构完全由模型自主决定。

    提供基础操作：
    - 创建/读取/更新/删除文件
    - 目录浏览（tree）
    - BM25 全文检索（基于 rank_bm25）
    - 灵活的 Python 接口供模型执行任意文件操作
    """
    _logged_methods = {"commit_all","get_last_commit","tree","read_file","write_file",
                       "list_files","search","execute_bash","append_file","delete_file",
                       "grep","read_lines","write_lines","get_structure_summary","list_files",
                       "search_bm25"
                       }

    def __init__(self, base_path: str, enable_git: bool = False):
        """Initialize file-system store rooted at ``base_path``.

        Args:
            base_path: Absolute root directory for the memory repo.
            enable_git: When True, ``git init`` the directory (if not yet a
                repo) and configure a local identity. Enables :meth:`commit_all`
                and allows agents / the dispatcher to produce a commit history
                of each ingest / consolidate.
        """
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.enable_git = bool(enable_git)
        if self.enable_git:
            self._ensure_git_repo()

        # 增量 BM25 索引
        self._bm25_index: _BM25IncrementalIndex = _BM25IncrementalIndex()
        self._bm25_index_initialized: bool = False

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """将 FileSystemStore 序列化为可 JSON 化的配置字典。

        Returns:
            包含重建实例所需全部参数的字典。
        """
        return {
            "backend": "filesystem",
            "base_path": self.base_path,
            "enable_git": self.enable_git,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FileSystemStore":
        """从配置字典反序列化创建 FileSystemStore 实例。

        Args:
            config: 序列化配置字典，至少包含 'base_path' 字段。

        Returns:
            FileSystemStore 实例。
        """
        base_path = config.get("base_path", "")
        enable_git = config.get("enable_git", False)
        return cls(base_path=base_path, enable_git=enable_git)

    # ------------------------------------------------------------------
    # Git helpers (optional; enabled via enable_git=True)
    # ------------------------------------------------------------------

    def _git(self, *args: str, timeout: int = 15) -> tuple[int, str, str]:
        """Run a git subcommand inside ``base_path``.

        Returns (returncode, stdout, stderr). Args are passed as a list so
        shell injection is not possible; callers control the argv.
        """
        try:
            proc = subprocess.run(  # noqa: S603 — argv is controlled internally
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_path),
            )
        except subprocess.TimeoutExpired:
            return (124, "", f"git {' '.join(args)} timed out after {timeout}s")
        except FileNotFoundError:
            return (127, "", "git binary not found on PATH")
        return (proc.returncode, proc.stdout, proc.stderr)

    def _ensure_git_repo(self) -> None:
        """Initialize ``base_path`` as a git repo on first use."""
        import os
        if os.path.isdir(os.path.join(self.base_path, ".git")):
            return
        rc, _, err = self._git("init", "-q", "-b", "main")
        if rc != 0:
            # Older gits don't support `-b` for init; retry without it.
            rc, _, err = self._git("init", "-q")
        if rc != 0:
            logger.warning("FileSystemStore: git init failed at %s: %s", self.base_path, err)
            self.enable_git = False
            return
        # Local identity so auto-commits don't error out on user systems
        # without a global gitconfig. Scope is per-repo (no --global).
        self._git("config", "user.email", "agent-memory@local")
        self._git("config", "user.name", "agent-memory")
        # Create an initial empty commit so later diffs / log always have a base.
        rc, _, _ = self._git("commit", "--allow-empty", "-m", "init: create memory repository")
        if rc != 0:
            logger.debug("FileSystemStore: initial empty commit skipped (non-fatal)")

    def commit_all(
        self,
        message: str,
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        """Stage every change and create one commit.

        Returns a dict with keys ``committed`` (bool), ``hash`` (short commit
        sha or empty), ``message`` (string actually used), and ``reason``
        (explanation when not committed). Never raises — callers can log and
        keep going.
        """
        if not self.enable_git:
            return {
                "committed": False,
                "hash": "",
                "message": message,
                "reason": "git is not enabled on this FileSystemStore",
            }

        # Stage everything (new, modified, deleted)
        rc, _, err = self._git("add", "-A")
        if rc != 0:
            return {
                "committed": False, "hash": "", "message": message,
                "reason": f"git add failed: {err.strip()}",
            }

        # Skip noop commits unless caller explicitly wants one.
        if not allow_empty:
            rc_diff, out, _ = self._git("diff", "--cached", "--name-only")
            if rc_diff == 0 and not out.strip():
                return {
                    "committed": False, "hash": "", "message": message,
                    "reason": "no staged changes",
                }

        args = ["commit", "-m", message[:4000]]  # clip absurdly long messages
        if allow_empty:
            args.append("--allow-empty")
        rc, out, err = self._git(*args)
        if rc != 0:
            return {
                "committed": False, "hash": "", "message": message,
                "reason": f"git commit failed: {(err or out).strip()}",
            }
        sha = self.get_last_commit()
        return {
            "committed": True, "hash": sha, "message": message, "reason": "",
        }

    def get_last_commit(self) -> str:
        """Return the short hash of HEAD, or empty string if unavailable."""
        if not self.enable_git:
            return ""
        rc, out, _ = self._git("rev-parse", "--short", "HEAD")
        return out.strip() if rc == 0 else ""

    # ------------------------------------------------------------------
    # Branch 管理方法（内部 API，仅供系统层调用，不暴露给模型 sandbox）
    # ------------------------------------------------------------------

    def get_current_branch(self) -> str:
        """获取当前所在分支名，若非 git repo 或 detached HEAD 则返回空字符串。"""
        if not self.enable_git:
            return ""
        rc, out, _ = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return out.strip() if rc == 0 else ""

    def create_branch(self, branch_name: str) -> dict[str, Any]:
        """从当前 HEAD 创建新分支并切换过去。

        Args:
            branch_name: 新分支名称（如 "session/s0464_00"）。

        Returns:
            包含 created (bool), branch (str), error (str) 的字典。
        """
        if not self.enable_git:
            return {"created": False, "branch": branch_name, "error": "git not enabled"}

        # 先确保工作区干净（自动 commit 未提交的变更）
        self.commit_all(f"auto-commit: before creating branch {branch_name}")

        # 创建并切换到新分支
        rc, out, err = self._git("checkout", "-b", branch_name)
        if rc != 0:
            return {"created": False, "branch": branch_name, "error": err.strip()}

        logger.info("FileSystemStore: created and switched to branch '%s'", branch_name)
        return {"created": True, "branch": branch_name, "error": ""}

    def checkout_branch(self, branch_name: str) -> dict[str, Any]:
        """切换到指定分支。

        Args:
            branch_name: 目标分支名称。

        Returns:
            包含 switched (bool), branch (str), error (str) 的字典。
        """
        if not self.enable_git:
            return {"switched": False, "branch": branch_name, "error": "git not enabled"}

        # 先确保工作区干净
        self.commit_all(f"auto-commit: before switching to branch {branch_name}")

        rc, out, err = self._git("checkout", branch_name)
        if rc != 0:
            return {"switched": False, "branch": branch_name, "error": err.strip()}

        logger.info("FileSystemStore: switched to branch '%s'", branch_name)
        return {"switched": True, "branch": branch_name, "error": ""}

    def merge_branch(
        self,
        source_branch: str,
        target_branch: str = "main",
        *,
        no_ff: bool = False,
    ) -> dict[str, Any]:
        """将 source_branch 合并到 target_branch。

        流程：切换到 target → merge source → 返回结果。

        Args:
            source_branch: 要合并的源分支。
            target_branch: 合并目标分支（默认 "main"）。
            no_ff: 若为 True，强制产生 merge commit（即使可以 fast-forward）。

        Returns:
            包含 merged (bool), hash (str), error (str) 的字典。
        """
        if not self.enable_git:
            return {"merged": False, "hash": "", "error": "git not enabled"}

        # 先确保当前分支工作区干净
        self.commit_all(f"auto-commit: before merging {source_branch} into {target_branch}")

        # 切换到目标分支
        rc, _, err = self._git("checkout", target_branch)
        if rc != 0:
            return {"merged": False, "hash": "", "error": f"checkout {target_branch} failed: {err.strip()}"}

        # 执行 merge
        merge_args = ["merge", source_branch]
        if no_ff:
            merge_args.append("--no-ff")
            merge_args.extend(["-m", f"merge: {source_branch} into {target_branch}"])

        rc, out, err = self._git(*merge_args)
        if rc != 0:
            # merge 冲突时尝试 abort 并返回错误
            self._git("merge", "--abort")
            return {"merged": False, "hash": "", "error": f"merge failed: {(err or out).strip()}"}

        sha = self.get_last_commit()
        logger.info(
            "FileSystemStore: merged '%s' into '%s' -> %s",
            source_branch, target_branch, sha,
        )
        return {"merged": True, "hash": sha, "error": ""}

    def delete_branch(self, branch_name: str) -> dict[str, Any]:
        """删除已合并的分支。

        Args:
            branch_name: 要删除的分支名称。

        Returns:
            包含 deleted (bool), branch (str), error (str) 的字典。
        """
        if not self.enable_git:
            return {"deleted": False, "branch": branch_name, "error": "git not enabled"}

        # 使用 -d（仅删除已合并的分支），避免误删未合并的工作
        rc, out, err = self._git("branch", "-d", branch_name)
        if rc != 0:
            return {"deleted": False, "branch": branch_name, "error": err.strip()}

        logger.info("FileSystemStore: deleted branch '%s'", branch_name)
        return {"deleted": True, "branch": branch_name, "error": ""}

    def tree(self, max_depth: int = 3) -> str:
        """返回目录树结构字符串。默认排除 ``.git/`` 等元数据目录。"""
        import os
        lines = []
        for root, dirs, files in os.walk(self.base_path):
            # 屏蔽 .git / __pycache__ 等元数据目录，避免把 git 内部对象曝露给模型
            dirs[:] = [d for d in dirs if d not in _FS_INTERNAL_SKIP_DIRS]
            level = root.replace(self.base_path, "").count(os.sep)
            if level >= max_depth:
                dirs.clear()
                continue
            indent = "  " * level
            lines.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = "  " * (level + 1)
            for f in sorted(files):
                if f in _FS_INTERNAL_SKIP_FILES:
                    continue
                lines.append(f"{sub_indent}{f}")
        return "\n".join(lines) if lines else "(empty)"

    def write_file(self, rel_path: str, content: str) -> str:
        """创建或覆写文件（自动创建父目录）。"""
        import os
        full_path = os.path.join(self.base_path, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        # 增量更新 BM25 索引
        self._bm25_index_update(rel_path, content)
        return f"Written: {rel_path} ({len(content)} chars)"

    def append_file(self, rel_path: str, content: str) -> str:
        """追加内容到文件。"""
        import os
        full_path = os.path.join(self.base_path, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(content)
        # 增量更新 BM25 索引（需要读取完整文件内容）
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_content = f.read()
            self._bm25_index_update(rel_path, full_content)
        except Exception:
            pass
        return f"Appended to: {rel_path} ({len(content)} chars)"

    def append_source_session_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Append raw session messages as JSONL into ``source_sessions/<id>.jsonl``.

        每条消息序列化为单独一行（`{"role": ..., "content": ...}`），追加写入。
        与 :mod:`agent_memory.tasks.retrieve_t2` prompt 中声明的归档格式保持
        一致：``source_sessions/<session_id>.jsonl``，每行一条 message。

        Notes:
            - 仅追加当前 buffer 的内容，不覆写；多次调用会按时间顺序累积。
            - ``session_id`` 中的路径分隔符等不安全字符会被替换为 ``_`` 后再
              拼接文件名，避免路径穿越。
            - 返回一个简短描述字符串，便于调用方做日志 / debug 观察。
        """
        import json
        import os

        if not session_id:
            return "ERROR: empty session_id"
        if not messages:
            return "noop: empty messages buffer"

        # 与 retrieve prompt 中声明的文件格式对齐：source_sessions/<id>.jsonl
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("_") or "session"
        rel_path = f"{SOURCE_SESSIONS_DIR}/{safe_id}.jsonl"
        full_path = os.path.join(self.base_path, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        written = 0
        with open(full_path, "a", encoding="utf-8") as f:
            for m in messages:
                # 只保留 role / content 两个核心字段；若原消息带额外字段也一并保留
                # （例如 speaker name、timestamp），方便后续回放时追溯更多上下文。
                if not isinstance(m, dict):
                    continue
                record = dict(m)
                record.setdefault("role", "user")
                record.setdefault("content", "")
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                written += 1
        return f"Appended {written} msg(s) to {rel_path}"

    def read_file(self, rel_path: str) -> str:
        """读取文件内容。

        对非 UTF-8 文件（例如被 ``list_files`` 意外命中的二进制）降级为
        ``errors='replace'``，避免 ``UnicodeDecodeError`` 向上层抛出。
        """
        import os
        full_path = os.path.join(self.base_path, rel_path)
        if not os.path.exists(full_path):
            return f"ERROR: File not found: {rel_path}"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    def read_index(self) -> tuple[str, str]:
        """读取记忆库索引文件，返回 (路径, 内容)。

        优先读取 .meta/index.md，不存在则回退到 index.md。
        """
        for path in (INDEX_FILE_PATH, INDEX_FILE_PATH_FALLBACK):
            content = self.read_file(path)
            if not content.startswith("ERROR:"):
                return path, content
        return INDEX_FILE_PATH, "(index.md 不存在 — 记忆库可能为空)"

    def delete_file(self, rel_path: str) -> str:
        """删除文件。"""
        import os
        full_path = os.path.join(self.base_path, rel_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            # 从 BM25 索引中移除
            self._bm25_index_remove(rel_path)
            return f"Deleted: {rel_path}"
        return f"Not found: {rel_path}"

    # ------------------------------------------------------------------
    # 结构化 grep / read_lines — fs_execute_bash 的低成本替代
    # ------------------------------------------------------------------
    # 目的：ingest / retrieve / consolidate 中大量的 "grep -nE <pat> <path> +
    # sed -n 'N,Mp' <path>" 组合占了很大一部分 tool-call 和 token 成本
    # （每次 bash 启动 / 返回全量 stdout）。这里提供两个结构化接口，让 LLM
    # 可以一步拿到 {line,text,before,after} 的精炼结果，避免再走 shell。

    def grep(
        self,
        pattern: str,
        paths: list[str] | str,
        *,
        context_lines: int = 2,
        max_matches: int = 50,
        case_insensitive: bool = True,
        regex: bool = True,
    ) -> list[dict[str, Any]]:
        """在指定路径下做正则 / 子串匹配，返回结构化命中结果。

        Args:
            pattern: 匹配模式。``regex=True`` 时按 Python ``re`` 语法；否则按
                子串匹配。
            paths: 单个相对路径或路径列表；若传目录则递归展开为目录下所有文件。
            context_lines: 每条命中附带的上下行数（前后各 N 行），默认 2。
            max_matches: 最多返回的命中数（保护过大文件），默认 50。
            case_insensitive: 大小写不敏感，默认 True。
            regex: 是否作为正则，默认 True。

        Returns:
            命中列表，每条形如::

                {
                  "path": "people/alex/facts.md",
                  "line": 42,
                  "match": "I stopped listening to book podcasts",
                  "before": ["...", "..."],   # 最多 context_lines 行
                  "after":  ["...", "..."],
                }

            无命中则返回空列表。格式错误（bad regex、路径不存在）返回
            ``[{"error": "..."}]``。
        """
        import os
        # 归一化 paths → file list
        if isinstance(paths, str):
            paths = [paths]
        file_list: list[str] = []
        for p in paths:
            full = os.path.join(self.base_path, p)
            if os.path.isdir(full):
                # 目录：递归枚举
                for root, dirs, files in os.walk(full):
                    dirs[:] = [d for d in dirs if d not in _FS_INTERNAL_SKIP_DIRS]
                    for f in files:
                        if f in _FS_INTERNAL_SKIP_FILES:
                            continue
                        rel = os.path.relpath(os.path.join(root, f), self.base_path)
                        file_list.append(rel)
            elif os.path.isfile(full):
                file_list.append(p)
            # 不存在的路径静默跳过（grep 的习惯）

        if not file_list:
            return [{"error": f"No files found under: {paths}"}]

        # 编译模式
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            if regex:
                prog = re.compile(pattern, flags)
            else:
                esc = re.escape(pattern)
                prog = re.compile(esc, flags)
        except re.error as e:
            return [{"error": f"invalid regex: {e}"}]

        results: list[dict[str, Any]] = []
        for rel in file_list:
            full = os.path.join(self.base_path, rel)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if prog.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    results.append({
                        "path": rel,
                        "line": i + 1,  # 1-indexed 与 grep -n 一致
                        "match": line.rstrip("\n"),
                        "before": [ln.rstrip("\n") for ln in lines[start:i]],
                        "after": [ln.rstrip("\n") for ln in lines[i + 1:end]],
                    })
                    if len(results) >= max_matches:
                        results.append({
                            "truncated": True,
                            "info": f"max_matches={max_matches} reached",
                        })
                        return results
        return results

    def read_lines(
        self,
        rel_path: str,
        start: int,
        end: int | None = None,
    ) -> str:
        """按行区间读取文件片段（等价于 ``sed -n 'start,end p'``）。

        Args:
            rel_path: 相对文件路径。
            start: 起始行号（1-indexed，包含）。
            end: 结束行号（1-indexed，包含）。省略或小于 start 则只读一行。

        Returns:
            切片后的文本（原样保留换行）；文件不存在返回 ``"ERROR: ..."``。
        """
        import os
        full = os.path.join(self.base_path, rel_path)
        if not os.path.exists(full):
            return f"ERROR: File not found: {rel_path}"
        if start < 1:
            start = 1
        if end is None or end < start:
            end = start
        try:
            out_lines: list[str] = []
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    if i < start:
                        continue
                    if i > end:
                        break
                    out_lines.append(f"{i}: {line}" if not line.endswith("\n")
                                     else f"{i}: {line}")
            return "".join(out_lines) if out_lines else (
                f"(empty: file has < {start} lines)"
            )
        except Exception as e:
            return f"ERROR: {e}"

    def list_files(self, rel_dir: str = "") -> list[str]:
        """列出目录下所有文件（相对路径）。默认跳过 ``.git/`` 等元数据目录。"""
        import os
        target = os.path.join(self.base_path, rel_dir)
        if not os.path.exists(target):
            return []
        result = []
        for root, dirs, files in os.walk(target):
            # 屏蔽 .git / __pycache__ 等元数据目录，避免把 git 内部二进制对象
            # 带进 BM25、read_file、log_fs_state 等下游通道。
            dirs[:] = [d for d in dirs if d not in _FS_INTERNAL_SKIP_DIRS]
            for f in files:
                if f in _FS_INTERNAL_SKIP_FILES:
                    continue
                full = os.path.join(root, f)
                result.append(os.path.relpath(full, self.base_path))
        return sorted(result)

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        scope: list[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """BM25 全文检索，返回 [(文件路径, 分数, 匹配片段)]。

        使用增量维护的倒排索引，避免每次搜索都全量重建。

        Args:
            query: 搜索查询文本。
            top_k: 返回的最大结果数。
            scope: 可选的路径范围过滤。支持文件路径或目录前缀，
                   例如 ["people/alex/", "events/"] 表示只在这些路径下搜索。
                   None 表示搜索全部文件。
        """
        # 确保索引已初始化
        self._ensure_bm25_index()

        try:
            return self._bm25_index.search(query, top_k=top_k, scope=scope)
        except Exception as e:
            logger.warning("BM25 incremental search failed: %s, falling back", e)
            # Fallback 到简单关键词匹配
            return self._simple_keyword_search(query, top_k)

    def _simple_keyword_search(self, query: str, top_k: int = 10) -> list[tuple[str, float, str]]:
        """简单关键词匹配（rank_bm25 不可用时的回退）。"""
        import os
        keywords = query.lower().split()
        results = []
        for rel_path in self.list_files():
            full_path = os.path.join(self.base_path, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content_lower = content.lower()
                score = sum(1 for kw in keywords if kw in content_lower)
                if score > 0:
                    snippet = content[:200].replace("\n", " ")
                    results.append((rel_path, float(score), snippet))
            except Exception:
                continue
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def execute_python(self, code: str) -> str:
        """执行任意 Python 代码操作文件系统（供模型灵活使用）。"""
        import os
        local_vars = {"base_path": self.base_path, "os": os}
        try:
            exec(code, {"__builtins__": __builtins__}, local_vars)
            return local_vars.get("result", "OK")
        except Exception as e:
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # Shell 执行（受限白名单 + 只读模式）
    # ------------------------------------------------------------------

    def get_structure_summary(self, max_depth: int = 2) -> str:
        """返回"一级目录 → 文件数"的人类可读摘要，供检索时注入 prompt。

        相比 ``tree()``，这个摘要更紧凑，适合放进 system/user prompt 里让模型
        在不消耗一轮工具调用的情况下快速感知记忆库布局。
        """
        if not os.path.isdir(self.base_path):
            return "(repository not initialized)"

        summary_lines: list[str] = []
        try:
            top_entries = sorted(os.listdir(self.base_path))
        except OSError as e:
            return f"(unable to list base_path: {e})"

        for entry in top_entries:
            if entry in _FS_INTERNAL_SKIP_DIRS or entry in _FS_INTERNAL_SKIP_FILES:
                continue
            full = os.path.join(self.base_path, entry)
            if os.path.isdir(full):
                # 统计该一级目录下的文件总数 + 按扩展名分桶
                ext_counts: dict[str, int] = {}
                total = 0
                for root, dirs, files in os.walk(full):
                    dirs[:] = [d for d in dirs if d not in _FS_INTERNAL_SKIP_DIRS]
                    for f in files:
                        if f in _FS_INTERNAL_SKIP_FILES:
                            continue
                        total += 1
                        ext = os.path.splitext(f)[1].lower() or "(noext)"
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1
                ext_str = ", ".join(
                    f"{ext}:{cnt}" for ext, cnt in sorted(ext_counts.items())
                )
                summary_lines.append(
                    f"- {entry}/ — {total} files ({ext_str})"
                    if ext_str else f"- {entry}/ — empty"
                )
            elif os.path.isfile(full):
                size = os.path.getsize(full)
                summary_lines.append(f"- {entry} ({size} bytes)")

        return "\n".join(summary_lines) if summary_lines else "(empty repository)"

    def execute_bash(
        self,
        command: str,
        timeout: int = 15,
        allow_write: bool = False,
        env_extra: dict[str, str] | None = None,
    ) -> str:
        """在 ``base_path`` 下执行受限的 shell 命令。

        **安全模型：**

        - 命令必须全部来自 :data:`FS_ALLOWED_COMMANDS` 白名单；未知命令直接拒绝。
        - ``allow_write=False``（默认，适用于 retrieve 等只读场景）时：
            - 拒绝 :data:`_FS_WRITE_COMMANDS` 中的写类命令；
            - 拒绝 shell 重定向写盘（``>`` / ``>>``；``2>`` / ``2>>`` 保留为
              stderr 重定向在 bash 里的常见用法，但这里统一按重定向写盘禁掉）；
            - 拒绝 git 的写类子命令（commit / add / reset / checkout / ...）。
        - 工作目录强制为 ``self.base_path``；注入 ``MEMORY_REPO_PATH`` 环境变量。
        - 通过 ``bash -c`` 执行（支持 pipe / && / ||），超时默认 15 秒。
        - stdout 截断到 5000 字符，stderr 截断到 2000 字符。

        Args:
            command: 要执行的 shell 命令字符串。
            timeout: 超时（秒）。
            allow_write: 是否允许写类命令（默认 False）。
            env_extra: 追加到子进程 env 的变量（如 ``CURRENT_TIME`` /
                ``CURRENT_SESSION_ID``）。

        Returns:
            形如 ``"$ <command>\\n<stdout>\\n[stderr:...]"`` 的字符串。失败时返
            回 ``"ERROR: ..."``。
        """
        if not command or not command.strip():
            return "ERROR: empty command"

        cmd = command.strip()

        # --- 1. 重定向写盘检查（仅在只读模式下） ---
        if not allow_write:
            # 识别 `>` 或 `>>` 重定向写盘；容忍 `2>&1` 这种 fd 复制。
            # 先剥掉被单/双引号包裹的字符串，避免内容里的 `>` 误判。
            stripped = re.sub(r'"[^"]*"', '""', cmd)
            stripped = re.sub(r"'[^']*'", "''", stripped)
            # 命中 `>file` 或 `>>file`，但不是 `2>&1` / `&>` 之类纯 fd 合流
            if re.search(r"(?<![0-9&])>>?\s*[^&\s]", stripped):
                return "ERROR: output redirection to file is disabled in read-only mode"

        # --- 2. git 写类子命令检查 ---
        for line in cmd.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "git" and i + 1 < len(parts):
                    # 跳过 -C <path>、--git-dir <path> 等形式的 flags
                    j = i + 1
                    while j < len(parts) and parts[j].startswith("-"):
                        j += 1
                        if j < len(parts) and not parts[j].startswith("-"):
                            j += 1
                    if j < len(parts):
                        sub = parts[j]
                        if sub in _FS_GIT_ALWAYS_BLOCKED:
                            return (
                                f"ERROR: blocked git subcommand: git {sub} "
                                "(branch / remote / worktree operations are never allowed)"
                            )
                        if (not allow_write) and sub in _FS_GIT_WRITE_SUBCOMMANDS:
                            return (
                                f"ERROR: blocked git subcommand: git {sub} "
                                "(only read-only git operations are allowed in read-only mode)"
                            )

        # --- 3. 命令白名单检查 ---
        cleaned = cmd
        # 去掉 heredoc
        cleaned = re.sub(
            r"<<-?\s*'?\"?(\w+)\"?'?\s*\n.*?\n\s*\1",
            "", cleaned, flags=re.DOTALL,
        )
        cleaned = re.sub(
            r"<<-?\s*'?\"?\w+\"?'?\s*\n.*",
            "", cleaned, flags=re.DOTALL,
        )
        # 去掉字符串字面量与注释
        cleaned = re.sub(r'"[^"]*"', '""', cleaned)
        cleaned = re.sub(r"'[^']*'", "''", cleaned)
        cleaned = re.sub(r"#[^\n]*", "", cleaned)

        tokens = re.findall(r"(?:^|[|;&]\s*)(\w[\w-]*)", cleaned, re.MULTILINE)
        for token in tokens:
            if token not in FS_ALLOWED_COMMANDS:
                return f"ERROR: command not allowed: {token!r}"
            if not allow_write and token in _FS_WRITE_COMMANDS:
                return (
                    f"ERROR: write-class command {token!r} is disabled in "
                    "read-only mode"
                )

        # --- 4. 路径越界检查（粗略） ---
        if ".." in cmd:
            # 允许 `..` 出现在字符串里（上面已经把字面量换成空引号），这里再粗查一下
            # 是否存在裸 `..` 路径片段
            if re.search(r"(?<![\w\-/])\.\.(?![\w\-])", cleaned):
                return "ERROR: path traversal ('..') is not allowed"

        # --- 5. 组装 env 并执行 ---
        env = os.environ.copy()
        env["MEMORY_REPO_PATH"] = str(self.base_path)
        if env_extra:
            env.update({k: str(v) for k, v in env_extra.items()})

        try:
            proc = subprocess.run(  # noqa: S603 — 已通过白名单+只读过滤
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_path),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s"
        except Exception as e:  # pragma: no cover
            return f"ERROR: execution failed: {e}"

        parts: list[str] = [f"$ {cmd}"]
        if proc.stdout:
            parts.append(proc.stdout[:5000])
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr[:2000]}")
        if proc.returncode != 0:
            parts.append(f"[exit {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"

    # ------------------------------------------------------------------
    # BM25 增量索引管理
    # ------------------------------------------------------------------

    def _ensure_bm25_index(self) -> None:
        """确保 BM25 索引已初始化（惰性加载：首次搜索时全量构建）。"""
        if self._bm25_index_initialized:
            return

        import os
        for rel_path in self.list_files():
            full_path = os.path.join(self.base_path, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._bm25_index.add_or_update(rel_path, content)
            except Exception:
                continue

        self._bm25_index_initialized = True
        logger.debug(
            "BM25 index initialized: %d documents", self._bm25_index.doc_count()
        )

    def _bm25_index_update(self, rel_path: str, content: str) -> None:
        """增量更新 BM25 索引中的单个文档。"""
        if not self._bm25_index_initialized:
            # 索引尚未初始化，跳过（首次搜索时会全量构建）
            return
        self._bm25_index.add_or_update(rel_path, content)

    def _bm25_index_remove(self, rel_path: str) -> None:
        """从 BM25 索引中移除单个文档。"""
        if not self._bm25_index_initialized:
            return
        self._bm25_index.remove(rel_path)
