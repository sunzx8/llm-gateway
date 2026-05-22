"""FileSystemStore 集成测试。

测试策略：
- 使用 pytest 的 tmp_path fixture 提供临时目录，无需外部依赖
- 每个测试类使用独立的 FileSystemStore 实例
- 覆盖：文件 CRUD、目录浏览、BM25 搜索、grep、read_lines、
         execute_bash（含安全白名单）、git 操作
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from storage.file_system_store import FileSystemStore


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_store(tmp_path: Path) -> "FileSystemStore":
    """创建一个以 tmp_path 为根目录的 FileSystemStore 实例（不启用 git）。"""
    from storage.file_system_store import FileSystemStore

    return FileSystemStore(base_path=str(tmp_path), enable_git=False)


@pytest.fixture
def fs_store_git(tmp_path: Path) -> "FileSystemStore":
    """创建一个启用 git 的 FileSystemStore 实例。"""
    from storage.file_system_store import FileSystemStore

    return FileSystemStore(base_path=str(tmp_path), enable_git=True)


# ===========================================================================
# 文件 CRUD 测试
# ===========================================================================


class TestFileSystemStoreCRUD:
    """文件创建、读取、追加、删除操作。"""

    def test_write_and_read_file(self, fs_store: "FileSystemStore") -> None:
        """写入文件后应能正确读取内容。"""
        result = fs_store.write_file("hello.txt", "Hello, World!")
        assert "Written" in result
        assert "hello.txt" in result

        content = fs_store.read_file("hello.txt")
        assert content == "Hello, World!"

    def test_write_file_creates_parent_dirs(self, fs_store: "FileSystemStore") -> None:
        """写入嵌套路径时应自动创建父目录。"""
        fs_store.write_file("a/b/c/deep.txt", "deep content")
        content = fs_store.read_file("a/b/c/deep.txt")
        assert content == "deep content"

    def test_write_file_overwrites(self, fs_store: "FileSystemStore") -> None:
        """对同一路径再次写入应覆盖原内容。"""
        fs_store.write_file("note.md", "version 1")
        fs_store.write_file("note.md", "version 2")
        assert fs_store.read_file("note.md") == "version 2"

    def test_append_file(self, fs_store: "FileSystemStore") -> None:
        """追加内容到已有文件。"""
        fs_store.write_file("log.txt", "line1\n")
        fs_store.append_file("log.txt", "line2\n")
        content = fs_store.read_file("log.txt")
        assert content == "line1\nline2\n"

    def test_append_file_creates_if_not_exists(self, fs_store: "FileSystemStore") -> None:
        """追加到不存在的文件应自动创建。"""
        fs_store.append_file("new.txt", "first line")
        assert fs_store.read_file("new.txt") == "first line"

    def test_read_file_not_found(self, fs_store: "FileSystemStore") -> None:
        """读取不存在的文件应返回 ERROR 信息。"""
        result = fs_store.read_file("nonexistent.txt")
        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_delete_file(self, fs_store: "FileSystemStore") -> None:
        """删除文件后应无法再读取。"""
        fs_store.write_file("to_delete.txt", "bye")
        result = fs_store.delete_file("to_delete.txt")
        assert "Deleted" in result

        content = fs_store.read_file("to_delete.txt")
        assert "ERROR" in content

    def test_delete_file_not_found(self, fs_store: "FileSystemStore") -> None:
        """删除不存在的文件应返回 Not found。"""
        result = fs_store.delete_file("ghost.txt")
        assert "Not found" in result


# ===========================================================================
# 目录浏览测试
# ===========================================================================


class TestFileSystemStoreTree:
    """目录树和文件列表。"""

    def test_tree_empty(self, fs_store: "FileSystemStore") -> None:
        """空目录的 tree 应只包含根目录名。"""
        tree = fs_store.tree()
        # 至少有根目录行
        assert "/" in tree or "(empty)" in tree

    def test_tree_with_files(self, fs_store: "FileSystemStore") -> None:
        """写入文件后 tree 应反映目录结构。"""
        fs_store.write_file("docs/readme.md", "# Hello")
        fs_store.write_file("docs/guide.md", "# Guide")
        fs_store.write_file("src/main.py", "print('hi')")

        tree = fs_store.tree()
        assert "docs" in tree
        assert "readme.md" in tree
        assert "src" in tree
        assert "main.py" in tree

    def test_tree_excludes_git_dir(self, fs_store_git: "FileSystemStore") -> None:
        """tree 应排除 .git 目录。"""
        fs_store_git.write_file("visible.txt", "yes")
        tree = fs_store_git.tree()
        assert ".git" not in tree
        assert "visible.txt" in tree

    def test_list_files(self, fs_store: "FileSystemStore") -> None:
        """list_files 应返回所有文件的相对路径列表。"""
        fs_store.write_file("a.txt", "a")
        fs_store.write_file("sub/b.txt", "b")

        files = fs_store.list_files()
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_list_files_subdir(self, fs_store: "FileSystemStore") -> None:
        """list_files 指定子目录应只返回该目录下的文件。"""
        fs_store.write_file("top.txt", "top")
        fs_store.write_file("sub/inner.txt", "inner")

        files = fs_store.list_files("sub")
        assert "sub/inner.txt" in files
        assert "top.txt" not in files

    def test_get_structure_summary(self, fs_store: "FileSystemStore") -> None:
        """get_structure_summary 应返回一级目录的文件统计。"""
        fs_store.write_file("people/alex.md", "# Alex")
        fs_store.write_file("people/bob.md", "# Bob")
        fs_store.write_file("events/2024.md", "# Events")

        summary = fs_store.get_structure_summary()
        assert "people" in summary
        assert "events" in summary
        # 应包含文件数
        assert "2 files" in summary


# ===========================================================================
# BM25 搜索测试
# ===========================================================================


class TestFileSystemStoreSearch:
    """BM25 全文检索。"""

    def test_search_bm25_basic(self, fs_store: "FileSystemStore") -> None:
        """BM25 搜索应返回包含关键词的文件。"""
        fs_store.write_file("python.md", "Python is a programming language")
        fs_store.write_file("java.md", "Java is also a programming language")
        fs_store.write_file("cooking.md", "How to make pasta")

        results = fs_store.search_bm25("python programming")
        assert len(results) > 0
        # 第一个结果应该是 python.md（最相关）
        paths = [r[0] for r in results]
        assert "python.md" in paths

    def test_search_bm25_no_match(self, fs_store: "FileSystemStore") -> None:
        """搜索无匹配内容时应返回空列表。"""
        fs_store.write_file("note.md", "hello world")
        results = fs_store.search_bm25("quantum physics")
        assert results == []

    def test_search_bm25_empty_repo(self, fs_store: "FileSystemStore") -> None:
        """空仓库搜索应返回空列表。"""
        results = fs_store.search_bm25("anything")
        assert results == []

    def test_search_bm25_top_k(self, fs_store: "FileSystemStore") -> None:
        """top_k 参数应限制返回数量。"""
        for i in range(10):
            fs_store.write_file(f"file_{i}.md", f"common keyword document {i}")

        results = fs_store.search_bm25("common keyword", top_k=3)
        assert len(results) <= 3


# ===========================================================================
# Grep 测试
# ===========================================================================


class TestFileSystemStoreGrep:
    """结构化 grep 搜索。"""

    def test_grep_basic_regex(self, fs_store: "FileSystemStore") -> None:
        """基本正则匹配应返回命中行。"""
        fs_store.write_file("code.py", "def hello():\n    return 'world'\n\ndef foo():\n    pass\n")

        results = fs_store.grep(r"def \w+", "code.py")
        assert len(results) == 2
        assert results[0]["line"] == 1
        assert "def hello" in results[0]["match"]
        assert results[1]["line"] == 4

    def test_grep_with_context(self, fs_store: "FileSystemStore") -> None:
        """应返回上下文行。"""
        content = "line1\nline2\nTARGET\nline4\nline5\n"
        fs_store.write_file("ctx.txt", content)

        results = fs_store.grep("TARGET", "ctx.txt", context_lines=1)
        assert len(results) == 1
        assert results[0]["before"] == ["line2"]
        assert results[0]["after"] == ["line4"]

    def test_grep_case_insensitive(self, fs_store: "FileSystemStore") -> None:
        """默认大小写不敏感。"""
        fs_store.write_file("mixed.txt", "Hello World\nhello world\nHELLO WORLD\n")

        results = fs_store.grep("hello", "mixed.txt", case_insensitive=True)
        assert len(results) == 3

    def test_grep_directory_recursive(self, fs_store: "FileSystemStore") -> None:
        """对目录搜索应递归所有文件。"""
        fs_store.write_file("src/a.py", "import os\n")
        fs_store.write_file("src/sub/b.py", "import sys\n")

        results = fs_store.grep("import", "src")
        assert len(results) == 2

    def test_grep_max_matches(self, fs_store: "FileSystemStore") -> None:
        """max_matches 应限制返回数量。"""
        lines = "\n".join([f"match line {i}" for i in range(100)])
        fs_store.write_file("many.txt", lines)

        results = fs_store.grep("match", "many.txt", max_matches=5)
        # 5 条命中 + 1 条 truncated 提示
        assert len(results) == 6
        assert results[-1].get("truncated") is True

    def test_grep_no_match(self, fs_store: "FileSystemStore") -> None:
        """无匹配时返回空列表。"""
        fs_store.write_file("empty_match.txt", "nothing here")
        results = fs_store.grep("zzzzz", "empty_match.txt")
        assert results == []

    def test_grep_invalid_regex(self, fs_store: "FileSystemStore") -> None:
        """无效正则应返回 error 信息。"""
        fs_store.write_file("x.txt", "content")
        results = fs_store.grep("[invalid", "x.txt")
        assert len(results) == 1
        assert "error" in results[0]


# ===========================================================================
# read_lines 测试
# ===========================================================================


class TestFileSystemStoreReadLines:
    """按行区间读取文件片段。"""

    def test_read_lines_range(self, fs_store: "FileSystemStore") -> None:
        """读取指定行范围。"""
        content = "\n".join([f"line {i}" for i in range(1, 11)])
        fs_store.write_file("numbered.txt", content)

        result = fs_store.read_lines("numbered.txt", start=3, end=5)
        assert "line 3" in result
        assert "line 4" in result
        assert "line 5" in result
        assert "line 2" not in result
        assert "line 6" not in result

    def test_read_lines_single(self, fs_store: "FileSystemStore") -> None:
        """只读一行（end 省略）。"""
        fs_store.write_file("single.txt", "aaa\nbbb\nccc\n")
        result = fs_store.read_lines("single.txt", start=2)
        assert "bbb" in result
        assert "aaa" not in result

    def test_read_lines_file_not_found(self, fs_store: "FileSystemStore") -> None:
        """文件不存在应返回 ERROR。"""
        result = fs_store.read_lines("ghost.txt", start=1)
        assert "ERROR" in result


# ===========================================================================
# execute_bash 测试（安全白名单）
# ===========================================================================


class TestFileSystemStoreBash:
    """Shell 命令执行（含安全限制）。"""

    def test_bash_basic_command(self, fs_store: "FileSystemStore") -> None:
        """允许的基本命令应正常执行。"""
        fs_store.write_file("test.txt", "hello bash")
        result = fs_store.execute_bash("cat test.txt")
        assert "hello bash" in result

    def test_bash_pipe(self, fs_store: "FileSystemStore") -> None:
        """管道命令应正常工作。"""
        fs_store.write_file("words.txt", "apple\nbanana\napricot\n")
        result = fs_store.execute_bash("grep '^a' words.txt | sort")
        assert "apple" in result
        assert "apricot" in result

    def test_bash_disallowed_command(self, fs_store: "FileSystemStore") -> None:
        """不在白名单中的命令应被拒绝。"""
        result = fs_store.execute_bash("curl http://example.com")
        assert "ERROR" in result
        assert "not allowed" in result

    def test_bash_readonly_blocks_write_commands(self, fs_store: "FileSystemStore") -> None:
        """只读模式下写类命令应被拒绝。"""
        result = fs_store.execute_bash("mkdir new_dir", allow_write=False)
        assert "ERROR" in result
        assert "read-only" in result.lower() or "disabled" in result.lower()

    def test_bash_readonly_blocks_redirect(self, fs_store: "FileSystemStore") -> None:
        """只读模式下输出重定向应被拒绝。"""
        result = fs_store.execute_bash("echo hello > output.txt", allow_write=False)
        assert "ERROR" in result
        assert "redirection" in result.lower() or "redirect" in result.lower()

    def test_bash_allow_write(self, fs_store: "FileSystemStore") -> None:
        """allow_write=True 时写类命令应被允许。"""
        result = fs_store.execute_bash("mkdir new_dir", allow_write=True)
        assert "ERROR" not in result
        # 验证目录确实被创建
        assert os.path.isdir(os.path.join(fs_store.base_path, "new_dir"))

    def test_bash_empty_command(self, fs_store: "FileSystemStore") -> None:
        """空命令应返回 ERROR。"""
        result = fs_store.execute_bash("")
        assert "ERROR" in result

    def test_bash_timeout(self, fs_store: "FileSystemStore") -> None:
        """超时命令应被终止。"""
        # sleep 不在白名单中，用 bash -c 'sleep 10' 来触发超时
        result = fs_store.execute_bash("bash -c 'while true; do true; done'", timeout=1)
        assert "ERROR" in result
        assert "timed out" in result.lower()

    def test_bash_path_traversal_blocked(self, fs_store: "FileSystemStore") -> None:
        """路径穿越（..）应被拒绝。"""
        result = fs_store.execute_bash("cat ../../etc/passwd")
        assert "ERROR" in result

    def test_bash_git_write_blocked_readonly(self, fs_store_git: "FileSystemStore") -> None:
        """只读模式下 git 写类子命令应被拒绝。"""
        result = fs_store_git.execute_bash("git add .", allow_write=False)
        assert "ERROR" in result
        assert "blocked" in result.lower()

    def test_bash_git_always_blocked(self, fs_store_git: "FileSystemStore") -> None:
        """git checkout/push 等危险子命令在任何模式下都应被拒绝。"""
        result = fs_store_git.execute_bash("git checkout main", allow_write=True)
        assert "ERROR" in result
        assert "blocked" in result.lower()

    def test_bash_git_read_allowed(self, fs_store_git: "FileSystemStore") -> None:
        """git log 等只读子命令应被允许。"""
        result = fs_store_git.execute_bash("git log --oneline -1")
        # 应该能看到 init commit
        assert "ERROR" not in result


# ===========================================================================
# Git 操作测试
# ===========================================================================


class TestFileSystemStoreGit:
    """Git 集成功能。"""

    def test_git_init(self, fs_store_git: "FileSystemStore") -> None:
        """enable_git=True 应自动初始化 git 仓库。"""
        git_dir = os.path.join(fs_store_git.base_path, ".git")
        assert os.path.isdir(git_dir)

    def test_commit_all_basic(self, fs_store_git: "FileSystemStore") -> None:
        """写入文件后 commit_all 应成功提交。"""
        fs_store_git.write_file("committed.txt", "tracked content")
        result = fs_store_git.commit_all("test: add committed.txt")

        assert result["committed"] is True
        assert result["hash"] != ""
        assert result["message"] == "test: add committed.txt"

    def test_commit_all_no_changes(self, fs_store_git: "FileSystemStore") -> None:
        """无变更时 commit_all 应返回 committed=False。"""
        result = fs_store_git.commit_all("empty commit")
        assert result["committed"] is False
        assert "no staged changes" in result["reason"]

    def test_commit_all_allow_empty(self, fs_store_git: "FileSystemStore") -> None:
        """allow_empty=True 时即使无变更也应能提交。"""
        result = fs_store_git.commit_all("empty allowed", allow_empty=True)
        assert result["committed"] is True

    def test_get_last_commit(self, fs_store_git: "FileSystemStore") -> None:
        """get_last_commit 应返回短 hash。"""
        sha = fs_store_git.get_last_commit()
        # 初始化时有一个 empty commit
        assert len(sha) >= 7

    def test_git_disabled(self, fs_store: "FileSystemStore") -> None:
        """enable_git=False 时 commit_all 应返回未启用提示。"""
        result = fs_store.commit_all("should not work")
        assert result["committed"] is False
        assert "not enabled" in result["reason"]

    def test_get_last_commit_disabled(self, fs_store: "FileSystemStore") -> None:
        """enable_git=False 时 get_last_commit 应返回空字符串。"""
        assert fs_store.get_last_commit() == ""


# ===========================================================================
# append_source_session_messages 测试
# ===========================================================================


class TestFileSystemStoreSessionMessages:
    """会话消息归档。"""

    def test_append_messages_basic(self, fs_store: "FileSystemStore") -> None:
        """应将消息追加为 JSONL 格式。"""
        import json

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ]
        result = fs_store.append_source_session_messages("sess_001", messages)
        assert "Appended 2 msg(s)" in result

        # 验证文件内容
        content = fs_store.read_file("source_sessions/sess_001.jsonl")
        lines = content.strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["role"] == "user"
        assert first["content"] == "你好"

    def test_append_messages_accumulates(self, fs_store: "FileSystemStore") -> None:
        """多次调用应累积追加。"""
        msgs1 = [{"role": "user", "content": "msg1"}]
        msgs2 = [{"role": "user", "content": "msg2"}]

        fs_store.append_source_session_messages("sess_002", msgs1)
        fs_store.append_source_session_messages("sess_002", msgs2)

        content = fs_store.read_file("source_sessions/sess_002.jsonl")
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_append_messages_empty_session_id(self, fs_store: "FileSystemStore") -> None:
        """空 session_id 应返回错误。"""
        result = fs_store.append_source_session_messages("", [{"role": "user", "content": "x"}])
        assert "ERROR" in result

    def test_append_messages_empty_list(self, fs_store: "FileSystemStore") -> None:
        """空消息列表应返回 noop。"""
        result = fs_store.append_source_session_messages("sess_003", [])
        assert "noop" in result

    def test_append_messages_sanitizes_session_id(self, fs_store: "FileSystemStore") -> None:
        """包含特殊字符的 session_id 应被安全化，文件只落在 source_sessions/ 下。"""
        messages = [{"role": "user", "content": "test"}]
        result = fs_store.append_source_session_messages("../../evil/path", messages)
        assert "Appended" in result
        # 文件应只存在于 source_sessions/ 目录下，不会穿越到上层
        files = fs_store.list_files("source_sessions")
        assert len(files) == 1
        # 验证文件确实在 source_sessions 目录内
        assert files[0].startswith("source_sessions/")
        # 验证不存在 source_sessions 之外的文件
        all_files = fs_store.list_files()
        for f in all_files:
            assert f.startswith("source_sessions/")


# ===========================================================================
# execute_python 测试
# ===========================================================================


class TestFileSystemStorePython:
    """Python 代码执行。"""

    def test_execute_python_basic(self, fs_store: "FileSystemStore") -> None:
        """应能执行简单的 Python 代码。"""
        code = "result = os.listdir(base_path)"
        result = fs_store.execute_python(code)
        # 空目录应返回空列表
        assert isinstance(result, list)

    def test_execute_python_file_ops(self, fs_store: "FileSystemStore") -> None:
        """应能通过 Python 代码操作文件。"""
        fs_store.write_file("data.txt", "12345")
        code = """
path = os.path.join(base_path, 'data.txt')
with open(path) as f:
    result = len(f.read())
"""
        result = fs_store.execute_python(code)
        assert result == 5

    def test_execute_python_error(self, fs_store: "FileSystemStore") -> None:
        """执行出错应返回 ERROR 信息。"""
        result = fs_store.execute_python("result = 1 / 0")
        assert "ERROR" in result
