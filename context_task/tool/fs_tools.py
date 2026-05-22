"""
文件系统工具集

包含所有文件系统相关的工具类。
"""

from typing import Any

from .base_tool import BaseTool


class FsWriteTool(BaseTool):
    name = "fs_write"
    description = (
        "在记忆文件系统中创建或覆盖一个文件。自动创建父目录。适用于结构化知识、分类内容、长文本记忆。\n\n"
        "⚠️ **Source 锚点契约（S3 硬规则）**：frontmatter 的 `source` 字段必须"
        "含行号（`source_sessions/<sid>.jsonl:<line>`），并带 `session_index` / "
        "`narrative_time` / `ingested_at`。缺行号的写入是历史遗留问题，不要再"
        "制造新的。\n\n"
        "💡 **写入 preferences.md 的建议（S6 方向性指南）**：参考 S6 三类进入"
        "方式（A 类目级+跨 session、B 类目级单声明、C 事件级信号），对单次事件级"
        "叙述**双写**（events.md + preferences.md 的 pending/signals 段），不要"
        "整条丢弃；纯反思 / 纯情绪无类目的句子落 facts.md 即可。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对文件路径（如 'facts/user_profile.md'、'preferences/tech.md'）"},
            "content": {"type": "string", "description": "要写入的文件内容"},
        },
        "required": ["path", "content"],
    }
    is_readonly = False
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.write_file(args["path"], args["content"])


class FsAppendTool(BaseTool):
    name = "fs_append"
    description = (
        "向已存在的文件追加内容（若不存在则创建）。适用于向日志式文件追加新条目。\n\n"
        "⚠️ **Source 锚点契约（S3 硬规则）**：追加的每条条目 frontmatter 或"
        "YAML 块里都要带 `source:<path>:<line>` / `session_index` / "
        "`narrative_time`，不要省略。\n\n"
        "💡 **追加 preferences.md**：参考 S6 三类进入方式；单 session 的类目级"
        "声明加 `single_session_declaration: true` 标签，事件级信号双写到 events.md + "
        "preferences.md 的 pending/signals 段——让下游 retrieve 有锚点可用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对文件路径"},
            "content": {"type": "string", "description": "要追加的内容"},
        },
        "required": ["path", "content"],
    }
    is_readonly = False
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.append_file(args["path"], args["content"])


class FsReadTool(BaseTool):
    name = "fs_read"
    description = (
        "读取文件完整内容。用于在写入前查看已有内容。\n"
        "⚠️ 成本警告：对 >3KB 的积累型文件（如 `people/<slug>/facts.md`），"
        "**优先**用 `fs_execute_bash` + `grep -nE` 拿行号，再 `sed -n 'N,Mp'` 读片段，"
        "不要直接 `fs_read` 全文——否则 input_tokens 会随文件增长 O(n²) 膨胀。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取的相对文件路径"},
        },
        "required": ["path"],
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.read_file(args["path"])


class FsTreeTool(BaseTool):
    name = "fs_tree"
    description = "查看记忆文件系统的目录树结构。"
    parameters = {
        "type": "object",
        "properties": {
            "max_depth": {"type": "integer", "description": "最大显示深度（默认：3）"},
        },
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.tree(int(args.get("max_depth", 3)))


class FsDeleteTool(BaseTool):
    name = "fs_delete"
    description = "从记忆文件系统中删除一个文件。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要删除的相对文件路径"},
        },
        "required": ["path"],
    }
    is_readonly = False
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.delete_file(args["path"])


class FsSearchTool(BaseTool):
    name = "fs_search"
    description = "对所有文件执行 BM25 全文搜索。返回排序后的结果，含文件路径与得分。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询"},
            "top_k": {"type": "integer", "description": "返回结果数（默认：10）"},
        },
        "required": ["query"],
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        results = fs.search_bm25(args["query"], int(args.get("top_k", 10)))
        return str(results)


class FsGrepTool(BaseTool):
    name = "fs_grep"
    description = (
        "结构化正则 / 子串匹配，在指定文件或目录下搜索，返回每条命中的"
        "`{path, line, match, before[], after[]}` JSON 列表。\n\n"
        "**优先使用本工具**：相比 `fs_execute_bash` + `grep -nE + sed -n` 的组合，"
        "本工具一步完成、返回精炼结构化结果，context 占用低，适合：\n"
        "  - 在 `source_sessions/<sid>.jsonl` 里定位用户原话行号\n"
        "  - 在 `people/<slug>/facts.md` 等积累型文件里查已存在主题\n"
        "  - G2 先查后写前的快速 de-dup 检查\n\n"
        "示例：`{\"pattern\":\"(enjoy|love|stopped|used to|hate)\", "
        "\"paths\":\"source_sessions/17_ctx0_s0.jsonl\", \"context_lines\": 1}`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则 / 子串模式"},
            "paths": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "单个相对路径字符串，或路径数组；可以是文件或目录（目录递归）。",
            },
            "context_lines": {"type": "integer", "description": "上下文行数，默认 2"},
            "max_matches": {"type": "integer", "description": "最多命中数，默认 50"},
            "case_insensitive": {"type": "boolean", "description": "大小写不敏感，默认 true"},
            "regex": {"type": "boolean", "description": "是否按正则，默认 true"},
        },
        "required": ["pattern", "paths"],
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        results = fs.grep(
            pattern=args["pattern"],
            paths=args["paths"],
            context_lines=int(args.get("context_lines", 2)),
            max_matches=int(args.get("max_matches", 50)),
            case_insensitive=args.get("case_insensitive", True),
            regex=args.get("regex", True),
        )
        return str(results)


class FsReadLinesTool(BaseTool):
    name = "fs_read_lines"
    description = (
        "按行区间读取文件片段（等价 `sed -n 'start,end p'`），返回带行号的文本。"
        "`fs_grep` 拿到行号后，用本工具读精确片段，避免 `fs_read` 全文。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对文件路径"},
            "start": {"type": "integer", "description": "起始行号（1-indexed，含）"},
            "end": {"type": "integer", "description": "结束行号（含）；省略则只读一行"},
        },
        "required": ["path", "start"],
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.read_lines(
            rel_path=args["path"],
            start=int(args["start"]),
            end=int(args["end"]) if args.get("end") is not None else None,
        )


class FsExecuteBashTool(BaseTool):
    name = "fs_execute_bash"
    description = (
        "在记忆库根目录执行 shell 命令，用于精确定位和轻量写操作。\n\n"
        "**读取场景强烈推荐（降低 input tokens）**：\n"
        "  - `grep -nE 'pattern' people/<slug>/facts.md` → 先拿行号\n"
        "  - `sed -n '78,82p' people/<slug>/facts.md`   → 再读片段\n"
        "  - `wc -l <file>` / `head -n 20 <file>` / `tail -n 30 <file>`\n"
        "  - `git log --oneline -n 20 -- <path>` / `git show <hash>`\n\n"
        "**写入/提交场景（ingest/consolidate 阶段开启）**：\n"
        "  - 结构化写入仍优先 `fs_write` / `fs_append` / `vec_add` / `graph_add_node`；shell 写盘 (`>`) 仅用于边界场景\n"
        "  - **每次完成一批 ingest/consolidate 改动后，请 `git add -A && git commit -m \"<摘要>\"`**\n"
        "    commit message 建议格式：`ingest: +N facts / +M insights — <主题>` 或 `evolve: <动作> — <原因>`\n\n"
        "**允许**：cat/ls/head/tail/wc/sort/uniq/tree/find/grep/awk/sed/cut/tr/diff/jq/"
        "mkdir/touch/mv/cp/rm/chmod/tee/echo/printf/python3/date，以及 `git add`/`git commit`/`git rm`/`git mv`/`git log`/`git show`/`git diff`/`git status`。\n"
        "**禁止**：`..` 跨目录逃逸；git 的 checkout/branch/merge/rebase/switch/worktree/push/pull/fetch/remote/clone。命令超时 15 秒。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "完整 shell 命令（支持 | / && / ||）。"},
            "timeout": {"type": "integer", "description": "超时秒数，默认 15。"},
        },
        "required": ["command"],
    }
    is_readonly = False  # 根据 allow_write 动态控制
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        session_id = deps.get("session_id", "")
        allow_write = deps.get("allow_write", False)
        command = args.get("command", "")
        if not command:
            return "ERROR: missing required arg 'command'"
        env_extra: dict[str, str] = {}
        if session_id:
            env_extra["CURRENT_SESSION_ID"] = session_id
        return fs.execute_bash(
            command=command,
            timeout=args.get("timeout", 15),
            allow_write=allow_write,
            env_extra=env_extra,
        )


class FsListTool(BaseTool):
    name = "fs_list"
    description = "列出指定目录下的文件"
    parameters = {
        "type": "object",
        "properties": {
            "dir": {"type": "string", "description": "目录路径，默认为根目录"},
        },
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return str(fs.list_files(args.get("dir", "")))


class FsGetStructureSummaryTool(BaseTool):
    name = "fs_get_structure_summary"
    description = (
        "获取记忆库的目录结构摘要（一级目录 → 文件数 + 扩展名分布）。\n\n"
        "比 `fs_tree` 更紧凑，适合快速感知记忆库布局，不消耗太多 token。\n"
        "返回形如：\n"
        "  - people/ — 12 files (.md:10, .jsonl:2)\n"
        "  - events/ — 5 files (.md:5)\n"
        "  - index.md (256 bytes)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "max_depth": {"type": "integer", "description": "摘要深度（默认 2）"},
        },
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        return fs.get_structure_summary(max_depth=int(args.get("max_depth", 2)))


class FsReadIndexTool(BaseTool):
    name = "fs_read_index"
    description = (
        "读取记忆库索引文件（.meta/index.md 或 index.md）。\n\n"
        "索引文件是记忆库的目录/导航，包含文件组织结构和内容概览。\n"
        "返回 `(路径, 内容)` 元组的字符串表示。"
    )
    parameters = {
        "type": "object",
        "properties": {},
    }
    is_readonly = True
    category = "fs"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        fs = deps["fs"]
        path, content = fs.read_index()
        return f"[{path}]\n{content}"
