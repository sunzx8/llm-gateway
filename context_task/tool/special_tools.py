"""
特殊工具集

包含流程控制类的特殊工具。
"""

import os
from pathlib import Path
from typing import Any

from .base_tool import BaseTool

# 参考策略文件的根目录（项目内的固定路径）
_REFERENCE_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "prompt" / "reference"


class ReadReferenceStrategyTool(BaseTool):
    """读取参考策略文档。

    底层根据传入的文件路径读取 context_task/prompt/reference 目录下的参考策略文件。
    支持按阶段（ingest/evolve/retrieve）和文件名读取。
    """

    name = "read_reference_strategy"
    description = (
        "读取参考策略文档。可用于查阅已有的优秀策略方案作为参考。\n\n"
        "参数 `path` 为相对于 reference 目录的路径，如：\n"
        "- `ingest/mem0_ingest_strategy.md` — 读取 mem0 摄入策略\n"
        "- `evolve/mem0_evolve_strategy.md` — 读取 mem0 演进策略\n"
        "- `retrieve/mem0_retrieve_strategy.md` — 读取 mem0 消费策略\n\n"
        "也可以只传阶段名（如 `ingest`）来列出该阶段下所有可用的参考文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "参考策略文件的相对路径（相对于 reference 目录），"
                    "如 'ingest/mem0_ingest_strategy.md'；"
                    "或只传阶段名（如 'ingest'、'evolve'、'retrieve'）列出该阶段下所有文件"
                ),
            },
        },
        "required": ["path"],
    }
    is_readonly = True
    category = "special"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        rel_path = args.get("path", "").strip()
        if not rel_path:
            return "ERROR: 参数 path 不能为空"

        target = _REFERENCE_STRATEGY_DIR / rel_path

        # 安全检查：防止路径穿越
        try:
            target.resolve().relative_to(_REFERENCE_STRATEGY_DIR.resolve())
        except ValueError:
            return "ERROR: 路径不合法，不允许访问 reference 目录之外的文件"

        # 如果是目录，列出其中的文件
        if target.is_dir():
            files = sorted(
                f.relative_to(_REFERENCE_STRATEGY_DIR)
                for f in target.rglob("*")
                if f.is_file() and f.suffix == ".md"
            )
            if not files:
                return f"目录 '{rel_path}' 下没有参考策略文件"
            listing = "\n".join(f"- {f}" for f in files)
            return f"目录 '{rel_path}' 下的参考策略文件：\n{listing}"

        # 如果是文件，读取内容
        if target.is_file():
            try:
                content = target.read_text(encoding="utf-8")
                return content
            except Exception as e:
                return f"ERROR: 读取文件失败: {e}"

        return f"ERROR: 文件不存在: {rel_path}"


class FinishTool(BaseTool):
    name = "finish"
    description = (
        "标识任务完成。result 里写本轮存储摘要。\n\n"
        "示例 result：\n"
        "`\"+3 facts (events.md), +2 preferences (preferences.md), "
        "+5 vec entries (facts_alex), +3 graph edges (prefers/tried/avoids). "
        "Topics: cooking, comedy, horror.\"`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "result": {"type": "string", "description": "已存储记忆的结构化摘要"},
        },
    }
    is_readonly = True  # finish 本身不写数据
    category = "special"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """finish 工具通常由 task 层特殊处理，不走通用 execute。"""
        return args.get("result", "")
