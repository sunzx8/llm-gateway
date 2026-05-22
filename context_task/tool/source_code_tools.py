"""
源代码读取工具集

提供读取本仓库内 FileSystemStore / VectorStoreBase / GraphStoreBase 类源代码的工具，
供代码生成策略和代码生成阶段使用，帮助 LLM 了解存储后端的完整接口实现。
"""

import inspect
from pathlib import Path
from typing import Any

from .base_tool import BaseTool

# 源代码文件的根目录
_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"


class ReadFileSystemStoreSourceTool(BaseTool):
    """读取 FileSystemStore 类的源代码。

    返回 storage/file_system_store.py 中 FileSystemStore 类的完整源代码，
    帮助 LLM 了解文件系统存储后端的所有可用方法和实现细节。
    """

    name = "read_fs_store_source"
    description = (
        "读取 FileSystemStore 类的完整源代码。\n\n"
        "返回 storage/file_system_store.py 文件的内容，包含 FileSystemStore 类的所有方法实现。\n"
        "用于了解文件系统存储后端的接口细节，帮助生成正确的消费/摄入代码。\n\n"
        "无需参数，直接调用即可。可选传入 methods_only=true 只返回方法签名列表。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "methods_only": {
                "type": "boolean",
                "description": (
                    "是否只返回方法签名列表（不含实现体）。"
                    "默认 false，返回完整源代码。"
                ),
            },
        },
        "required": [],
    }
    is_readonly = True
    category = "special"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        methods_only = args.get("methods_only", False)
        source_file = _STORAGE_DIR / "file_system_store.py"

        if not source_file.exists():
            return "ERROR: file_system_store.py 文件不存在"

        try:
            content = source_file.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: 读取文件失败: {e}"

        if methods_only:
            return self._extract_methods(content, "FileSystemStore")

        return content


    def _extract_methods(self, content: str, class_name: str) -> str:
        """从源代码中提取指定类的方法签名列表。"""
        import re
        lines = content.split("\n")
        in_class = False
        methods = []
        class_indent = 0

        for i, line in enumerate(lines):
            # 检测类定义开始
            if re.match(rf"^class {class_name}\b", line):
                in_class = True
                class_indent = len(line) - len(line.lstrip())
                continue

            if in_class:
                # 检测类结束（遇到同级或更高级的定义）
                stripped = line.lstrip()
                if stripped and not line.startswith(" " * (class_indent + 1)):
                    if re.match(r"^(class |def )", stripped):
                        break

                # 提取方法签名
                match = re.match(r"^(\s+)(async\s+)?def\s+(\w+)\s*\((.*)$", line)
                if match:
                    indent = match.group(1)
                    async_prefix = match.group(2) or ""
                    method_name = match.group(3)
                    # 收集完整签名（可能跨多行）
                    sig_lines = [line.rstrip()]
                    j = i + 1
                    while j < len(lines) and ")" not in sig_lines[-1]:
                        sig_lines.append(lines[j].rstrip())
                        j += 1
                    # 提取 docstring（如果有）
                    docstring = ""
                    if j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            docstring = next_line.strip('"').strip("'").strip()
                            if not docstring and j + 1 < len(lines):
                                docstring = lines[j + 1].strip()

                    sig = "\n".join(sig_lines)
                    if docstring:
                        methods.append(f"{sig}\n        \"\"\"{docstring}\"\"\"")
                    else:
                        methods.append(sig)

        if not methods:
            return f"未找到类 {class_name} 的方法定义"

        header = f"# {class_name} 方法签名列表\n\n"
        return header + "\n\n".join(methods)


class ReadVectorStoreBaseSourceTool(BaseTool):
    """读取 VectorStoreBase 类的源代码。

    返回 storage/stores_base.py 中 VectorStoreBase 抽象基类的完整定义，
    帮助 LLM 了解向量存储后端的接口规范。
    """

    name = "read_vec_store_source"
    description = (
        "读取 VectorStoreBase 抽象基类的完整源代码。\n\n"
        "返回 storage/stores_base.py 文件的内容，包含 VectorStoreBase 及相关 Mixin 的定义。\n"
        "用于了解向量存储后端的接口规范（方法签名、参数、返回值格式），帮助生成正确的消费/摄入代码。\n\n"
        "无需参数，直接调用即可。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    is_readonly = True
    category = "special"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        source_file = _STORAGE_DIR / "stores_base.py"

        if not source_file.exists():
            return "ERROR: stores_base.py 文件不存在"

        try:
            content = source_file.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: 读取文件失败: {e}"

        return content


class ReadGraphStoreBaseSourceTool(BaseTool):
    """读取 GraphStoreBase 类的源代码。

    返回 storage/graph_stores.py 中 GraphStoreBase 的实现代码，
    帮助 LLM 了解图存储后端的完整接口和实现细节。
    """

    name = "read_graph_store_source"
    description = (
        "读取图存储后端的完整源代码。\n\n"
        "返回 storage/graph_stores.py 文件的内容，包含 GraphStoreBase 的具体实现。\n"
        "用于了解图存储后端的接口细节（节点/边操作、搜索、子图获取等），帮助生成正确的消费/摄入代码。\n\n"
        "无需参数，直接调用即可。可选传入 methods_only=true 只返回方法签名列表。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "methods_only": {
                "type": "boolean",
                "description": (
                    "是否只返回方法签名列表（不含实现体）。"
                    "默认 false，返回完整源代码。"
                ),
            },
        },
        "required": [],
    }
    is_readonly = True
    category = "special"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        methods_only = args.get("methods_only", False)
        source_file = _STORAGE_DIR / "graph_stores.py"

        if not source_file.exists():
            return "ERROR: graph_stores.py 文件不存在"

        try:
            content = source_file.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: 读取文件失败: {e}"

        if methods_only:
            return self._extract_class_methods(content)

        return content

    def _extract_class_methods(self, content: str) -> str:
        """从源代码中提取所有类的方法签名。"""
        import re
        lines = content.split("\n")
        result_parts = []
        current_class = None

        for i, line in enumerate(lines):
            # 检测类定义
            class_match = re.match(r"^class (\w+)\b", line)
            if class_match:
                current_class = class_match.group(1)
                result_parts.append(f"\n# === {current_class} ===\n")
                continue

            # 提取方法签名
            if current_class:
                match = re.match(r"^(\s+)(async\s+)?def\s+(\w+)\s*\((.*)$", line)
                if match:
                    # 收集完整签名（可能跨多行）
                    sig_lines = [line.rstrip()]
                    j = i + 1
                    while j < len(lines) and "):" not in sig_lines[-1] and ") ->" not in sig_lines[-1]:
                        sig_lines.append(lines[j].rstrip())
                        j += 1
                    sig = "\n".join(sig_lines)
                    result_parts.append(sig)

        if not result_parts:
            return "未找到类方法定义"

        return "\n\n".join(result_parts)
