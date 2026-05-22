"""路径安全：拦截越界访问。

所有 fs 类工具（ls / read_file / write_file / edit_file / grep）在拼接磁盘
路径前必须用 ``safe_resolve`` 校验。

协议：
- LLM 应当传入**绝对路径**（如 ``/tmp/llm_gateway/alex/memory_repo/people/alex/profile.md``）
- 解析后必须位于记忆库根目录之内（含根本身），否则拒绝
- 跟随符号链接后越界也会被拒绝（双方都用 ``realpath`` 展开后对比）

返回值是**相对于 base 的相对路径**，工具层把它传给 ``FileSystemStore.read_file/write_file/grep`` 等使用相对路径 API 的底层方法。

拒绝示例：
- ``"/etc/passwd"`` —— 越界
- ``"/tmp/llm_gateway/alex/memory_repo/../../etc/passwd"`` —— 越界

允许示例：
- 记忆库根目录本身
- 任何最终解析后位于根目录之内的路径
"""

from __future__ import annotations

import os


def safe_resolve(base: str, path: str) -> tuple[bool, str]:
    """把 ``path`` 解析为相对于 ``base`` 的相对路径，并校验未越出 ``base``。

    Args:
        base: 记忆库根目录（绝对路径）。
        path: 工具调用方传入的路径，期望是绝对路径。

    Returns:
        ``(ok, rel_or_err)``：
        - ok=True 时，``rel_or_err`` 是相对于 base 的相对路径（根目录返回 ""）
        - ok=False 时，``rel_or_err`` 是给 LLM 看的中文错误信息
    """
    if path is None or path == "":
        return False, "路径不能为空"

    base_abs = os.path.realpath(base)

    # 绝对路径直接 realpath；相对路径先拼到 base 上再 realpath（容错）
    if os.path.isabs(path):
        full = os.path.realpath(path)
    else:
        full = os.path.realpath(os.path.join(base_abs, path))

    # full 必须等于 base_abs 自身，或位于 base_abs 之下
    if full != base_abs and not full.startswith(base_abs + os.sep):
        return False, f"路径越界：{path}（必须位于记忆库根目录 {base_abs} 之内）"

    rel = os.path.relpath(full, base_abs)
    if rel == ".":
        rel = ""
    return True, rel
