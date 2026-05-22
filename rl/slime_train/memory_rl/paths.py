"""寻找 llm_gateway 根目录并把它的父目录加入 ``sys.path``。

迁移前老逻辑寻找 ``src/agent_memory``；现在寻找 ``llm_gateway`` 根（含
``gateway/`` + ``storage/`` 即认作根），然后把根的父目录加到 sys.path，使得
``import llm_gateway`` 可用。

入口仍叫 ``ensure_workspace_paths`` 保持兼容。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _looks_like_llm_gateway_root(p: Path) -> bool:
    """``p/gateway`` + ``p/storage`` 都存在就视作 llm_gateway 根目录。"""
    return (
        p.is_dir()
        and (p / "gateway").is_dir()
        and (p / "storage").is_dir()
        and p.name == "llm_gateway"
    )


def find_llm_gateway_root(start: str | None = None) -> Path:
    """向上查找 ``llm_gateway`` 根目录。"""
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if _looks_like_llm_gateway_root(parent):
            return parent
    # 兜底：从 __file__ 倒推：本文件位于 llm_gateway/rl/slime_train/memory_rl/paths.py
    return Path(__file__).resolve().parents[3]


# 保留老名字，便于 reward / convert_to_slime_format 等调用方零修改
find_workspace_root = find_llm_gateway_root


def ensure_workspace_paths(start: str | None = None) -> Path:
    """把 llm_gateway 根和父目录加入 ``sys.path``。

    迁移期代码同时存在两类 import：
    - ``import llm_gateway...`` 需要 llm_gateway 的父目录；
    - ``import storage/utils/context_task...`` 需要 llm_gateway 根目录。

    返回 llm_gateway 根目录的 :class:`Path`。
    """
    root = find_llm_gateway_root(start)
    for path in (str(root.parent), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


__all__ = [
    "find_llm_gateway_root",
    "find_workspace_root",
    "ensure_workspace_paths",
]
