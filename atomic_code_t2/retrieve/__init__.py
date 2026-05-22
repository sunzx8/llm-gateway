"""Atomic_Code_T2 检索子包（项目侧入口）。

架构：
- **项目侧（本目录）**：
  - `base_plan.py`：稳定接口 `BasePlan`（不复制，所有 plan 文件通过绝对 import 引用）
  - `loader.py`：把用户副本目录当作 Python 包动态加载
  - `templates/`：将被整目录复制到用户记忆库的模板包
      - `atomic.py`：所有原子操作
      - `prompt.py`：query 改写 prompt
      - `plans/default_plan.py`：默认方案（用原子操作拼起来）

- **用户侧（运行时）**：
  - `<fs.base_path>/.codegen/retrieve/`：每个用户独立一份完整副本，可自由演进。
    LLM 未来可以在此目录修改/新增原子操作，新增方案文件，互不影响其他用户。
"""

from .base_plan import BasePlan
from .loader import invalidate_plan_cache, load_plan_class

__all__ = ["BasePlan", "load_plan_class", "invalidate_plan_cache"]
