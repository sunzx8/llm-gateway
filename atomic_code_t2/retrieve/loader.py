"""动态加载用户记忆库下的 retrieve plan 包。

设计：
- 每个用户的 retrieve 方案集合是一个**完整的 Python 包**，挂在
  `<fs.base_path>/.codegen/retrieve/` 下：
      __init__.py
      atomic.py
      prompt.py
      plans/
          __init__.py
          default_plan.py
- 包里的代码用**包内相对 import**（`from ..atomic import ...`），所以同一份代码
  既能在用户副本里跑，也能被项目代码静态分析。
- 加载方式：临时把 `<fs.base_path>/.codegen/` 加入 `sys.path`，
  以 `retrieve.plans.<plan_name>` 的形式 import 进来，找出 BasePlan 子类返回。
- 用唯一的 root 包名（带 user_hash 后缀）避免不同用户副本之间在 sys.modules 缓存冲突。

注意：加载完不会立刻清理 sys.modules——因为同一进程内同用户可能多次 retrieve，
缓存反而能加速。如果用户副本文件被外部改了，需要主动调 `invalidate_plan_cache()`。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Type

import logger.logger as logger

from .base_plan import BasePlan


def _make_root_pkg_name(retrieve_dir: Path) -> str:
    """给一个用户副本目录生成一个稳定且唯一的根包名。

    用目录绝对路径做 hash，保证：
    - 同一个目录每次加载得到同样的根包名 → sys.modules 复用、缓存有效
    - 不同用户的目录 hash 不同 → 不会互相覆盖
    """
    h = hashlib.md5(str(retrieve_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"_atomic_t2_user_{h}"


def load_plan_class(
    retrieve_dir: str | Path,
    plan_module: str = "plans.default_plan",
) -> Type[BasePlan]:
    """从用户副本目录加载一个 BasePlan 子类。

    Args:
        retrieve_dir: 用户副本目录的绝对路径
            （形如 `<fs.base_path>/.codegen/retrieve`）。
        plan_module: 要加载的 plan 模块路径，相对于 `retrieve_dir`，**点号分隔**，
            不带 `.py` 后缀。例如：
            - "plans.default_plan"
            - "plans.hybrid_v2"

    Returns:
        加载到的 BasePlan 子类（未实例化）。

    Raises:
        FileNotFoundError: 目录不存在或 plan 模块文件不存在。
        ImportError: 加载失败或模块里没找到 BasePlan 子类。
    """
    retrieve_path = Path(retrieve_dir).resolve()
    if not retrieve_path.is_dir():
        raise FileNotFoundError(f"retrieve 副本目录不存在: {retrieve_path}")

    # plan 文件存在性校验（提前给出友好错误）
    rel_file = Path(*plan_module.split(".")).with_suffix(".py")
    plan_file = retrieve_path / rel_file
    if not plan_file.is_file():
        raise FileNotFoundError(
            f"plan 模块文件不存在: {plan_file} "
            f"(retrieve_dir={retrieve_path}, plan_module={plan_module!r})"
        )

    # 我们要 import 的全路径形如 `<root_pkg>.plans.default_plan`
    # `<root_pkg>` 对应 `retrieve_path` 本身（即用户副本目录被视为一个包）
    root_pkg = _make_root_pkg_name(retrieve_path)
    full_module_name = f"{root_pkg}.{plan_module}"

    # 把"用户副本目录的父目录"加入 sys.path，并把 retrieve_path 注册为 root_pkg 包
    parent_dir = str(retrieve_path.parent)
    parent_added = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        parent_added = True

    try:
        # 关键：把 retrieve_path 目录注册为 `<root_pkg>` 包
        # （retrieve_path 下应该有 __init__.py，否则下面 import 会失败）
        if root_pkg not in sys.modules:
            init_py = retrieve_path / "__init__.py"
            if not init_py.is_file():
                raise ImportError(
                    f"用户副本目录缺少 __init__.py: {init_py}（"
                    f"无法被识别为 Python 包）"
                )
            spec = importlib.util.spec_from_file_location(
                root_pkg,
                init_py,
                submodule_search_locations=[str(retrieve_path)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法为用户副本目录创建 spec: {retrieve_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[root_pkg] = module
            spec.loader.exec_module(module)

        # 现在可以走标准 import 流程拉到子模块
        plan_mod = importlib.import_module(full_module_name)

        # 在模块里找 BasePlan 的子类
        found_class: type | None = None
        for _, obj in inspect.getmembers(plan_mod, inspect.isclass):
            if (
                issubclass(obj, BasePlan)
                and obj is not BasePlan
                # 只认在本 plan 模块内定义的类（避免抓到 import 进来的别的 BasePlan 子类）
                and obj.__module__ == full_module_name
            ):
                found_class = obj
                break

        if found_class is None:
            raise ImportError(
                f"在 {plan_file} 中未找到 BasePlan 的子类"
            )

        logger.debug(
            "[retrieve.loader] loaded %s.%s from %s",
            full_module_name, found_class.__name__, retrieve_path,
        )
        return found_class

    finally:
        if parent_added:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass


def invalidate_plan_cache(retrieve_dir: str | Path) -> int:
    """清理某用户副本目录在 sys.modules 中的缓存。

    用户副本文件被外部修改后调用，使下一次 `load_plan_class` 重新加载。

    Returns:
        清理掉的 sys.modules 条目数。
    """
    retrieve_path = Path(retrieve_dir).resolve()
    root_pkg = _make_root_pkg_name(retrieve_path)
    keys_to_drop = [k for k in sys.modules if k == root_pkg or k.startswith(root_pkg + ".")]
    for k in keys_to_drop:
        sys.modules.pop(k, None)
    logger.debug(
        "[retrieve.loader] invalidated %d cached modules under %s",
        len(keys_to_drop), root_pkg,
    )
    return len(keys_to_drop)
