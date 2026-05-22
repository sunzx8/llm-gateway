"""动态加载 LLM 生成的记忆摄入/消费实现。

核心设计：
- 使用 importlib.util.spec_from_file_location 加载，不污染 sys.path
- 使用 uuid 生成唯一模块名，避免 sys.modules 缓存冲突
- 通过 issubclass 按类型发现子类，不依赖固定类名
- 使用完毕后从 sys.modules 移除，避免内存泄漏
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import uuid
import logging
from pathlib import Path
from typing import Type

from context_task.codegen.base_memory_ingestor import BaseMemoryIngestor
from context_task.codegen.base_memory_consumer import BaseMemoryConsumer

# 用于修正 LLM 生成的错误 import 路径的正则表达式
# 匹配 "from context_task.codegen.<module_name> import ..." 并替换为 "from <module_name> import ..."
_BAD_IMPORT_RE = re.compile(
    r"^from\s+context_task\.codegen\.((?:retrieve|ingest)_base_memory)\s+import\s+",
    re.MULTILINE,
)


def load_ingestor_class(module_path: str | Path) -> Type[BaseMemoryIngestor]:
    """动态加载生成的摄入实现类。

    Args:
        module_path: 生成的 .py 文件绝对路径。

    Returns:
        加载的类（BaseMemoryIngestor 的子类）。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError: 加载失败或未找到子类。
    """
    return _load_subclass(module_path, BaseMemoryIngestor)


def load_consumer_class(module_path: str | Path) -> Type[BaseMemoryConsumer]:
    """动态加载生成的消费实现类。

    Args:
        module_path: 生成的 .py 文件绝对路径。

    Returns:
        加载的类（BaseMemoryConsumer 的子类）。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError: 加载失败或未找到子类。
    """
    return _load_subclass(module_path, BaseMemoryConsumer)


def _load_subclass(module_path: str | Path, base_class: type) -> type:
    """通用的子类加载逻辑。

    关键设计点：
    1. 用 uuid 生成唯一模块名 → 避免 sys.modules 缓存冲突
    2. spec_from_file_location → 不需要修改 sys.path
    3. issubclass 扫描 → 不依赖固定类名，LLM 可自由命名
    4. finally 中清理 sys.modules → 避免内存泄漏和缓存污染
    5. 临时将子类文件所在目录加入 sys.path → 支持子类 import 同目录下的增强基类

    Args:
        module_path: .py 文件绝对路径。
        base_class: 要查找的基类。

    Returns:
        找到的子类。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError: 加载失败或未找到子类。
    """
    _logger = logging.getLogger(__name__)
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"生成的代码文件不存在: {path}")

    # ── 兼容性处理：自动修正 LLM 生成的错误 import 路径 ──
    # 旧版本生成的子类可能包含 "from context_task.codegen.retrieve_base_memory import ..."
    # 正确的写法应该是 "from retrieve_base_memory import ..."（裸模块名）
    source = path.read_text(encoding="utf-8")
    fixed_source = _BAD_IMPORT_RE.sub(r"from \1 import ", source)
    if fixed_source != source:
        _logger.info("loader: 自动修正了 %s 中的 import 路径", path.name)
        path.write_text(fixed_source, encoding="utf-8")

    # 用唯一模块名避免 sys.modules 缓存冲突
    module_name = f"_codegen_{base_class.__name__}_{uuid.uuid4().hex[:8]}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块 spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # 将子类文件所在目录临时加入 sys.path，
    # 以支持子类 import 同目录下的增强基类（如 from retrieve_base_memory import EnhancedMemoryConsumer）
    parent_dir = str(path.parent.resolve())
    path_added = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        path_added = True

    _logger.debug(
        "loader: 加载 %s (module=%s, parent_dir=%s, path_added=%s)",
        path.name, module_name, parent_dir, path_added,
    )

    # 记录加载前的 sys.modules 快照，用于清理因加载子类而新增的模块（如增强基类）
    modules_before = set(sys.modules.keys())

    try:
        spec.loader.exec_module(module)

        # 通过类型发现子类
        # 优先选择非抽象的具体子类；如果全是抽象类则回退选择最深层的子类
        found_class = None
        fallback_class = None
        all_classes_info = []  # 用于调试日志
        for name, obj in inspect.getmembers(module, inspect.isclass):
            is_sub = issubclass(obj, base_class)
            is_base = obj is base_class
            obj_module = getattr(obj, "__module__", None)
            is_abstract = inspect.isabstract(obj)
            all_classes_info.append(
                f"{name}(module={obj_module}, is_sub={is_sub}, "
                f"is_base={is_base}, is_abstract={is_abstract})"
            )
            if (
                is_sub
                and not is_base
                # 排除从其他模块 import 进来的类（如增强基类），只保留本文件中定义的类
                and obj_module == module_name
            ):
                # 检查是否为抽象类（含有未实现的抽象方法）
                if is_abstract:
                    # 记录为回退选项，但继续寻找非抽象的具体子类
                    fallback_class = obj
                else:
                    found_class = obj
                    break

        _logger.info(
            "loader: %s 中发现的类: [%s] → found=%s, fallback=%s",
            path.name,
            ", ".join(all_classes_info),
            found_class.__name__ if found_class else None,
            fallback_class.__name__ if fallback_class else None,
        )

        if found_class is None and fallback_class is not None:
            # 只有抽象类可用（如文件中只 import 了增强基类但没定义具体子类）
            raise ImportError(
                f"在 {path} 中只找到抽象类 {fallback_class.__name__}，"
                f"未找到实现了 retrieve_memory 的具体子类"
            )

        if found_class is None:
            raise ImportError(
                f"在 {path} 中未找到 {base_class.__name__} 的子类"
            )

        return found_class

    finally:
        # 清理 sys.modules：移除本次加载新增的所有模块（子类 + 增强基类等）
        # 避免缓存污染和内存泄漏，也确保增强基类更新后能加载到新版本
        modules_after = set(sys.modules.keys())
        for new_module in modules_after - modules_before:
            # 保留项目自身的稳定模块（如 context_task.*, storage.*, utils.* 等），
            # 只清理动态加载的临时模块（增强基类、子类等用户记忆库中的代码）
            mod = sys.modules.get(new_module)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", None)
            if mod_file and str(Path(mod_file).parent.resolve()) == parent_dir:
                # 该模块来自子类同目录（即 .codegen/ 目录），清理它
                sys.modules.pop(new_module, None)
        # 始终清理子类模块自身
        sys.modules.pop(module_name, None)
        # 恢复 sys.path
        if path_added and parent_dir in sys.path:
            sys.path.remove(parent_dir)
