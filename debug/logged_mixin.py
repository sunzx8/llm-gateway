"""通用方法日志 Mixin — 自动为子类指定方法注入调用日志。

使用方式：
    1. 让基类继承 LoggedMethodsMixin
    2. 在基类中声明 _logged_methods = {'method1', 'method2', ...}
    3. 所有子类实现这些方法时，调用会自动打印日志（类名、方法名、参数摘要、返回值摘要）

示例::

    class VectorStoreBase(ABC, LoggedMethodsMixin):
        _logged_methods = {'add', 'search', 'delete', ...}

    class PgVectorStore(VectorStoreBase):
        async def add(self, ...):  # 自动带日志，无需手动装饰
            ...
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any

import logger.logger as logger


def _truncate_repr(obj: Any, max_len: int = 200) -> str:
    """安全截断对象的 repr，避免日志过长。"""
    try:
        s = repr(obj)
    except Exception:
        s = f"<{type(obj).__name__}: repr failed>"
    if len(s) > max_len:
        return s[:max_len] + f"...({len(s)} chars)"
    return s


def _get_caller_location() -> str:
    """获取调用者（wrapper 的上一层）的文件名、行号、类名和方法名。"""
    import os
    import sys
    # frame 0 = _get_caller_location, frame 1 = wrapper, frame 2 = 真正的调用者
    frame = sys._getframe(2)
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    func_name = frame.f_code.co_name
    # 尝试从 f_locals 中获取类名
    cls_name = ""
    if "self" in frame.f_locals:
        cls_name = type(frame.f_locals["self"]).__name__
    elif "cls" in frame.f_locals:
        cls_name = getattr(frame.f_locals["cls"], "__name__", "")
    # 取相对路径，避免日志过长
    cwd = os.getcwd()
    if filename.startswith(cwd):
        filename = filename[len(cwd) + 1:]
    # 组装调用者标识：ClassName.method_name (file:line)
    if cls_name:
        caller_id = f"{cls_name}.{func_name}"
    else:
        caller_id = func_name
    return f"{caller_id} ({filename}:{lineno})"


def _logged_method(fn):
    """通用方法装饰器：自动识别 sync/async，打印类名+方法名+参数+耗时+调用者位置。"""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            cls_name = type(self).__name__
            caller = _get_caller_location()
            logger.info(
                "[%s.%s] called by %s | args=%s kwargs=%s",
                cls_name, fn.__name__, caller,
                _truncate_repr(args), _truncate_repr(kwargs),
                depth=1,
            )
            t0 = time.perf_counter()
            try:
                result = await fn(self, *args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "[%s.%s] returned (%.1fms) caller=%s | result=%s",
                    cls_name, fn.__name__, elapsed, caller,
                    _truncate_repr(result),
                    depth=1,
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error(
                    "[%s.%s] raised %s (%.1fms) caller=%s: %s",
                    cls_name, fn.__name__, type(e).__name__, elapsed, caller, e,
                    depth=1,
                )
                raise
        return wrapper
    else:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            cls_name = type(self).__name__
            caller = _get_caller_location()
            logger.info(
                "[%s.%s] called by %s | args=%s kwargs=%s",
                cls_name, fn.__name__, caller,
                _truncate_repr(args), _truncate_repr(kwargs),
                depth=1,
            )
            t0 = time.perf_counter()
            try:
                result = fn(self, *args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "[%s.%s] returned (%.1fms) caller=%s | result=%s",
                    cls_name, fn.__name__, elapsed, caller,
                    _truncate_repr(result),
                    depth=1,
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error(
                    "[%s.%s] raised %s (%.1fms) caller=%s: %s",
                    cls_name, fn.__name__, type(e).__name__, elapsed, caller, e,
                    depth=1,
                )
                raise
        return wrapper


class LoggedMethodsMixin:
    """通用 Mixin：子类继承后，_logged_methods 中指定的方法自动带调用日志。

    工作原理：
        利用 __init_subclass__ 钩子，在子类**定义时**（非实例化时）自动检测
        子类中新定义的方法，如果方法名在 _logged_methods 集合中，则用
        _logged_method 装饰器包装。

    特性：
        - 支持 sync 和 async 方法
        - 打印：类名、方法名、参数摘要、耗时、返回值摘要
        - 异常时打印异常类型和信息，然后 re-raise
        - 不影响 isinstance 判断
        - 不依赖工厂模式，任何地方创建的实例都自动生效
        - 通过 _enable_method_logging 开关可全局关闭

    配置项（子类可覆盖）：
        _logged_methods: set[str]       — 需要被日志包装的方法名集合
        _enable_method_logging: bool    — 是否启用日志（默认 True）
        _log_repr_max_len: int          — 参数/返回值 repr 截断长度（默认 200）
    """

    _logged_methods: set[str] = set()
    _enable_method_logging: bool = True
    _log_repr_max_len: int = 200

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls._enable_method_logging:
            return
        for name in cls._logged_methods:
            # 只装饰本类新定义的方法，避免重复装饰父类已包装的方法
            method = cls.__dict__.get(name)
            if method and callable(method):
                setattr(cls, name, _logged_method(method))
