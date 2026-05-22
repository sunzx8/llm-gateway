import logging
import os
import sys

from loguru import logger

from .context import get_request_id
from config.models import LoggingConfig

"""
第三方代码                     logger 模块
─────────────────────────────────────────────────────
logging.getLogger("foo")
    .info("hello")
        │
        ▼
  InterceptHandler.emit()   ← 拦截标准 logging
        │
        ▼
  loguru logger             ← 转发给 loguru
        │
        ▼
  logger.patch(request_id)  ← 自动注入 request_id
        │
        ▼
  输出到文件/控制台 
"""

class InterceptHandler(logging.Handler):
    """Intercept standard logging records and forward them to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Map standard logging level name to loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk up the call stack to find the real caller (skip logging internals)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Inject request_id into every log record automatically via patcher
logger = logger.patch(lambda record: record["extra"].update(
    request_id=get_request_id() or "-"
))

LOGURU_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<4} | "
    "{extra[request_id]} | "
    "{module}.{function}:{line} | {message}"
)


def init_logger(logging_config: LoggingConfig) -> None:
    """Initialize loguru logger with file and/or console sinks."""
    lcfg = logging_config

    # Remove the default loguru sink
    logger.remove()

    # File sink
    if lcfg.file_enabled:
        log_directory = lcfg.file_path
        if not os.path.exists(log_directory):
            os.makedirs(log_directory, exist_ok=True)
            os.chmod(log_directory, 0o777)

        logger.add(
            log_directory + "/llm-gateway-{time:YYYY-MM-DD}.log",
            level=lcfg.level,
            format=LOGURU_FORMAT,
            rotation="24 hour",
            retention="7 days",
            compression="gz",
            encoding="utf-8",
            enqueue=False,
        )

    # Console sink
    if lcfg.console_enabled:
        logger.add(
            sys.stdout,
            level=lcfg.level,
            format=LOGURU_FORMAT,
            colorize=lcfg.console_colored,
        )

    # Intercept all standard logging and forward to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Remove LiteLLM's own StreamHandlers so logs only flow through
    # our InterceptHandler → loguru pipeline (respecting our level config)
    for _name in ("LiteLLM Proxy", "LiteLLM Router", "LiteLLM"):
        _lg = logging.getLogger(_name)
        _lg.handlers.clear()
        _lg.propagate = True
        _lg.setLevel(logging.DEBUG)  # let loguru sink do the level filtering


def info(message, *args, **kwargs):
    """
    记录INFO级别日志

    Args:
        message: 日志消息（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 exc_info, depth）
    """
    exc = kwargs.pop("exc_info", False)
    extra_depth = kwargs.pop("depth", 0)
    if args:
        message = message % args
    logger.opt(depth=1 + extra_depth, exception=exc).info(message)


def error(message, *args, **kwargs):
    """
    记录ERROR级别日志

    Args:
        message: 日志消息（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 exc_info, depth）
    """
    exc = kwargs.pop("exc_info", False)
    extra_depth = kwargs.pop("depth", 0)
    if args:
        message = message % args
    logger.opt(depth=1 + extra_depth, exception=exc).error(message)


def warning(message, *args, **kwargs):
    """
    记录WARNING级别日志

    Args:
        message: 日志消息（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 exc_info, depth）
    """
    exc = kwargs.pop("exc_info", False)
    extra_depth = kwargs.pop("depth", 0)
    if args:
        message = message % args
    logger.opt(depth=1 + extra_depth, exception=exc).warning(message)


def debug(message, *args, **kwargs):
    """
    记录DEBUG级别日志

    Args:
        message: 日志消息（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 exc_info, depth）
    """
    exc = kwargs.pop("exc_info", False)
    extra_depth = kwargs.pop("depth", 0)
    if args:
        message = message % args
    logger.opt(depth=1 + extra_depth, exception=exc).debug(message)


def trace(message, *args, **kwargs):
    """
    记录TRACE级别日志

    Args:
        message: 日志消息（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 exc_info, depth）
    """
    exc = kwargs.pop("exc_info", False)
    extra_depth = kwargs.pop("depth", 0)
    if args:
        message = message % args
    logger.opt(depth=1 + extra_depth, exception=exc).trace(message)


def isEnabledFor(level: int) -> bool:
    """
    判断指定的日志级别是否启用（兼容标准 logging 接口）

    Args:
        level: 标准 logging 级别（如 logging.DEBUG=10, logging.INFO=20）

    Returns:
        bool: 该级别是否启用
    """
    # 将标准 logging 级别映射到 loguru 级别名称
    _level_map = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    level_name = _level_map.get(level, "DEBUG")
    try:
        loguru_level = logger.level(level_name)
        # 检查当前 logger 是否有 handler 能处理该级别
        # loguru 中通过检查 _core.min_level 来判断
        return logger._core.min_level <= loguru_level.no
    except (ValueError, AttributeError):
        return True


def exception(message, *args, **kwargs):
    """
    记录异常日志（自动附带当前异常堆栈）

    Args:
        message: 日志消息或异常对象（支持 % 格式化占位符）
        *args: 格式化参数
        **kwargs: 额外关键字参数（如 depth）
    """
    extra_depth = kwargs.pop("depth", 0)
    if isinstance(message, Exception):
        logger.opt(depth=1 + extra_depth, exception=message).error(str(message), **kwargs)
    else:
        if args:
            message = message % args
        logger.opt(depth=1 + extra_depth, exception=True).error(message, **kwargs)
