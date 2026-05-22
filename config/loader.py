"""Configuration loader (thread-safe singleton).

Reads a YAML config file and returns a validated ``AppConfig`` instance.
The loaded config is cached as a module-level singleton so that all callers
share the same instance.  A ``threading.Lock`` guards the initialisation to
prevent races when multiple threads call ``load_config`` / ``get_config``
concurrently.
"""

from __future__ import annotations

import threading
from pathlib import Path
import os

import yaml

from config.models import AppConfig
import logger.logger as logger

# ---- singleton state (guarded by _lock) ----
_lock = threading.Lock()
_config: AppConfig | None = None

_ENV_PREFIX = "os.environ/"


def _resolve_env_vars(obj):
    """递归遍历配置数据，将 'os.environ/VAR_NAME' 格式的值替换为对应环境变量的值。

    支持任意嵌套深度的 dict 和 list 结构。

    Args:
        obj: 待处理的配置数据（可以是 dict、list、str 或其他类型）。

    Returns:
        替换环境变量后的配置数据。

    Raises:
        ValueError: 当环境变量未设置时抛出异常。
    """
    if isinstance(obj, str):
        if obj.startswith(_ENV_PREFIX):
            env_key = obj[len(_ENV_PREFIX):]
            value = os.environ.get(env_key)
            if value is None:
                raise ValueError(
                    f"环境变量 '{env_key}' 未设置，"
                    f"请确保在运行前已导出该环境变量。"
                )
            return value
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    else:
        return obj


def load_config(config_path: str) -> AppConfig:
    """Load, validate and cache configuration from a YAML file.

    If the singleton has already been initialised, the cached instance is
    returned immediately without re-reading the file.  Use ``reload_config``
    to force a refresh.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The singleton ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config content fails validation.
    """
    global _config

    # Fast path – already loaded (double-checked locking).
    if _config is not None:
        return _config

    with _lock:
        # Re-check after acquiring the lock.
        if _config is not None:
            return _config

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw = _resolve_env_vars(raw)
        _config = AppConfig.model_validate(raw)
        logger.info("Configuration loaded and validated from: %s", config_path)
        return _config


def get_config() -> AppConfig:
    """Return the singleton ``AppConfig`` instance.

    This is the preferred way to access the configuration after the
    application has been initialised via ``load_config``.

    Returns:
        The cached ``AppConfig`` instance.

    Raises:
        RuntimeError: If ``load_config`` has not been called yet.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration has not been loaded yet. "
            "Call load_config(config_path) first."
        )
    return _config


def reload_config(config_path: str) -> AppConfig:
    """Force re-read and re-validate the configuration file.

    This replaces the cached singleton with a freshly loaded instance.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The newly loaded ``AppConfig`` instance.
    """
    global _config

    with _lock:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw = _resolve_env_vars(raw)
        _config = AppConfig.model_validate(raw)
        logger.info("Configuration reloaded from: %s", config_path)
        return _config
