"""
Gateway App - FastAPI 主应用

启动方式：
    python3 main.py -c config.yaml --host 127.0.0.1 --port 8000

用户 base_url：
    http://your-server:8000/llm/v1

查询可用模型：
    GET http://your-server:8000/llm/v1/models

自定义钩子：
    继承 GatewayCallbacks，重写 on_request / on_response，
    在 lifespan 中替换 DefaultLoggingCallbacks 即可。
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import importlib
import importlib.metadata

import litellm
import yaml
from fastapi import FastAPI

from gateway.callbacks import _GatewayCallbacks
from config.loader import  load_config
from config.models import AppConfig
from gateway.request_id import RequestIdMiddleware
from interceptor.handler import CallbackHandler, DefaultCallbackHandler
from context_task.ingest_scheduler import IngestScheduler
from logger.logger import init_logger
import logger.logger as logger

# ---------------------------------------------------------------------------
# Callback Handler 加载
# ---------------------------------------------------------------------------

def _load_callback_handler(clazz:str) -> CallbackHandler:
    """
    clazz：完整类路径，例如 mypackage.mymodule.MyHandler
    - 若未设置，使用内置的 DefaultCallbackHandler。
    - 若类不存在或不是 CallbackHandler 的子类，抛出友好异常。
    """
    class_path = clazz.strip()

    if not class_path:
        logger.info("LLM_HOOK_HANDLER not set, using DefaultCallbackHandler.")
        return DefaultCallbackHandler()

    # 拆分模块路径和类名
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path '{class_path}'. "
            "Expected format: 'module.path.ClassName', e.g. 'mypackage.mymodule.MyHandler'."
        )

    module_path, cls_name = class_path.rsplit(".", 1)

    # 尝试 import 模块
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Cannot import module '{module_path}': {e}. "
            "Please check that the module is installed and the path is correct."
        ) from e

    # 尝试获取类
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{cls_name}' not found in module '{module_path}'. "
            "Please check the class name spelling."
        )

    # 校验是否为 CallbackHandler 的子类
    if not (isinstance(cls, type) and issubclass(cls, CallbackHandler) and cls is not CallbackHandler):
        raise TypeError(
            f"'{class_path}' is not a valid subclass of "
            "'intercepter.handler.CallbackHandler'. "
            "Please make sure your class inherits from CallbackHandler and implements "
            "on_request() and on_response()."
        )

    logger.info("Using custom CallbackHandler: %s", class_path)
    return cls()


# ---------------------------------------------------------------------------
# 配置加载工具函数
# ---------------------------------------------------------------------------

def _write_litellm_tmp_config(litellm_config: dict) -> str:
    """
    将 litellm 配置写入带随机数的临时文件，返回临时文件路径。

    文件写入项目根目录（而非系统 /tmp），因为 litellm 的
    ``get_instance_fn`` 会基于配置文件所在目录解析
    ``custom_provider_map`` 中 ``custom_handler`` 的相对模块路径。
    使用 uuid4 保证多进程同时启动时不会互相覆盖。
    """
    project_root = str(Path(__file__).resolve().parent.parent)
    tmp_path = os.path.join(project_root, f".litellm_cfg_{uuid.uuid4().hex}.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(litellm_config, f, allow_unicode=True)
    return tmp_path


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期管理"""

    tmp_config_path: str | None = None
    ingest_scheduler = None

    try:
        # ---- 1. 读取配置文件 ----
        config_path = os.environ.get("GATEWAY_CONFIG_PATH", "")
        if not config_path:
            logger.warning("GATEWAY_CONFIG_PATH not set, skipping config load.")
            app_config: AppConfig = AppConfig()
        else:
            app_config:AppConfig = load_config(config_path)
            logger.info("Config loaded from: %s", config_path)

        gateway_config = app_config.llm_gateway
        litellm_config = app_config.litellm

        # ---- 2. 应用 gateway 自身配置 ----
        init_logger(gateway_config.logging)

        # ---- 3. 注册 Callbacks ----
        # 若未设置，则使用内置的 DefaultCallbackHandler
        callbackhandler = _load_callback_handler(gateway_config.llm_hook_handler)
        callbacks = _GatewayCallbacks(callbackhandler)
        litellm.callbacks = [callbacks]
        litellm.success_callback = ["opik"]
        litellm.failure_callback = ["opik"]
        logger.info("Callbacks registered: %s + litellm opik", type(callbacks).__name__)

        # ---- 4. 将 litellm 配置写入临时文件并加载 ----
        if litellm_config:
            tmp_config_path = _write_litellm_tmp_config(litellm_config)
            logger.debug("LiteLLM tmp config written to: %s", tmp_config_path)
            try:
                import litellm.proxy.proxy_server as proxy_server
                from litellm.proxy.proxy_server import ProxyConfig

                proxy_config = ProxyConfig()
                # load_config returns (router, model_list, general_settings)
                result = await proxy_config.load_config(
                    router=proxy_server.llm_router,
                    config_file_path=tmp_config_path,
                )
                router, model_list, general_settings = result
                # Write back to proxy_server globals so /v1/models etc. can read them
                proxy_server.llm_router = router
                proxy_server.llm_model_list = model_list
                proxy_server.general_settings = general_settings or {}

                logger.info(
                    "LiteLLM config loaded successfully. Models: %s",
                    [m.get("model_name") for m in (model_list or [])],
                )
            except Exception as e:
                logger.warning("Failed to load LiteLLM config: %s", e, exc_info=True)
        else:
            logger.warning("No 'litellm' section in config, LiteLLM using defaults.")

        _litellm_version = importlib.metadata.version("litellm")
        logger.info("LLM Gateway started. LiteLLM version: %s", _litellm_version)
        _gateway_bind_address = os.environ.get("GATEWAY_BIND_ADDRESS", "0.0.0.0:8000")
        logger.info("User base_url: http://%s/llm/v1", _gateway_bind_address)

        # ---- 5. 启动定时记忆摄入调度器 ----
        ctx_task_cfg = app_config.memory_config.context_task
        if ctx_task_cfg.scheduled_ingest_enabled:
            ingest_scheduler = IngestScheduler(
                memory_config_dict=app_config.memory_config.model_dump(),
                interval=ctx_task_cfg.scheduled_ingest_interval_seconds,
                max_workers=ctx_task_cfg.scheduled_ingest_max_workers,
            )
            ingest_scheduler.start()
            logger.info("定时记忆摄入调度器已启动")

    finally:
        # ---- 6. 清理临时文件（无论启动是否成功）----
        if tmp_config_path and os.path.exists(tmp_config_path):
            try:
                os.unlink(tmp_config_path)
                logger.debug("LiteLLM tmp config removed: %s", tmp_config_path)
            except OSError:
                pass

    yield

    # ---- 7. 关闭定时记忆摄入调度器 ----
    if ingest_scheduler is not None:
        logger.info("正在关闭定时记忆摄入调度器...")
        ingest_scheduler.shutdown(timeout=30.0)

    logger.info("LLM Gateway shutting down.")


# ---------------------------------------------------------------------------
# 主应用
# ---------------------------------------------------------------------------

main_app = FastAPI(
    title="LLM Gateway",
    description="基于 LiteLLM 的模型代理网关，支持多模型路由、负载均衡和自定义钩子",
    version="0.1.0",
    lifespan=lifespan,
)

# ---- Request ID Middleware ----
# Extracts request_id from request header/body and injects into response header
main_app.add_middleware(RequestIdMiddleware)


# ---- 业务路由 ----

@main_app.get("/health", tags=["Gateway"])
def health():
    """健康检查"""
    return {"status": "ok", "version": "0.1.0"}


@main_app.get("/info", tags=["Gateway"])
def info():
    """查看网关信息"""
    return {
        "litellm_version": importlib.metadata.version("litellm"),
        "llm_base_url": "/llm/v1",
        "models_endpoint": "/llm/v1/models",
        "chat_endpoint": "/llm/v1/chat/completions",
    }


# ---- 挂载 LiteLLM Proxy ----

def _mount_litellm():
    """延迟挂载，避免 import 时触发 LiteLLM 初始化"""
    try:
        from litellm.proxy.proxy_server import app as litellm_proxy_app
        main_app.mount("/llm", litellm_proxy_app)
        logger.info("LiteLLM proxy mounted at /llm")
    except Exception as e:
        logger.error("Failed to mount LiteLLM proxy: %s", e)
        raise


_mount_litellm()
