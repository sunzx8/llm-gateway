#!/usr/bin/env python3
"""
LLM Gateway 启动入口

用法：
    python3 main.py -c config.yaml
    python3 main.py -c config.yaml --host 127.0.0.1 --port 8081
    python3 main.py -c /etc/llm_gateway/prod.yaml --host 0.0.0.0 --port 8000
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import os

# Opik 追踪初始化
try:
    import opik
    from datetime import datetime as _dt
    _opik_project = f"llm_gateway_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    _opik_url = os.environ.get("OPIK_URL", "")
    _opik_use_local = _opik_url != ""  # 有自定义 URL 则视为自部署
    opik.configure(
        api_key=os.environ.get("OPIK_API_KEY", ""),
        workspace=os.environ.get("OPIK_WORKSPACE", "llm-gateway"),
        project_name=_opik_project,
        use_local=_opik_use_local,
        force=True,
        **({"url_override": _opik_url} if _opik_url else {}),
    )

    import logging
    logging.getLogger(__name__).info(f"[Opik] 初始化成功, project={_opik_project}")
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"[Opik] 初始化失败: {e}")

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Gateway 记忆增强模型网关",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="CONFIG_FILE",
        help="配置文件路径，例如：config.yaml",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="监听端口",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开启热重载（仅开发环境使用）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 通过环境变量将配置文件路径传递给 gateway/app.py 的 lifespan
    os.environ["GATEWAY_CONFIG_PATH"] = os.path.abspath(args.config)
    os.environ["GATEWAY_BIND_ADDRESS"] = f"{args.host}:{args.port}"

    uvicorn.run(
        "gateway.app:main_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        workers=1,
        limit_concurrency=200,
        reload_includes=["*.py", "*.yaml"],
    )
