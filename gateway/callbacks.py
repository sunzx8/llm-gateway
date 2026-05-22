"""
Gateway Callbacks - LLM 请求前/响应后的钩子

使用方式：
    继承 GatewayCallbacks，重写 on_request 和 on_response 方法，
    在 app.py 的 lifespan 中注册你的实现即可。

示例：
    class MyCallbacks(GatewayCallbacks):
        async def on_request(self, context: RequestContext) -> RequestContext:
            # 在这里做：鉴权、限流、修改请求参数、注入 system prompt 等
            print(f"收到请求: model={context.model}, user={context.user_id}")
            return context

        async def on_response(self, context: ResponseContext) -> None:
            # 在这里做：计费、审计日志、监控上报等
            print(f"响应完成: tokens={context.total_tokens}, cost=${context.cost:.6f}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger
import json

from debug import dump_tree
from interceptor.handler import CallbackHandler, RequestContext, ResponseContext
from logger.context import get_request_id, get_user_id, get_session_id, set_task_results, append_task_results
import logger.logger as logger



# ---------------------------------------------------------------------------
# 基类：GatewayCallbacks
# ---------------------------------------------------------------------------

class _GatewayCallbacks(CustomLogger):
    """
    LLM Gateway 钩子基类。

    继承此类并重写 on_request / on_response 方法，
    然后在 app.py 的 lifespan 中注册：
        litellm.callbacks = [MyCallbacks()]
    """

    def __init__(self,callbackHandler: CallbackHandler):
        self.callbackHandler = callbackHandler

    # -----------------------------------------------------------------------
    # 以下是 LiteLLM CustomLogger 的内部实现，将 LiteLLM 事件桥接到上面的钩子
    # 子类通常不需要重写这些方法
    # -----------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> dict:
        """LiteLLM 内部钩子：请求转发前，桥接到 on_request"""
        metadata = data.get("metadata", {})
        try:
            # 构建 RequestContext
            logger.info(f"async_pre_call_hook:\n{dump_tree(data,max_depth=2)}")
            # 从 metadata 中提取 user_id（由客户端在请求体 metadata 字段中传递）
            user_id = metadata.get("user_id") or get_user_id() or None
            if not user_id:
                raise ValueError("user_id must be set in request body metadata")
            
            request_id = metadata.get("headers", {}).get("x-request-id", "") or get_request_id()
            session_id = metadata.get("session_id") or get_session_id() or None
            context = RequestContext(
                model=data.get("model", ""),
                messages=data.get("messages", []),
                user_id=user_id,
                api_key_prefix="",
                request_id=request_id,
                session_id=session_id,
                raw_data=data,
            )

            # 调用子类实现的 on_request
            context = await self.callbackHandler.on_request(context)

            # 将修改同步回 data（支持修改 model 和 messages）
            data["model"] = context.model
            data["messages"] = context.messages

            # 将 on_request 中产生的 task_results 保存到 metadata，供后续响应时使用
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["_task_results"] = context.task_results

            # 同时存入 contextvars，供中间件注入到响应 JSON 中
            # 注意：必须使用 append 而非 set，因为 Starlette BaseHTTPMiddleware
            # 的 call_next 在子 task 中执行，set 操作只影响子 task 的 context 副本，
            # 而 append 修改的是中间件初始化时创建的同一个列表对象，父 task 能看到变化。
            append_task_results(context.task_results)

        except Exception as e:
            logger.error(f"on_request hook error: {e}", exc_info=True)

        return data

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """LiteLLM 内部钩子：请求成功，桥接到 on_response"""
        logger.info(f"async_log_success_event kwargs:\n{dump_tree(kwargs,max_depth=2)}")
        logger.info(f"async_log_success_event response_obj:\n{dump_tree(response_obj,max_depth=2)}")
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        user_id = metadata.get("user_id") or get_user_id() or None
        if not user_id:
            raise ValueError("user_id must be set in request body metadata")
        request_id = metadata.get("headers", {}).get("x-request-id", "") or get_request_id()
        try:
            usage = getattr(response_obj, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            # 估算费用

            # 提取响应文本
            try:
                response_content = response_obj.choices[0].message.content
            except Exception:
                response_content = None

            # 从 metadata 中恢复 on_request 阶段的 task_results
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            request_task_results = metadata.get("_task_results", [])

            context = ResponseContext(
                model=kwargs.get("model", ""),
                user_id=user_id,
                request_id=request_id,
                messages=kwargs.get("messages", []),
                response_content=response_content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=0.0,
                duration_seconds=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                success=True,
            )

            context = await self.callbackHandler.on_response(context)

            # 合并 on_request 和 on_response 阶段的 task_results，注入到响应对象中
            all_task_results = request_task_results + (context.task_results or [])
            if all_task_results:
                response_obj._hidden_params["task_results"] = all_task_results
                # 将 on_response 阶段新产生的 task_results 追加到 contextvars
                # （on_request 阶段的已在 async_pre_call_hook 中 append 过，避免重复）
                if context.task_results:
                    append_task_results(context.task_results)

        except Exception as e:
            logger.error(f"on_response hook error: {e}", exc_info=True)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """LiteLLM 内部钩子：请求失败，也桥接到 on_response（success=False）"""
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        user_id = metadata.get("user_id") or get_user_id() or None
        if not user_id:
            raise ValueError("user_id must be set in request body metadata")
        try:
            request_id = metadata.get("headers", {}).get("x-request-id", "") or get_request_id()
            # 从 metadata 中恢复 on_request 阶段的 task_results
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            request_task_results = metadata.get("_task_results", [])

            context = ResponseContext(
                model=kwargs.get("model", ""),
                request_id=request_id,
                user_id=user_id,
                messages=kwargs.get("messages", []),
                response_content=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0,
                duration_seconds=(end_time - start_time).total_seconds(),
                start_time=start_time,
                end_time=end_time,
                success=False,
                error=kwargs.get("exception"),
            )

            context = await self.callbackHandler.on_response(context)

            # 合并 on_request 和 on_response 阶段的 task_results
            all_task_results = request_task_results + (context.task_results or [])
            if all_task_results and response_obj:
                try:
                    response_obj._hidden_params["task_results"] = all_task_results
                    # 将 on_response 阶段新产生的 task_results 追加到 contextvars
                    # （on_request 阶段的已在 async_pre_call_hook 中 append 过，避免重复）
                    if context.task_results:
                        append_task_results(context.task_results)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"on_response (failure) hook error: {e}", exc_info=True)
