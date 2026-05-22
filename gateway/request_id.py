"""
Request ID Middleware - Extract request_id and metadata from request body,
inject session_id into response body metadata.

Supports:
1. HTTP Header: X-Request-Id for request tracing
2. Request body metadata: {"metadata": {"user_id": "xxx", "session_id": "xxx"}}
3. Non-streaming (JSON) responses: inject task_results and session_id into JSON body
4. Streaming (SSE) responses: inject task_results and session_id as a custom SSE chunk
   before the [DONE] marker

The session_id will be returned in the response body metadata.
"""

import json
import time
import uuid
from typing import Any, AsyncIterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from logger.context import (
    get_request_id, set_request_id, reset_request_id,
    get_task_results, set_task_results, reset_task_results,
    get_session_id, set_session_id, reset_session_id,
    set_user_id, reset_user_id,
)
import logger.logger as logger


RESPONSE_HEADER_NAME = "X-Request-Id"
REQUEST_HEADER_NAME = "X-Request-Id"

# SSE 自定义 chunk 的标识，客户端可据此识别注入的元数据
_GATEWAY_META_EVENT = "__gateway_meta__"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Extracts request_id from HTTP header
    2. Extracts user_id and session_id from request body metadata
    3. Injects session_id into response body metadata
    4. Injects task_results into response body (JSON or SSE)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id: str | None = None
        t0 = time.monotonic()  # 记录请求开始时间

        # 从 HTTP header 提取 request_id
        request_id = request.headers.get(REQUEST_HEADER_NAME)
        if not request_id:
            request_id = uuid.uuid4().hex

        # 从请求体 metadata 中提取 user_id 和 session_id
        user_id = ""
        session_id = ""
        is_stream_request = False
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)
                    metadata = data.get("metadata", {})
                    if isinstance(metadata, dict):
                        user_id = metadata.get("user_id", "") or ""
                        session_id = metadata.get("session_id", "") or ""
                    # 检测是否为 streaming 请求
                    is_stream_request = bool(data.get("stream"))
            except (json.JSONDecodeError, TypeError):
                pass

        # Store in context var
        token = set_request_id(request_id)
        session_token = set_session_id(session_id)
        user_token = set_user_id(user_id)
        # 初始化 task_results contextvars
        task_token = set_task_results([])

        try:
            response = await call_next(request)

            # Inject request_id into response header
            if request_id:
                response.headers[RESPONSE_HEADER_NAME] = str(request_id)

            # 获取最终的 session_id（可能在 on_request 中被更新）
            final_session_id = get_session_id()

            # 判断响应是否为 SSE streaming
            content_type = response.headers.get("content-type", "")
            is_sse = "text/event-stream" in content_type

            task_results = get_task_results()

            if (task_results or final_session_id) and response.status_code == 200:
                if is_sse:
                    # ── Streaming (SSE) 响应：包装 body_iterator，在 [DONE] 前注入自定义 chunk ──
                    response = self._wrap_sse_response(
                        response, task_results, final_session_id,
                    )
                else:
                    # ── Non-streaming (JSON) 响应：读取整个 body 并注入 ──
                    response = await self._inject_into_json_response(
                        response, task_results, final_session_id,
                    )

            # 记录端到端耗时
            elapsed = time.monotonic() - t0
            logger.info(
                "API request completed: method=%s path=%s request_id=%s status=%d "
                "elapsed=%.2fs streaming=%s",
                request.method, request.url.path, request_id, response.status_code,
                elapsed, is_sse,
            )
            return response
        finally:
            reset_request_id(token)
            reset_session_id(session_token)
            reset_user_id(user_token)
            reset_task_results(task_token)

    # ------------------------------------------------------------------
    # Non-streaming JSON 响应注入
    # ------------------------------------------------------------------

    @staticmethod
    async def _inject_into_json_response(
        response: Response,
        task_results: list[dict[str, Any]],
        final_session_id: str,
    ) -> Response:
        """读取整个 JSON body，注入 task_results 和 session_id，返回新 Response。"""
        body_chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                body_chunks.append(chunk)
            else:
                body_chunks.append(chunk.encode("utf-8"))
        body = b"".join(body_chunks)

        try:
            resp_data = json.loads(body)
            if task_results:
                resp_data["task_results"] = task_results
            # 注入 metadata.session_id 到响应体
            if final_session_id:
                if "metadata" not in resp_data or not isinstance(resp_data.get("metadata"), dict):
                    resp_data["metadata"] = {}
                resp_data["metadata"]["session_id"] = final_session_id
            new_body = json.dumps(resp_data, ensure_ascii=False, default=str).encode("utf-8")
            # 重新构建响应，需要移除原始 Content-Length，让 Starlette 自动计算
            new_headers = dict(response.headers)
            new_headers.pop("content-length", None)
            return Response(
                content=new_body,
                status_code=response.status_code,
                headers=new_headers,
                media_type="application/json",
            )
        except (json.JSONDecodeError, TypeError):
            # 非 JSON 响应，不注入
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

    # ------------------------------------------------------------------
    # Streaming SSE 响应注入
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_sse_response(
        response: Response,
        task_results: list[dict[str, Any]],
        final_session_id: str,
    ) -> StreamingResponse:
        """包装 SSE body_iterator，在 [DONE] 前注入携带 task_results 的自定义 chunk。

        注入的自定义 SSE chunk 格式：
            data: {"__gateway_meta__": true, "task_results": [...], "metadata": {"session_id": "xxx"}}

        客户端解析 SSE 时，遇到包含 "__gateway_meta__" 字段的 chunk 即可提取 task_results。
        """
        original_iterator = response.body_iterator

        async def _wrapped_iterator() -> AsyncIterator[bytes]:
            # 构建注入的元数据 chunk
            meta_payload: dict[str, Any] = {_GATEWAY_META_EVENT: True}
            if task_results:
                meta_payload["task_results"] = task_results
            if final_session_id:
                meta_payload["metadata"] = {"session_id": final_session_id}
            meta_chunk = "data: " + json.dumps(meta_payload, ensure_ascii=False, default=str) + "\n\n"
            meta_bytes = meta_chunk.encode("utf-8")

            injected = False
            async for chunk in original_iterator:
                if isinstance(chunk, str):
                    chunk_bytes = chunk.encode("utf-8")
                else:
                    chunk_bytes = chunk

                # 检测 [DONE] 标记，在其前面注入元数据 chunk
                # LiteLLM 的 SSE 格式：每个 chunk 是 "data: {json}\n\n" 或 "data: [DONE]\n\n"
                if not injected and b"data: [DONE]" in chunk_bytes:
                    # 先发送元数据 chunk
                    yield meta_bytes
                    injected = True

                yield chunk_bytes

            # 如果整个流中没有 [DONE]（异常情况），在末尾追加元数据
            if not injected:
                yield meta_bytes

        new_headers = dict(response.headers)
        new_headers.pop("content-length", None)
        return StreamingResponse(
            _wrapped_iterator(),
            status_code=response.status_code,
            headers=new_headers,
            media_type="text/event-stream",
        )
