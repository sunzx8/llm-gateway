# ---------------------------------------------------------------------------
# 数据类：请求上下文
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
import datetime 
from typing import Any
from abc import ABC, abstractmethod
from datetime import datetime
import logger.logger as logger

@dataclass
class RequestContext:
    """请求前钩子的上下文，包含本次请求的所有信息，可以直接修改字段来影响实际请求"""

    # 请求的模型名（对应 config.yaml 中的 model_name）
    model: str

    # 对话消息列表，格式同 OpenAI messages
    messages: list[dict]

    # 发起请求的用户 ID（来自 Virtual Key 绑定，未设置则为 None）
    user_id: str | None

    # 发起请求的 API Key（脱敏后的前缀）
    api_key_prefix: str | None

    # 原始请求的完整参数字典，可以在此修改任意参数
    raw_data: dict[str, Any]

    request_id: str | None

    # 会话 ID
    session_id: str | None = None

    # 请求到达时间
    request_time: datetime = field(default_factory=datetime.now)

    # 中间任务执行结果列表，每个元素包含 task_type、input_params、result
    task_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 数据类：响应上下文
# ---------------------------------------------------------------------------

@dataclass
class ResponseContext:
    """响应后钩子的上下文，包含本次请求和响应的完整信息"""

    # 请求的模型名
    model: str

    # 发起请求的用户 ID
    user_id: str | None

    # 请求 ID，与 RequestContext 中的 request_id 一致
    request_id: str | None

    # 请求消息列表
    messages: list[dict]

    # 响应内容（流式时为 None）
    response_content: str | None

    # Token 用量
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # 本次请求的估算费用（美元）
    cost: float

    # 请求耗时（秒）
    duration_seconds: float

    # 请求开始时间
    start_time: datetime

    # 请求结束时间
    end_time: datetime

    # 是否成功
    success: bool = True

    # 会话 ID
    session_id: str | None = None

    # 失败时的异常信息
    error: Exception | None = None

    # 中间任务执行结果列表，每个元素包含 task_type、input_params、result
    task_results: list[dict[str, Any]] = field(default_factory=list)


class CallbackHandler(ABC):

    @abstractmethod
    async def on_request(self, context: RequestContext) -> RequestContext:
        """ """
    
    @abstractmethod
    async def on_response(self, context: ResponseContext) -> ResponseContext:
        """响应后钩子，可在 context.task_results 中追加任务执行结果，返回 ResponseContext。"""

class DefaultCallbackHandler(CallbackHandler):

    async def on_request(self, context: RequestContext) -> RequestContext:
        logger.info(f"On Request but do nothing,{context}")
        context.messages = [{
            "role": "user",
            "content": "Hello"
        }]
        return context

    async def on_response(self, context: ResponseContext) -> ResponseContext:
        logger.info(f"On Response but do nothing")
        return context
