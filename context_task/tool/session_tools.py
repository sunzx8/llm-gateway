"""
会话工具集

包含会话相关的工具类。
"""

import json as _json
from typing import Any

from .base_tool import BaseTool


class SessionViewTool(BaseTool):
    name = "session_view"
    description = (
        "查看当前会话的 messages 队列信息，包括消息内容和每条 assistant 回复的请求耗时（duration_ms）。\n\n"
        "返回格式为 JSON 列表，每个元素包含：\n"
        "  - index: 消息的全局序号\n"
        "  - role: 消息角色（user/assistant/system）\n"
        "  - content: 消息内容（截断至前500字符）\n"
        "  - duration_ms: 请求耗时（毫秒），仅 assistant 消息包含此字段，表示从接收到用户问题到获取到该回复的端到端耗时\n\n"
        "可选参数 last_n 用于只查看最近 N 条消息，避免返回过多内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "last_n": {"type": "integer", "description": "只返回最近 N 条消息（默认返回全部）"},
            "include_content": {"type": "boolean", "description": "是否包含消息内容（默认 true），设为 false 时只返回元数据和耗时"},
        },
    }
    is_readonly = True
    category = "session"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        """执行 session_view，通过 deps 中的 user_id 和 session_id 获取会话信息。"""
        from storage.session_store import session_manager

        user_id = deps.get("user_id", "default_user")
        session_id = deps.get("session_id", "")

        if not session_id:
            return _json.dumps({"error": "no session_id available"}, ensure_ascii=False)

        session = await session_manager.get_session(user_id, session_id)
        if session is None:
            return _json.dumps({
                "error": f"session not found for user_id={user_id}, session_id={session_id}",
                "total_messages": 0,
            }, ensure_ascii=False)

        last_n = int(args.get("last_n", 0))
        include_content = args.get("include_content", True)

        # 按全局序号排序获取消息
        sorted_items = sorted(session.messages.items(), key=lambda x: int(x[0]))

        # 如果指定了 last_n，只取最后 N 条
        if last_n > 0:
            sorted_items = sorted_items[-last_n:]

        result_messages = []
        for idx_str, msg in sorted_items:
            item: dict[str, Any] = {
                "index": int(idx_str),
                "role": msg.get("role", ""),
            }
            if include_content:
                content = msg.get("content", "")
                # 截断过长内容
                item["content"] = content[:500] + ("..." if len(content) > 500 else "")
            if "received_at" in msg:
                item["received_at"] = msg["received_at"]
            if "duration_ms" in msg:
                item["duration_ms"] = msg["duration_ms"]
            result_messages.append(item)

        summary = {
            "user_id": user_id,
            "session_id": session_id,
            "total_messages": len(session.messages),
            "returned_messages": len(result_messages),
            "messages": result_messages,
        }
        return _json.dumps(summary, ensure_ascii=False, indent=2)
