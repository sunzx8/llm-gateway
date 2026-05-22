"""
SQL 工具集

包含 SQL 查询相关的工具类。
"""

from typing import Any

from .base_tool import BaseTool


class SqlQueryReadTool(BaseTool):
    name = "sql_query_read"
    description = "执行只读SQL查询（仅PG后端可用）"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL查询语句"},
            "params": {"type": "object", "description": "查询参数"},
            "limit": {"type": "integer", "description": "结果限制，默认200"},
            "timeout_s": {"type": "number", "description": "超时时间，默认5秒"},
        },
        "required": ["query"],
    }
    is_readonly = True
    requires_backend = "pg"
    category = "sql"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        if not hasattr(vec, "sql_query_read"):
            return "ERROR: sql_query_read only available on pg backend"
        result = await vec.sql_query_read(
            query=args["query"],
            params=args.get("params") or None,
            limit=args.get("limit", 200),
            timeout_s=args.get("timeout_s", 5.0),
        )
        return str(result)
