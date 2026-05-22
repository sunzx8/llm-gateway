"""
向量数据库工具集

包含所有向量数据库相关的工具类。
"""

from typing import Any

from .base_tool import BaseTool


class VecAddTool(BaseTool):
    name = "vec_add"
    description = (
        "向 collection 添加文本（自动 embed + 建索引）。collection 不存在时自动创建。\n\n"
        "**texts**：优先用户第一人称原话（\"I stopped listening...\"），不要第三人称改写。\n"
        "**metadatas**：每条 text 一个 dict，缺失时自动兜底为 {type:'auto'}。\n\n"
        "示例调用：\n"
        "```json\n"
        "{\"collection\": \"facts_alex\",\n"
        " \"texts\": [\"I stopped listening to book podcasts because they felt repetitive.\"],\n"
        " \"metadatas\": [{\"type\": \"event_mentioned\", \"evidence_type\": \"recent_event\",\n"
        "   \"subject\": \"alex\", \"source\": \"source_sessions/17_ctx0_s0.jsonl:76\",\n"
        "   \"session_index\": 0, \"narrative_time\": \"s0/line76\", \"topic\": \"book_podcasts\"}]}\n"
        "```"
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "collection 名（如 'facts_alex', 'events_user', 'preferences_alex'）"},
            "texts": {"type": "array", "items": {"type": "string"}, "description": "文本列表（第一人称原话优先）"},
            "metadatas": {
                "type": "array",
                "items": {"type": "object"},
                "description": "每条 text 的 metadata dict（推荐 key：type/evidence_type/subject/source/narrative_time/topic）",
            },
        },
        "required": ["collection", "texts"],
    }
    is_readonly = False
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        import time as _time
        vec = deps["vec"]
        record_id = int(_time.time() * 1000)
        ids = await vec.add(
            collection=args["collection"],
            texts=args["texts"],
            metadatas=args.get("metadatas"),
            record_id=record_id,
        )
        return f"Added {len(ids)} entries to '{args['collection']}': {ids} (record_id={record_id})"


class VecSearchTool(BaseTool):
    name = "vec_search"
    description = (
        "在指定 collection 中执行语义搜索。按相似度返回排序后的文本。\n\n"
        "💡 可选 `filter`：按 metadata 等值过滤（如 "
        "`{\"type\":\"event_mentioned\"}`），便于在重复检查 / 查重时精准"
        "定位同类条目。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "要搜索的 collection"},
            "query": {"type": "string", "description": "搜索查询"},
            "top_k": {"type": "integer", "description": "返回结果数（默认：10）"},
            "filter": {"type": "object", "description": "可选 metadata 等值过滤"},
        },
        "required": ["collection", "query"],
    }
    is_readonly = True
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        results = await vec.search(
            collection=args["collection"],
            query=args["query"],
            top_k=int(args.get("top_k", 10)),
            metadata_filter=args.get("filter") or None,
        )
        return str(results)


class VecSearchAllTool(BaseTool):
    name = "vec_search_all"
    description = "在所有 collection 中执行语义搜索。按相似度返回排序后的文本。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询"},
            "top_k": {"type": "integer", "description": "返回结果数（默认：15）"},
            "filter": {"type": "object", "description": "可选 metadata 等值过滤"},
        },
        "required": ["query"],
    }
    is_readonly = True
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        results = await vec.search_all(
            query=args["query"],
            top_k=int(args.get("top_k", 15)),
            metadata_filter=args.get("filter") or None,
        )
        return str(results)


class VecListCollectionsTool(BaseTool):
    name = "vec_list_collections"
    description = "列出所有 vector collection 及其大小。"
    parameters = {"type": "object", "properties": {}}
    is_readonly = True
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        # 兼容不同后端：有的用 get_stats()，有的用 list_collections()
        if hasattr(vec, "get_stats"):
            return str(vec.get_stats())
        return str(vec.list_collections())


class VecScrollTool(BaseTool):
    name = "vec_scroll"
    description = "滚动浏览向量数据库中的条目（仅PG后端可用）"
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "集合名称"},
            "filter": {"type": "object", "description": "元数据过滤器"},
            "cursor": {"type": "string", "description": "游标位置"},
            "page_size": {"type": "integer", "description": "页面大小，默认100"},
        },
    }
    is_readonly = True
    requires_backend = "pg"
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        if not hasattr(vec, "scroll"):
            return "ERROR: vec_scroll only available on pg backend"
        result = await vec.scroll(
            collection=args.get("collection"),
            metadata_filter=args.get("filter") or None,
            cursor=args.get("cursor"),
            page_size=args.get("page_size", 100),
        )
        return str(result)


class VecDeleteCollectionTool(BaseTool):
    name = "vec_delete_collection"
    description = "删除整个 vector collection。"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "要删除的 collection 名"},
        },
        "required": ["name"],
    }
    is_readonly = False
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        return vec.delete_collection(args["name"])


class VecCreateCollectionTool(BaseTool):
    name = "vec_create_collection"
    description = "创建一个新的 vector collection。"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "要创建的 collection 名"},
        },
        "required": ["name"],
    }
    is_readonly = False
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        return vec.create_collection(args["name"])


class VecDeleteTool(BaseTool):
    name = "vec_delete"
    description = "从指定 collection 中删除指定 ID 的条目。"
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "collection 名"},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "要删除的条目 ID 列表"},
        },
        "required": ["collection", "ids"],
    }
    is_readonly = False
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        return vec.delete(args["collection"], args["ids"])


class VecUpdateTool(BaseTool):
    name = "vec_update"
    description = (
        "更新指定 collection 中某条目的文本和/或元数据。\n\n"
        "至少提供 `new_text` 或 `new_metadata` 之一。`new_metadata` 为增量合并"
        "（不会覆盖未提及的 key）。\n\n"
        "示例：`{\"collection\": \"facts_alex\", \"entry_id\": \"abc123\", "
        "\"new_text\": \"I now enjoy cooking\", \"new_metadata\": {\"confidence\": \"high\"}}`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "collection 名"},
            "entry_id": {"type": "string", "description": "要更新的条目 ID"},
            "new_text": {"type": "string", "description": "新的文本内容（可选）"},
            "new_metadata": {"type": "object", "description": "要合并的新 metadata（可选）"},
        },
        "required": ["collection", "entry_id"],
    }
    is_readonly = False
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        return vec.update(
            collection=args["collection"],
            entry_id=args["entry_id"],
            new_text=args.get("new_text"),
            new_metadata=args.get("new_metadata"),
        )


class VecSearchAllWithEmbeddingTool(BaseTool):
    name = "vec_search_all_with_embedding"
    description = (
        "使用预计算的 embedding 向量跨所有 collection 检索。\n\n"
        "当你已经有一个 embedding 向量（例如从 graph 节点获取的 embedding）时，"
        "可以直接用它搜索，避免重复调用 embed 接口。\n\n"
        "返回 `[{id, text, score, metadata, collection}]`。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query_embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": "预计算的查询 embedding 向量",
            },
            "top_k": {"type": "integer", "description": "返回结果数（默认：10）"},
            "filter": {"type": "object", "description": "可选 metadata 等值过滤"},
        },
        "required": ["query_embedding"],
    }
    is_readonly = True
    category = "vec"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        vec = deps["vec"]
        results = await vec.search_all_with_embedding(
            query_embedding=args["query_embedding"],
            top_k=int(args.get("top_k", 10)),
            metadata_filter=args.get("filter") or None,
        )
        return str(results)
