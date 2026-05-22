"""
图数据库工具集

包含所有图数据库相关的工具类。
"""

from typing import Any

from .base_tool import BaseTool


class GraphAddNodeTool(BaseTool):
    name = "graph_add_node"
    description = (
        "向知识图中添加或更新一个节点。\n\n"
        "**node_id 命名**：用小写 slug（如 `comedy_shows`、`cooking_class`、`budgeting`）。\n"
        "**label 推荐值**：`Person` / `Topic` / `Activity` / `Genre` / `Preference`。\n\n"
        "示例：`{\"node_id\": \"comedy_shows\", \"label\": \"Topic\", \"properties\": {\"category\": \"entertainment\"}}`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "节点唯一标识（小写 slug，如 'comedy_shows'）"},
            "label": {"type": "string", "description": "节点类型（Person / Topic / Activity / Genre / Preference）"},
            "properties": {"type": "object", "description": "节点属性"},
            "timestamp": {"type": "string", "description": "记录时间"},
        },
        "required": ["node_id"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        import time as _time
        graph = deps["graph"]
        record_id = int(_time.time() * 1000)
        result = graph.add_node(
            node_id=args["node_id"],
            label=args.get("label", ""),
            properties=args.get("properties"),
            timestamp=args.get("timestamp", ""),
            record_id=record_id,
        )
        return result


class GraphAddEdgeTool(BaseTool):
    name = "graph_add_edge"
    description = (
        "在两个节点间添加一条关系边。节点不存在时自动创建。\n\n"
        "**relation 命名**：用小写动词短语（如 `prefers`、`avoids`、`tried`、`withdrew_from`）。"
        "查看图统计里已有的 relation 类型，优先沿用已有名称保持一致。\n\n"
        "示例：`{\"source\": \"user\", \"target\": \"comedy_shows\", \"relation\": \"prefers\", \"properties\": {\"confidence\": \"high\"}}`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "源节点 ID"},
            "target": {"type": "string", "description": "目标节点 ID"},
            "relation": {"type": "string", "description": "关系类型（小写动词短语，沿用已有 relation 名保持一致）"},
            "properties": {"type": "object", "description": "边属性"},
            "timestamp": {"type": "string", "description": "关系建立时间"},
        },
        "required": ["source", "target", "relation"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        import time as _time
        graph = deps["graph"]
        record_id = int(_time.time() * 1000)
        return graph.add_edge(
            source=args["source"],
            target=args["target"],
            relation=args["relation"],
            properties=args.get("properties"),
            timestamp=args.get("timestamp", ""),
            record_id=record_id,
        )


class GraphGetNeighborsTool(BaseTool):
    name = "graph_get_neighbors"
    description = "获取某节点的所有邻居（相连的实体与关系）。"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "要查询的节点"},
            "relation": {"type": "string", "description": "按关系类型过滤（可选）"},
            "direction": {"type": "string", "description": "'out'、'in' 或 'both'（默认：'both'）"},
        },
        "required": ["node_id"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        results = graph.get_neighbors(
            args["node_id"],
            relation=args.get("relation"),
            direction=args.get("direction", "both"),
        )
        return str(results)


class GraphSearchNodesTool(BaseTool):
    name = "graph_search_nodes"
    description = "按标签、属性或关键词搜索节点。"
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "按节点标签过滤"},
            "keyword": {"type": "string", "description": "在节点 ID、标签、属性中做关键词搜索"},
        },
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        results = graph.search_nodes(
            label=args.get("label"),
            keyword=args.get("keyword"),
        )
        return str(results)


class GraphStatsTool(BaseTool):
    name = "graph_stats"
    description = "获取图数据库统计（节点数、边数、关系类型）。"
    parameters = {"type": "object", "properties": {}}
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return str(graph.get_stats())


class GraphDeleteNodeTool(BaseTool):
    name = "graph_delete_node"
    description = "删除指定节点及其关联的边。"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "要删除的节点 ID"},
        },
        "required": ["node_id"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return graph.delete_node(args["node_id"])


class GraphDeleteEdgeTool(BaseTool):
    name = "graph_delete_edge"
    description = "删除指定的边。"
    parameters = {
        "type": "object",
        "properties": {
            "edge_id": {"type": "string", "description": "要删除的边 ID"},
        },
        "required": ["edge_id"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return graph.delete_edge(args["edge_id"])


class GraphGetSubgraphTool(BaseTool):
    name = "graph_get_subgraph"
    description = "获取以指定节点为中心的子图（BFS 扩展到指定深度）。"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "中心节点 ID"},
            "depth": {"type": "integer", "description": "扩展深度（默认 2）"},
        },
        "required": ["node_id"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        result = graph.get_subgraph(
            args["node_id"],
            depth=args.get("depth", 2),
        )
        return str(result)


class GraphCypherReadTool(BaseTool):
    name = "graph_cypher_read"
    description = "执行只读Cypher查询（仅PG后端可用）"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Cypher查询语句"},
            "params": {"type": "object", "description": "查询参数"},
            "limit": {"type": "integer", "description": "结果限制，默认100"},
            "timeout_s": {"type": "number", "description": "超时时间，默认5秒"},
        },
        "required": ["query"],
    }
    is_readonly = True
    requires_backend = "pg"
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        if not hasattr(graph, "cypher_read"):
            return "ERROR: graph_cypher_read only available on pg backend"
        result = await graph.cypher_read(
            query=args["query"],
            params=args.get("params") or None,
            limit=args.get("limit", 100),
            timeout_s=args.get("timeout_s", 5.0),
        )
        return str(result)


class GraphCypherWriteTool(BaseTool):
    name = "graph_cypher_write"
    description = (
        "对当前 user 的 AGE graph 执行**可写** Cypher（PG 后端独有）。"
        "适合 consolidate 阶段做批量重组：补全 stance_evolution 边、合并节点、"
        "重写 properties 等。\n\n"
        "**约束**：自动锁定在当前 user 的 graph，10s timeout。\n\n"
        "**示例**：\n"
        "  - 给 user 节点和 topic 之间补一条 stance 边：\n"
        "    `MATCH (u {__id:'user'}), (t:Topic {__id:'cooking'}) MERGE (u)-[r:prefers]->(t) SET r.__ts = '2026-04-28' RETURN r`\n"
        "  - 删除一条错误的边：\n"
        "    `MATCH ()-[r {__id:'e_42'}]->() DELETE r`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Cypher（可写）"},
            "params": {"type": "object", "description": "Cypher $name 参数"},
            "timeout_s": {"type": "number", "description": "默认 10"},
        },
        "required": ["query"],
    }
    is_readonly = False
    requires_backend = "pg"
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        if not hasattr(graph, "cypher_write"):
            return "ERROR: graph_cypher_write only available on pg backend"
        result = await graph.cypher_write(
            query=args["query"],
            params=args.get("params") or None,
            timeout_s=args.get("timeout_s", 10.0),
        )
        return str(result)


class GraphGetNodeTool(BaseTool):
    name = "graph_get_node"
    description = (
        "获取指定节点的完整信息（ID、标签、属性、时间戳等）。\n\n"
        "返回节点的全部属性字典，若节点不存在则返回 None。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "要查询的节点 ID"},
        },
        "required": ["node_id"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        result = graph.get_node(args["node_id"])
        if result is None:
            return f"Node '{args['node_id']}' not found."
        return str(result)


class GraphSearchEdgesTool(BaseTool):
    name = "graph_search_edges"
    description = (
        "按关系类型、源节点、目标节点或关键词搜索边。\n\n"
        "至少提供一个过滤条件。返回匹配的边列表，每条边包含 "
        "`{id, source, target, relation, properties}`。\n\n"
        "示例：`{\"relation\": \"prefers\", \"source\": \"user\"}`"
    )
    parameters = {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "description": "按关系类型过滤（如 'prefers'、'avoids'）"},
            "source": {"type": "string", "description": "按源节点 ID 过滤"},
            "target": {"type": "string", "description": "按目标节点 ID 过滤"},
            "keyword": {"type": "string", "description": "在边的源/目标/关系/属性中做关键词搜索"},
        },
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        results = graph.search_edges(
            relation=args.get("relation"),
            source=args.get("source"),
            target=args.get("target"),
            keyword=args.get("keyword"),
        )
        return str(results)


class GraphSearchNodesByEmbeddingTool(BaseTool):
    name = "graph_search_nodes_by_embedding"
    description = (
        "基于向量相似度搜索图中的节点。\n\n"
        "需要提供查询 embedding 向量，返回按相似度降序排列的节点列表，"
        "每个节点附带 `similarity` 字段。\n\n"
        "⚠️ 内存版在节点数量大时性能较差；大规模场景建议使用 `vec_search_all` 做语义检索。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query_embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": "查询 embedding 向量",
            },
            "top_k": {"type": "integer", "description": "返回最相似的前 k 个节点（默认 10）"},
            "threshold": {"type": "number", "description": "最低相似度阈值（默认 0.0）"},
        },
        "required": ["query_embedding"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        results = graph.search_nodes_by_embedding(
            query_embedding=args["query_embedding"],
            top_k=int(args.get("top_k", 10)),
            threshold=float(args.get("threshold", 0.0)),
        )
        return str(results)


class GraphSearchByTimeTool(BaseTool):
    name = "graph_search_by_time"
    description = (
        "基于时间范围搜索图中的节点和边。\n\n"
        "支持粗粒度时间匹配：\n"
        "  - `\"2017\"` → 匹配 occurred_at 以 \"2017\" 开头的所有记录\n"
        "  - `\"2019-03\"` → 匹配 occurred_at 以 \"2019-03\" 开头的记录\n"
        "  - `\"2018~2020\"` → 匹配 2018 到 2020 年的记录（范围查询）\n\n"
        "返回 `{\"nodes\": [...], \"edges\": [...]}`。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "time_query": {"type": "string", "description": "时间查询字符串（如 '2017'、'2019-03'、'2018~2020'）"},
            "node_id": {"type": "string", "description": "可选，限定与某个节点相关的记录"},
            "label": {"type": "string", "description": "可选，限定节点标签"},
        },
        "required": ["time_query"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        result = graph.search_by_time(
            time_query=args["time_query"],
            node_id=args.get("node_id"),
            label=args.get("label"),
        )
        return str(result)


class GraphExecuteQueryTool(BaseTool):
    name = "graph_execute_query"
    description = (
        "执行类 Cypher 查询语句（简化版，供模型灵活使用）。\n\n"
        "支持的快捷查询：\n"
        "  - `\"show nodes\"` / `\"all nodes\"` → 列出前 50 个节点\n"
        "  - `\"show edges\"` / `\"all edges\"` → 列出前 50 条边\n"
        "  - `\"stats\"` → 返回图统计信息\n\n"
        "PG 后端还支持完整的 Cypher 语法。内存版仅支持上述快捷查询。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询语句（快捷命令或 Cypher）"},
        },
        "required": ["query"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return graph.execute_query(args["query"])


class GraphSetNodeEmbeddingTool(BaseTool):
    name = "graph_set_node_embedding"
    description = "为已有节点设置嵌入向量。用于后续基于向量相似度搜索节点。"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "节点 ID"},
            "embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": "嵌入向量",
            },
        },
        "required": ["node_id", "embedding"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return graph.set_node_embedding(args["node_id"], args["embedding"])


class GraphGetNodeEmbeddingTool(BaseTool):
    name = "graph_get_node_embedding"
    description = "获取节点的嵌入向量。返回向量列表或 None（若未设置）。"
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "节点 ID"},
        },
        "required": ["node_id"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        result = graph.get_node_embedding(args["node_id"])
        if result is None:
            return f"No embedding found for node '{args['node_id']}'."
        return str(result)


class GraphSetEdgeEmbeddingTool(BaseTool):
    name = "graph_set_edge_embedding"
    description = "为边设置嵌入向量。"
    parameters = {
        "type": "object",
        "properties": {
            "edge_id": {"type": "string", "description": "边 ID"},
            "embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": "嵌入向量",
            },
        },
        "required": ["edge_id", "embedding"],
    }
    is_readonly = False
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        return graph.set_edge_embedding(args["edge_id"], args["embedding"])


class GraphGetEdgeEmbeddingTool(BaseTool):
    name = "graph_get_edge_embedding"
    description = "获取边的嵌入向量。返回向量列表或 None（若未设置）。"
    parameters = {
        "type": "object",
        "properties": {
            "edge_id": {"type": "string", "description": "边 ID"},
        },
        "required": ["edge_id"],
    }
    is_readonly = True
    category = "graph"

    async def execute(self, args: dict[str, Any], **deps) -> str:
        graph = deps["graph"]
        result = graph.get_edge_embedding(args["edge_id"])
        if result is None:
            return f"No embedding found for edge '{args['edge_id']}'."
        return str(result)
