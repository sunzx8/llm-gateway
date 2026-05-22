"""Format sub-reward for Retrieve query rewrite outputs."""

from __future__ import annotations

try:  # pragma: no cover - fallback for direct script-style imports
    from .parser import strip_think_wrapper, try_parse_json
except ImportError:  # pragma: no cover
    from parser import strip_think_wrapper, try_parse_json


def compute_r_format(response: str) -> float:
    """Evaluate whether model output matches the expected JSON query schema."""
    text = strip_think_wrapper(response)
    if not text:
        return 0.0

    score = 0.0
    parsed = try_parse_json(text)
    if parsed is None:
        return 0.0
    score += 0.3

    queries = parsed.get("queries")
    if not isinstance(queries, list) or len(queries) == 0:
        return score
    score += 0.3

    valid_types = {"semantic", "keyword", "entity", "temporal", "fs_scope"}
    well_formed = 0
    for query in queries:
        if isinstance(query, dict) and query.get("type") in valid_types and query.get("text"):
            well_formed += 1
    if queries:
        score += 0.2 * (well_formed / len(queries))

    graph_config = parsed.get("graph_config")
    if isinstance(graph_config, dict) and "depth" in graph_config:
        score += 0.2

    return min(score, 1.0)
