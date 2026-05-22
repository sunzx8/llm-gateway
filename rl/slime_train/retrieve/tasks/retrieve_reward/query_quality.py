"""Query-quality sub-reward for Retrieve query rewrite outputs."""

from __future__ import annotations

try:  # pragma: no cover - fallback for direct script-style imports
    from .parser import strip_think_wrapper, try_parse_json
except ImportError:  # pragma: no cover
    from parser import strip_think_wrapper, try_parse_json


def compute_r_query_quality(response: str) -> float:
    """Evaluate query diversity, count range, and text sanity."""
    text = strip_think_wrapper(response)
    parsed = try_parse_json(text)
    if parsed is None:
        return 0.0

    queries = parsed.get("queries", [])
    if not isinstance(queries, list) or len(queries) == 0:
        return 0.0

    score = 0.0
    type_counts: dict[str, int] = {}
    for query in queries:
        if isinstance(query, dict) and query.get("type"):
            query_type = query["type"]
            type_counts[query_type] = type_counts.get(query_type, 0) + 1

    if type_counts.get("semantic", 0) >= 2:
        score += 0.3
    elif type_counts.get("semantic", 0) >= 1:
        score += 0.15

    if type_counts.get("keyword", 0) >= 1:
        score += 0.2

    if type_counts.get("entity", 0) >= 1:
        score += 0.2

    total = len(queries)
    if 3 <= total <= 12:
        score += 0.2
    elif 1 <= total < 3:
        score += 0.1

    non_empty_reasonable = sum(
        1 for query in queries
        if isinstance(query, dict)
        and isinstance(query.get("text"), str)
        and 2 <= len(query["text"]) <= 200
    )
    if queries:
        score += 0.1 * (non_empty_reasonable / len(queries))

    return min(score, 1.0)
