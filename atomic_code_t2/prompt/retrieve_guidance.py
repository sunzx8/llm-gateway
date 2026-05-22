"""Atomic_Code_T2 专用的回答引导模板。

拼装到 query_memory 中，告诉回答 LLM 记忆的排序规则。
"""

ATOMIC_T2_QUERY_GUIDANCE_TEMPLATE = """\
# About the Memory Above

Current time: {now}

The memory is assembled from three backends: File System (BM25 snippets), Vector DB (atomic facts), and Graph DB (entity relations).

## Ordering Rule (Important)

All entries within each section are listed in **chronological order — later = more recent = more authoritative**.
When the same topic appears with contradictory states, trust the later entry for the user's current position.

## Reading Tips

- Parenthesized dates like `(2023-07-01, Sat)` indicate when events actually happened.
- Do not invent details absent from the memory. Semantic synonyms count as support (e.g. "online community" supports "online forum").
- When multiple options are consistent with memory, prefer the one with the most **direct, specific** evidence over one merely inferred from a broad pattern.
"""
