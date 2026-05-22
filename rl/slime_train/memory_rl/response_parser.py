from __future__ import annotations

import json
import re
from typing import Any


def strip_think_wrapper(response: str) -> str:
    """Remove thinking wrappers, special tokens and markdown JSON fences."""
    text = (response or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text and "</think>" not in text:
        return ""

    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "")

    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()
    return text.strip()


def try_parse_json(text: str) -> dict | None:
    """Parse a JSON object from text or from the first/last brace span."""
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            data = json.loads(text[first:last + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_tool_calls_response(response: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse policy output into a list of ``{tool, arguments}`` calls.

    Supported payloads:
    - ``{"tool_calls": [{"tool": "fs_write", "arguments": {...}}]}``
    - ``{"operations": [...]}``
    - ``{"fs_ops": [...], "vec_ops": [...], "graph_ops": [...]}``
    """
    # 优先直接解析（避免 strip_think_wrapper 误截含 </think> 文本的合法 JSON）
    parsed = try_parse_json(response.strip() if response else "")
    if parsed is None:
        parsed = try_parse_json(strip_think_wrapper(response))
    if parsed is None:
        return [], None

    if isinstance(parsed.get("tool_calls"), list):
        return [_normalize_call(c) for c in parsed["tool_calls"]], parsed
    if isinstance(parsed.get("operations"), list):
        return [_normalize_call(c) for c in parsed["operations"]], parsed

    calls: list[dict[str, Any]] = []
    for op in parsed.get("fs_operations", parsed.get("fs_ops", [])) or []:
        calls.append({"tool": "fs_write", "arguments": op})
    for op in parsed.get("vec_operations", parsed.get("vec_ops", [])) or []:
        calls.append({"tool": "vec_write", "arguments": op})
    for op in parsed.get("graph_operations", parsed.get("graph_ops", [])) or []:
        calls.append({"tool": "graph_write", "arguments": op})
    if parsed.get("summary"):
        calls.append({"tool": "finish", "arguments": {"summary": str(parsed["summary"])}})
    return calls, parsed


def _normalize_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"tool": "", "arguments": {}}
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    tool = call.get("tool") or call.get("name") or function.get("name") or ""
    arguments = call.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    normalized = {"tool": str(tool), "arguments": arguments}
    if call.get("id"):
        normalized["id"] = str(call["id"])
    return normalized


def format_reward(response: str) -> float:
    """Lightweight structural reward for JSON tool-call outputs."""
    calls, parsed = parse_tool_calls_response(response)
    if parsed is None:
        return 0.0
    score = 0.3
    if calls:
        score += 0.3
    valid_tools = {
        "fs_write", "vec_write", "graph_write", "finish",
        "fs_append", "fs_delete", "vec_add", "vec_delete",
        "graph_add_node", "graph_add_edge", "graph_delete_node", "graph_delete_edge",
    }
    well_formed = 0
    for call in calls:
        if call.get("tool") in valid_tools and isinstance(call.get("arguments"), dict):
            well_formed += 1
    if calls:
        score += 0.3 * (well_formed / len(calls))
    if any(call.get("tool") == "finish" for call in calls):
        score += 0.1
    return min(score, 1.0)
