from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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


def parse_task_loop_payload(response: str) -> dict[str, Any] | None:
    """Parse the native task-loop JSON wrapper without scanning nested tool XML.

    Slime stores rollout output in ``sample.response`` as a string. In task-loop
    mode that string is a JSON wrapper containing fields such as
    ``post_consolidate_snapshot`` and ``agentic_trace``. The trace may contain
    raw model text with ``<tool_call>`` XML, so callers that need wrapper fields
    must parse this JSON envelope directly instead of using the generic tool-call
    parser.
    """
    candidates = []
    text = (response or "").strip()
    if text:
        candidates.append(text)
    clean_text = strip_think_wrapper(response) if response else ""
    if clean_text and clean_text != text:
        candidates.append(clean_text)

    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if any(key in data for key in ("agentic_trace", "task_result", "post_consolidate_snapshot", "stop_reason")):
            return data
    return None


# ---------------------------------------------------------------------------
# Qwen3 Coder XML tool-call parser
# ---------------------------------------------------------------------------
# Adapted from sglang Qwen3CoderDetector (sglang/srt/function_call/qwen3_coder_detector.py)
#
# Qwen3.6 chat_template produces tool calls in this format:
#
#   <tool_call>
#   <function=function_name>
#   <parameter=param_name>
#   value
#   </parameter>
#   <parameter=param2>
#   value2
#   </parameter>
#   </function>
#   </tool_call>
#
# Multiple tool_call blocks may appear in one response.

# Regex patterns from sglang Qwen3CoderDetector
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)

# Tool schema registry for type-aware parameter conversion (optional)
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {}


def set_tool_schemas(tools: list[dict[str, Any]] | None) -> None:
    """Register tool schemas for type-aware parameter parsing.

    Call this once at generate startup with the tools list from metadata.
    """
    global _TOOL_SCHEMAS
    _TOOL_SCHEMAS = {}
    if not tools:
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") or tool
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        params = func.get("parameters", {})
        if isinstance(params, dict) and "properties" in params:
            _TOOL_SCHEMAS[name] = params["properties"]
        elif isinstance(params, dict):
            _TOOL_SCHEMAS[name] = params


def _get_param_config(func_name: str) -> dict[str, Any]:
    """Get parameter schema for a function."""
    return _TOOL_SCHEMAS.get(func_name, {})


def _convert_param_value(param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
    """Convert parameter value based on its type in the schema.

    Logic adapted from sglang Qwen3CoderDetector._convert_param_value.
    """
    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        # No schema info, try to infer type
        return _maybe_json_value(param_value)

    if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
        param_type = str(param_config[param_name]["type"]).strip().lower()
    else:
        param_type = "string"

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    elif param_type.startswith("int") or param_type.startswith("uint") or param_type.startswith("long"):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.debug(f"Cannot convert '{param_value}' to int for {func_name}.{param_name}")
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            val = float(param_value)
            if val.is_integer() and "." not in param_value:
                return int(val)
            return val
        except (ValueError, TypeError):
            return param_value
    elif param_type in ("boolean", "bool"):
        return param_value.lower() == "true"
    elif param_type in ("object", "array", "arr") or param_type.startswith("dict") or param_type.startswith("list"):
        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(param_value)
            except (ValueError, SyntaxError):
                return param_value
    else:
        return _maybe_json_value(param_value)


def _maybe_json_value(value: str) -> Any:
    """Try to parse a parameter value as JSON; fall back to raw string."""
    value = value.strip()
    if not value:
        return value
    if value[0] in "[{\"" or value in {"true", "false", "null"}:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    # Try numeric
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        pass
    return value


def _parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen3 Coder-style XML ``<tool_call><function=...>`` blocks.

    Logic adapted from sglang Qwen3CoderDetector.detect_and_parse().
    Returns a list of ``{"tool": name, "arguments": {...}}`` dicts.
    """
    if "<tool_call>" not in text:
        return []

    results: list[dict[str, Any]] = []
    raw_tool_calls = _TOOL_CALL_RE.findall(text)
    if not raw_tool_calls:
        # Fallback: maybe the whole text is inside the tag or tags are stripped
        if "<function=" in text:
            raw_tool_calls = [text]

    for tool_content in raw_tool_calls:
        # Find function calls
        funcs = _FUNCTION_RE.findall(tool_content)
        for func_match in funcs:
            func_body = func_match[0] or func_match[1]
            if ">" not in func_body:
                continue

            name_end = func_body.index(">")
            func_name = func_body[:name_end].strip()
            params_str = func_body[name_end + 1:]

            param_config = _get_param_config(func_name)
            parsed_params: dict[str, Any] = {}

            for p_match in _PARAMETER_RE.findall(params_str):
                if ">" not in p_match:
                    continue
                p_idx = p_match.index(">")
                p_name = p_match[:p_idx].strip()
                p_val = p_match[p_idx + 1:]
                # Remove prefixing and trailing \n (same as sglang)
                if p_val.startswith("\n"):
                    p_val = p_val[1:]
                if p_val.endswith("\n"):
                    p_val = p_val[:-1]

                parsed_params[p_name] = _convert_param_value(
                    p_val, p_name, param_config, func_name
                )

            results.append({"tool": func_name, "arguments": parsed_params})

    return results


def parse_tool_calls_response(response: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse policy output into a list of ``{tool, arguments}`` calls.

    Supported payloads (checked in order):
    1. Qwen3 Coder XML format: ``<tool_call><function=...><parameter=...>...</function></tool_call>``
       (uses sglang Qwen3CoderDetector regex patterns)
    2. JSON: ``{"tool_calls": [{"tool": "fs_write", "arguments": {...}}]}``
    3. JSON: ``{"operations": [...]}``
    4. JSON: ``{"fs_ops": [...], "vec_ops": [...], "graph_ops": [...]}``
    """
    text = response.strip() if response else ""

    # Strip thinking content first for all formats
    clean_text = strip_think_wrapper(response) if response else ""

    # 1. Try JSON task-loop payloads first when the whole response is JSON.
    # The JSON may contain raw assistant content with <tool_call> blocks inside
    # agentic_trace; parsing XML first would lose fields like post_consolidate_snapshot.
    parsed = try_parse_json(text) if text.startswith("{") else None
    if parsed is None:
        parsed = try_parse_json(clean_text) if clean_text.startswith("{") else None

    # 2. Try Qwen3 Coder XML format (check both raw and think-stripped text)
    if parsed is None:
        qwen_calls = _parse_qwen_tool_calls(text) or _parse_qwen_tool_calls(clean_text)
        if qwen_calls:
            return qwen_calls, {"tool_calls": qwen_calls, "_format": "qwen3_coder_xml"}

    # 3. Try loose JSON formats
    if parsed is None:
        parsed = try_parse_json(text)
    if parsed is None:
        parsed = try_parse_json(clean_text)
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
    """Lightweight structural reward for tool-call outputs (JSON or Qwen XML)."""
    calls, parsed = parse_tool_calls_response(response)
    if parsed is None:
        return 0.0
    score = 0.3
    if calls:
        score += 0.3
    valid_tools = {
        # File system tools
        "fs_read", "read_file", "fs_read_file", "fs_read_lines", "fs_tree",
        "fs_search", "fs_grep", "fs_bm25_search", "fs_execute_bash",
        "fs_write", "fs_append", "fs_update_line", "fs_update_meta", "fs_delete",
        # Vector tools
        "vec_search", "vec_search_all", "vec_semantic_search", "vec_list_collections",
        "vec_add", "vec_delete", "vec_write",
        # Graph tools
        "graph_search_nodes", "graph_entity_search", "graph_get_neighbors",
        "graph_get_subgraph", "graph_stats", "graph_add_node", "graph_add_edge",
        "graph_delete_node", "graph_delete_edge", "graph_write",
        # Retrieve tools
        "submit",
        # Shared
        "finish",
    }
    well_formed = 0
    for call in calls:
        if call.get("tool") in valid_tools and isinstance(call.get("arguments"), dict):
            well_formed += 1
    if calls:
        score += 0.3 * (well_formed / len(calls))
    if any(call.get("tool") in {"finish", "submit"} for call in calls):
        score += 0.1
    return min(score, 1.0)
