"""Parsing helpers for Retrieve RLVR reward responses."""

from __future__ import annotations

import json
import re


def strip_think_wrapper(response: str) -> str:
    """Remove thinking wrappers/special tokens and return the JSON-like payload."""
    text = response.strip()
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
    """Parse a JSON object from raw text or from the first/last brace span."""
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(text[first_brace:last_brace + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


# Backward-compatible private aliases used by older tests/imports.
_strip_think_wrapper = strip_think_wrapper
_try_parse_json = try_parse_json
