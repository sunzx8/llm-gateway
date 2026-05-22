

from .logged_mixin import LoggedMethodsMixin, _logged_method  # noqa: F401


def dump_tree(obj, max_str_len=80, max_list_items=5, max_depth=10) -> str:
    """
    Recursively render a complex object (dict/list/any) in a tree-like format,
    similar to the `du` or `tree` command output.

    Handles non-JSON-serializable objects and circular references gracefully.

    Args:
        obj: The object to display.
        max_str_len: Truncate string values longer than this.
        max_list_items: Show at most this many list items before summarizing.
        max_depth: Maximum recursion depth to prevent stack overflow.

    Returns:
        A multi-line string representing the tree structure.
    """
    lines = []
    seen = {}  # Map object id -> path string, to detect circular refs and report target

    def _truncate(s, limit):
        s = str(s)
        if len(s) > limit:
            return s[:limit] + f"... ({len(s)} chars)"
        return s

    def _is_primitive(v):
        """Check if value is a primitive type that won't cause recursion."""
        return v is None or isinstance(v, (bool, int, float, str, bytes))

    def _walk(obj, child_prefix, depth, path):
        if isinstance(obj, dict):
            items = list(obj.items())
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                branch = "└── " if is_last_item else "├── "
                extension = "    " if is_last_item else "│   "
                child_path = f"{path}.{key}" if path else str(key)

                if _is_primitive(value):
                    display_val = _safe_repr(value)
                    lines.append(f"{child_prefix}{branch}{key} = {display_val}")
                elif isinstance(value, dict):
                    obj_id = id(value)
                    if obj_id in seen:
                        lines.append(f"{child_prefix}{branch}{key}/ <circular ref of {seen[obj_id]}>")
                    elif depth >= max_depth:
                        lines.append(f"{child_prefix}{branch}{key}/ ({len(value)} keys) <max depth>")
                    else:
                        seen[obj_id] = child_path
                        lines.append(f"{child_prefix}{branch}{key}/ ({len(value)} keys)")
                        _walk(value, child_prefix + extension, depth + 1, child_path)
                elif isinstance(value, (list, tuple)):
                    obj_id = id(value)
                    if obj_id in seen:
                        lines.append(f"{child_prefix}{branch}{key}[] <circular ref of {seen[obj_id]}>")
                    elif depth >= max_depth:
                        lines.append(f"{child_prefix}{branch}{key}[] ({len(value)} items) <max depth>")
                    else:
                        seen[obj_id] = child_path
                        lines.append(f"{child_prefix}{branch}{key}[] ({len(value)} items)")
                        _walk_list(value, child_prefix + extension, depth + 1, child_path)
                else:
                    display_val = _safe_repr(value)
                    lines.append(f"{child_prefix}{branch}{key} = {display_val}")

        elif isinstance(obj, (list, tuple)):
            _walk_list(obj, child_prefix, depth, path)

    def _walk_list(lst, prefix, depth, path):
        total = len(lst)
        show_count = min(total, max_list_items)
        for i in range(show_count):
            is_last_item = (i == show_count - 1) and (total <= max_list_items)
            branch = "└── " if is_last_item else "├── "
            extension = "    " if is_last_item else "│   "
            child_path = f"{path}[{i}]"

            item = lst[i]
            if _is_primitive(item):
                display_val = _safe_repr(item)
                lines.append(f"{prefix}{branch}[{i}] = {display_val}")
            elif isinstance(item, dict):
                obj_id = id(item)
                if obj_id in seen:
                    lines.append(f"{prefix}{branch}[{i}]/ <circular ref of {seen[obj_id]}>")
                elif depth >= max_depth:
                    lines.append(f"{prefix}{branch}[{i}]/ ({len(item)} keys) <max depth>")
                else:
                    seen[obj_id] = child_path
                    lines.append(f"{prefix}{branch}[{i}]/ ({len(item)} keys)")
                    _walk(item, prefix + extension, depth + 1, child_path)
            elif isinstance(item, (list, tuple)):
                obj_id = id(item)
                if obj_id in seen:
                    lines.append(f"{prefix}{branch}[{i}][] <circular ref of {seen[obj_id]}>")
                elif depth >= max_depth:
                    lines.append(f"{prefix}{branch}[{i}][] ({len(item)} items) <max depth>")
                else:
                    seen[obj_id] = child_path
                    lines.append(f"{prefix}{branch}[{i}][] ({len(item)} items)")
                    _walk_list(item, prefix + extension, depth + 1, child_path)
            else:
                display_val = _safe_repr(item)
                lines.append(f"{prefix}{branch}[{i}] = {display_val}")

        if total > max_list_items:
            lines.append(f"{prefix}└── ... and {total - max_list_items} more items")

    def _safe_repr(value):
        """Safely represent a value, handling non-serializable objects."""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return _truncate(repr(value), max_str_len)
        elif isinstance(value, bytes):
            return _truncate(repr(value), max_str_len)
        else:
            # For complex objects, use a shallow repr with recursion guard
            try:
                type_name = type(value).__name__
                # Avoid calling repr on objects that might trigger deep recursion
                if hasattr(value, '__dict__'):
                    attrs = list(vars(value).keys())[:5]
                    attr_str = ", ".join(attrs)
                    if len(vars(value)) > 5:
                        attr_str += ", ..."
                    return f"<{type_name}({attr_str})>"
                else:
                    r = repr(value)
                    return _truncate(r, max_str_len)
            except Exception:
                return f"<{type(value).__name__}: unrepresentable>"

    # Root node - depth starts at 1 so max_depth=1 means only show root's direct children
    if isinstance(obj, dict):
        seen[id(obj)] = "."
        lines.append(f"./ ({len(obj)} keys)")
        _walk(obj, "", 1, "")
    elif isinstance(obj, (list, tuple)):
        seen[id(obj)] = "."
        lines.append(f"[] ({len(obj)} items)")
        _walk_list(obj, "", 1, "")
    else:
        lines.append(_safe_repr(obj))

    return "\n".join(lines)
