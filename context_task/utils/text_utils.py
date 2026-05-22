"""文本处理工具函数。"""

from __future__ import annotations


def truncate_with_hint(
    text: str,
    max_chars: int,
    file_path: str = "",
) -> str:
    """截断文本并在末尾添加截断提示，帮助模型感知内容不完整。

    如果文本长度未超过 max_chars，则原样返回。
    如果超过，则截断到 max_chars 并追加一段提示信息，
    告知模型内容被截断以及完整文件的路径。

    Args:
        text: 原始文本内容。
        max_chars: 最大保留字符数。
        file_path: 完整文件的路径，用于提示模型通过工具读取。

    Returns:
        截断后带提示的文本，或原始文本（未超长时）。
    """
    if not text or len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    hint_parts = [
        f"\n\n... [⚠️ 内容已截断：原文共 {len(text)} 字符，此处仅展示前 {max_chars} 字符]",
    ]
    if file_path:
        hint_parts.append(
            f"请使用 `fs_read` 工具读取完整文件：`{file_path}`"
        )
    hint_parts.append("...")

    return truncated + " ".join(hint_parts)
