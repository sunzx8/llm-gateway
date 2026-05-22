"""Frozen-model QA prompt construction and answer scoring helpers."""

from __future__ import annotations

import re


def build_qa_prompt(question: str, context: str, ground_truth: dict) -> list[dict[str, str]]:
    """Build a prompt that asks the frozen model to answer with <answer> tags."""
    answer_type = ground_truth.get("answer_type", "fill")

    if answer_type == "mcq":
        options = ground_truth.get("options", [])
        options_text = "\n".join(options)
        user_content = (
            f"Based on the following memory context, answer the question.\n\n"
            f"## Memory Context:\n{context}\n\n"
            f"## Question:\n{question}\n\n"
            f"## Options:\n{options_text}\n\n"
            f"You may explain your reasoning, but you MUST put your final answer "
            f"(the letter only, e.g. A, B, C, or D) inside <answer></answer> tags.\n"
            f"Example: <answer>B</answer>"
        )
    else:
        user_content = (
            f"Based on the following memory context, answer the question.\n\n"
            f"## Memory Context:\n{context}\n\n"
            f"## Question:\n{question}\n\n"
            f"You may explain your reasoning, but you MUST put your final concise answer "
            f"inside <answer></answer> tags.\n"
            f"If the context doesn't contain enough information, put <answer>I don't know</answer>.\n"
            f"Example: <answer>Italian food, especially handmade pasta</answer>"
        )

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the given memory context. Always put your final answer inside <answer></answer> tags.",
        },
        {"role": "user", "content": user_content},
    ]


def extract_answer_tag(text: str) -> str:
    """Extract the final <answer>...</answer> block, falling back to full text."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def score_mcq(model_answer: str, ground_truth: dict) -> float:
    """Score a multiple-choice answer as 1.0 or 0.0."""
    correct_option = ground_truth.get("correct_option", "").upper().strip()
    if not correct_option:
        return 0.0

    extracted = extract_answer_tag(model_answer)
    answer_clean = extracted.strip().upper()

    if answer_clean == correct_option:
        return 1.0

    match = re.match(r"^[(\s]*([A-D])[)\s.\-:]*", answer_clean)
    if match and match.group(1) == correct_option:
        return 1.0

    options_in_answer = set(re.findall(r"\b([A-D])\b", answer_clean))
    if options_in_answer == {correct_option}:
        return 1.0

    return 0.0


def score_fill(model_answer: str, ground_truth: dict) -> float:
    """Score a fill-in answer by exact acceptable-answer hit or keyword coverage."""
    gold_answer = ground_truth.get("answer", "")
    if not gold_answer:
        return 0.0

    extracted = extract_answer_tag(model_answer)
    model_lower = extracted.lower().strip()

    acceptable = ground_truth.get("acceptable_answers", [])
    if not acceptable:
        acceptable = [gold_answer]

    best_score = 0.0
    for answer in acceptable:
        answer_lower = answer.lower().strip()
        if not answer_lower:
            continue

        if answer_lower in model_lower:
            return 1.0

        answer_words = extract_content_words(answer_lower)
        if not answer_words:
            continue
        hits = sum(1 for word in answer_words if word in model_lower)
        best_score = max(best_score, hits / len(answer_words))

    return min(best_score, 1.0)


def extract_content_words(text: str) -> list[str]:
    """Extract content words for approximate fill-answer matching."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "and", "or", "but", "not", "no",
        "it", "its", "this", "that", "i", "my", "me", "we", "you",
        "he", "she", "they", "what", "which", "who", "how",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in stop_words and len(word) > 1]


# Backward-compatible private aliases used by older imports/tests.
_build_qa_prompt = build_qa_prompt
_extract_answer_tag = extract_answer_tag
_score_mcq = score_mcq
_score_fill = score_fill
_extract_content_words = extract_content_words
