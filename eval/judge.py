"""LLM-as-judge scorer for open-ended evaluation.

Supports three scoring modes:
1. Generic 1-5 scale scoring (for general use)
2. LongMemEval-aligned yes/no answer checking (per question_type prompt templates)
3. LOCOMO multi-dimensional scoring (factual accuracy, relevance, completeness,
   contextual appropriateness) — aligned with Chhikara et al. (2025) / APEX-MEM
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from config.models import LLMConfig
from utils.memory_llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    score: float  # 1-5 scale or 1.0/0.0 for yes/no
    label: bool = False  # True = correct, False = incorrect (for yes/no mode)
    reasoning: str = ""
    raw_output: str = ""


@dataclass
class LocomoJudgeResult:
    """Multi-dimensional judge result for LOCOMO evaluation.

    Following Chhikara et al. (2025), evaluates on 4 dimensions:
    - factual_accuracy: correctness of factual claims (1-5)
    - relevance: how relevant the response is to the question (1-5)
    - completeness: whether all required information is present (1-5)
    - contextual_appropriateness: proper use of conversational context (1-5)

    The overall score is the mean of all 4 dimensions (1-5 scale).
    label=True if overall >= 4.0 (considered "correct").
    """

    factual_accuracy: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    contextual_appropriateness: float = 0.0
    overall: float = 0.0  # 均值 (1-5)
    score: float = 0.0  # 归一化到 0-1 (for accuracy: 1.0 if overall >= 4.0)
    label: bool = False
    reasoning: str = ""
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Generic 1-5 scale judge (for general use)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's response.
Rate the response on a scale of 1-5 based on the following criteria:

1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat correct but missing key information
4 = Mostly correct with minor issues
5 = Fully correct, comprehensive, and well-formatted

You MUST output in this exact format:
Score: <number 1-5>
Reasoning: <brief explanation>
"""


# ---------------------------------------------------------------------------
# LongMemEval-aligned yes/no judge (per question_type prompt templates)
# Directly aligned with: LongMemEval/src/evaluation/evaluate_qa.py
# ---------------------------------------------------------------------------

LONGMEMEVAL_PROMPTS = {
    "single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the answer, answer no."
    ),
    "single-session-assistant": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the answer, answer no."
    ),
    "multi-session": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the answer, answer no."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), "
        "the model's response is still correct."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer "
        "is the required answer."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, "
        "and a response from a model. Please answer yes if the response satisfies "
        "the desired response. Otherwise, answer no. The model does not need to "
        "reflect all the points in the rubric. The response is correct as long as "
        "it recalls and utilizes the user's personal information correctly."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information "
        "is given but the asked information is not."
    ),
}

# ---------------------------------------------------------------------------
# LoCoMo-specific yes/no judge prompts (per question category)
#
# LoCoMo question_types: single-hop, multi-hop, temporal, open-ended, adversarial
# These prompts are tailored to each category's evaluation semantics.
# ---------------------------------------------------------------------------

LOCOMO_PROMPTS = {
    "single-hop": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the key "
        "information of the correct answer, you should also answer yes. "
        "Minor wording differences are acceptable as long as the meaning is preserved."
    ),
    "multi-hop": (
        "I will give you a question, a correct answer, and a response from a model. "
        "The question requires combining information from multiple parts of a conversation. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes."
    ),
    "temporal": (
        "I will give you a question, a correct answer, and a response from a model. "
        "The question involves temporal reasoning about events or changes over time. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer, you should also answer yes. "
        "In addition, do not penalize off-by-one errors for the number of days/weeks/months. "
        "If the correct answer includes multiple temporal states separated by ';', the model "
        "only needs to get the most recent/relevant state correct."
    ),
    "open-ended": (
        "I will give you a question, a reference answer, and a response from a model. "
        "The question is open-ended and may have multiple valid ways to answer. "
        "Please answer yes if the response demonstrates correct recall of the relevant "
        "personal information and provides a reasonable answer. Otherwise, answer no. "
        "The model does not need to match the reference word-for-word. It is correct "
        "as long as it utilizes the user's personal context accurately."
    ),
    "adversarial": (
        "I will give you a question, an explanation, and a response from a model. "
        "The question is adversarial — it asks about something that was NOT actually "
        "discussed in the conversation or contains false premises. "
        "Please answer yes if the model correctly identifies the question as unanswerable, "
        "refuses to answer, or indicates that the information was not discussed. "
        "Answer no if the model fabricates an answer or confidently provides incorrect information."
    ),
}

_LOCOMO_DEFAULT_PROMPT = LOCOMO_PROMPTS["single-hop"]

# Fallback for unknown question types
_DEFAULT_PROMPT = LONGMEMEVAL_PROMPTS["single-session-user"]


# ---------------------------------------------------------------------------
# LoCoMo multi-dimensional judge (aligned with Chhikara et al. 2025 / APEX-MEM)
#
# Instead of simple yes/no, evaluates on 4 dimensions:
#   1. Factual Accuracy — correctness of factual claims
#   2. Relevance — how relevant the response is to the question
#   3. Completeness — whether all required information is present
#   4. Contextual Appropriateness — proper use of conversational context
#
# Reference: arXiv:2603.02473 §6.1 (APEX-MEM evaluation methodology)
# ---------------------------------------------------------------------------

LOCOMO_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's response \
to a question about a personal conversation history.

You MUST evaluate the response on exactly 4 dimensions, each on a 1-5 scale:

1. **Factual Accuracy** (1-5): Are the factual claims in the response correct \
compared to the reference answer? Does it avoid hallucinating information?
   - 1 = Completely wrong or fabricated
   - 2 = Mostly incorrect with some correct elements
   - 3 = Partially correct but contains significant errors
   - 4 = Mostly correct with minor inaccuracies
   - 5 = Fully correct, all facts match the reference

2. **Relevance** (1-5): Is the response relevant to the question asked? \
Does it address what was actually asked?
   - 1 = Completely irrelevant or off-topic
   - 2 = Tangentially related but misses the point
   - 3 = Somewhat relevant but includes much irrelevant content
   - 4 = Mostly relevant with minor digressions
   - 5 = Directly and precisely addresses the question

3. **Completeness** (1-5): Does the response include all the key information \
from the reference answer? Are there missing details?
   - 1 = Missing all key information
   - 2 = Contains only a small fraction of required information
   - 3 = Contains some key information but misses important parts
   - 4 = Contains most key information with minor omissions
   - 5 = Complete — all key information from the reference is present

4. **Contextual Appropriateness** (1-5): Does the response properly use the \
conversational context? Does it correctly attribute information to the right \
person/time/event?
   - 1 = Completely misuses or ignores context
   - 2 = Major context errors (wrong person, wrong time, etc.)
   - 3 = Some context errors but partially correct
   - 4 = Mostly appropriate context usage with minor issues
   - 5 = Perfect context usage — correct attribution and temporal awareness

You MUST output in this EXACT format (one line per field):
Factual Accuracy: <1-5>
Relevance: <1-5>
Completeness: <1-5>
Contextual Appropriateness: <1-5>
Reasoning: <brief explanation of your ratings>
"""

# Per-category additional instructions for LOCOMO multi-dimensional judge
LOCOMO_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "single-hop": (
        "This is a single-hop factual question that requires recalling a specific "
        "piece of information from the conversation. Focus especially on Factual "
        "Accuracy — the response should contain the exact correct answer. "
        "Minor wording differences are acceptable."
    ),
    "multi-hop": (
        "This is a multi-hop question that requires combining information from "
        "multiple parts of the conversation. Pay attention to both Factual Accuracy "
        "and Completeness — the response should correctly synthesize all relevant "
        "pieces of information. If the reference contains comma-separated answers, "
        "all parts should be present."
    ),
    "temporal": (
        "This is a temporal reasoning question about events or changes over time. "
        "Focus on Factual Accuracy and Contextual Appropriateness — the response "
        "should correctly identify the temporal state (most recent if multiple exist). "
        "Do NOT penalize off-by-one errors for counts of days/weeks/months."
    ),
    "open-ended": (
        "This is an open-ended question that may have multiple valid ways to answer. "
        "Focus on Relevance and Contextual Appropriateness — the response should "
        "demonstrate correct recall of personal information and provide a reasonable "
        "answer. It does NOT need to match the reference word-for-word."
    ),
    "adversarial": (
        "This is an adversarial question — it asks about something NOT actually "
        "discussed in the conversation or contains false premises. "
        "The CORRECT behavior is to refuse to answer or indicate the information "
        "was not discussed. Score Factual Accuracy=5 if the model correctly refuses; "
        "Score Factual Accuracy=1 if it fabricates an answer. "
        "Relevance=5 if it addresses why it cannot answer; Completeness=5 if it "
        "correctly identifies the question as unanswerable; "
        "Contextual Appropriateness=5 if it does not hallucinate context."
    ),
}

_LOCOMO_DEFAULT_INSTRUCTION = LOCOMO_CATEGORY_INSTRUCTIONS["single-hop"]


class LLMJudge:
    """LLM-based judge supporting multiple scoring modes.

    Modes:
    - Generic 1-5 scale scoring (score method)
    - LongMemEval yes/no answer checking (score_yesno method)
    - LOCOMO multi-dimensional scoring (score_locomo method)
    """

    def __init__(self, llm_config: LLMConfig | None = None):
        config = llm_config or LLMConfig()
        self.llm = LLMInterface(config)

    # ------------------------------------------------------------------
    # Generic 1-5 scoring
    # ------------------------------------------------------------------

    async def score(self, question: str, reference: str, prediction: str) -> JudgeResult:
        """Score a single prediction on a 1-5 scale."""
        user_msg = (
            f"## Question:\n{question}\n\n"
            f"## Reference Answer:\n{reference}\n\n"
            f"## AI Response to Evaluate:\n{prediction}\n\n"
            "Please rate the AI response. Output:\nScore: <1-5>\nReasoning: <explanation>"
        )
        try:
            response = await self.llm.generate(
                JUDGE_SYSTEM_PROMPT,
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            return self._parse_score_output(response.content)
        except Exception as e:
            logger.warning("Judge scoring failed: %s", e)
            return JudgeResult(score=0.0, reasoning=f"Error: {e}")

    async def score_batch(
        self,
        questions: list[str],
        references: list[str],
        predictions: list[str],
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[JudgeResult]:
        """Score a batch of predictions (1-5 scale).

        Args:
            semaphore: Optional semaphore to limit concurrent LLM calls.
                       If None, runs sequentially for backward compatibility.
        """
        if semaphore is not None:
            async def _score_one(idx: int, q: str, r: str, p: str) -> tuple[int, JudgeResult]:
                async with semaphore:
                    result = await self.score(q, r, p)
                    return idx, result

            tasks = [
                _score_one(i, q, r, p)
                for i, (q, r, p) in enumerate(zip(questions, references, predictions))
            ]
            indexed_results = await asyncio.gather(*tasks)
            indexed_results.sort(key=lambda x: x[0])
            return [r for _, r in indexed_results]
        else:
            results = []
            for q, r, p in zip(questions, references, predictions):
                result = await self.score(q, r, p)
                results.append(result)
            return results

    # ------------------------------------------------------------------
    # Yes/no judging (supports LongMemEval & LoCoMo prompt sets)
    # ------------------------------------------------------------------

    async def score_yesno(
        self,
        question: str,
        reference: str,
        prediction: str,
        question_type: str = "",
        prompt_set: str = "longmemeval",
    ) -> JudgeResult:
        """Score using yes/no answer checking with benchmark-specific prompts.

        Args:
            question_type: The question category (e.g. "single-hop", "temporal-reasoning").
            prompt_set: Which prompt templates to use.
                - "longmemeval" (default): LongMemEval-aligned prompts.
                - "locomo": LoCoMo category-specific prompts.

        Returns:
            JudgeResult with label=True/False and score=1.0/0.0.
        """
        # Select prompt template set
        if prompt_set == "locomo":
            prompts = LOCOMO_PROMPTS
            default_prompt = _LOCOMO_DEFAULT_PROMPT
        else:
            prompts = LONGMEMEVAL_PROMPTS
            default_prompt = _DEFAULT_PROMPT

        # Detect abstention-style questions
        is_abstention = question_type in ("abstention", "adversarial")

        # Get the appropriate prompt template
        if question_type == "abstention":
            template = LONGMEMEVAL_PROMPTS["abstention"]
        elif question_type == "adversarial":
            template = prompts.get("adversarial", default_prompt)
        else:
            template = prompts.get(question_type, default_prompt)

        prompt = (
            f"{template}\n\n"
            f"Question: {question}\n\n"
            f"{'Explanation' if is_abstention else 'Correct Answer'}: {reference}\n\n"
            f"Model Response: {prediction}\n\n"
            f"{'Does the model correctly identify the question as unanswerable or refuse to fabricate an answer?' if is_abstention else 'Is the model response correct?'} "
            f"Answer yes or no only."
        )

        try:
            response = await self.llm.generate(
                "You are an evaluation assistant. Answer yes or no only.",
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                # Reasoning models (GLM-5.1, DeepSeek-R1, etc.) need more tokens
                # because they output reasoning_content before the final answer.
                # Official LongMemEval uses max_tokens=10 with GPT-4o (non-reasoning),
                # but we need ~512 to handle reasoning models safely.
                max_tokens=512,
            )
            raw = response.content.strip()
            label = "yes" in raw.lower()
            logger.debug("Judge yes/no [%s]: raw=%r → label=%s", prompt_set, raw[:80], label)
            return JudgeResult(
                score=1.0 if label else 0.0,
                label=label,
                reasoning="",
                raw_output=raw,
            )
        except Exception as e:
            logger.warning("Yes/no judge failed: %s", e)
            return JudgeResult(score=0.0, label=False, reasoning=f"Error: {e}")

    async def score_yesno_batch(
        self,
        questions: list[str],
        references: list[str],
        predictions: list[str],
        question_types: list[str],
        semaphore: asyncio.Semaphore | None = None,
        prompt_set: str = "longmemeval",
        progress_callback=None,
    ) -> list[JudgeResult]:
        """Score a batch with yes/no judging.

        Args:
            semaphore: Optional semaphore to limit concurrent LLM calls.
                       If None, runs sequentially for backward compatibility.
            prompt_set: "longmemeval" or "locomo" — selects prompt templates.
            progress_callback: Optional callable(done_count) called after each item completes.
        """
        total = len(questions)
        done_count = 0

        if semaphore is not None:
            async def _judge_one(idx: int, q: str, r: str, p: str, qt: str) -> tuple[int, JudgeResult]:
                nonlocal done_count
                async with semaphore:
                    logger.info("  Judge %d/%d: type=%s, question=%s...", idx + 1, total, qt, q[:50])
                    result = await self.score_yesno(q, r, p, qt, prompt_set=prompt_set)
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count)
                    return idx, result

            tasks = [
                _judge_one(i, q, r, p, qt)
                for i, (q, r, p, qt) in enumerate(zip(questions, references, predictions, question_types))
            ]
            indexed_results = await asyncio.gather(*tasks)
            indexed_results.sort(key=lambda x: x[0])
            return [r for _, r in indexed_results]
        else:
            results = []
            for i, (q, r, p, qt) in enumerate(zip(questions, references, predictions, question_types)):
                logger.info("  Judge %d/%d: type=%s, question=%s...", i + 1, total, qt, q[:50])
                result = await self.score_yesno(q, r, p, qt, prompt_set=prompt_set)
                results.append(result)
                done_count += 1
                if progress_callback:
                    progress_callback(done_count)
            return results

    # ------------------------------------------------------------------
    # LOCOMO multi-dimensional scoring
    # (aligned with Chhikara et al. 2025 / APEX-MEM arXiv:2603.02473)
    # ------------------------------------------------------------------

    async def score_locomo(
        self,
        question: str,
        reference: str,
        prediction: str,
        question_type: str = "",
    ) -> LocomoJudgeResult:
        """Score a LOCOMO prediction on 4 dimensions (1-5 each).

        Dimensions: factual_accuracy, relevance, completeness, contextual_appropriateness.
        Overall = mean of 4 dimensions. label=True if overall >= 4.0.

        For adversarial questions, the reference is treated as an explanation
        of why the question is unanswerable.

        Args:
            question: The question text.
            reference: The ground-truth answer (or explanation for adversarial).
            prediction: The model's response to evaluate.
            question_type: LOCOMO category (single-hop, multi-hop, temporal, open-ended, adversarial).

        Returns:
            LocomoJudgeResult with per-dimension scores and overall accuracy.
        """
        is_adversarial = question_type == "adversarial"
        category_instruction = LOCOMO_CATEGORY_INSTRUCTIONS.get(
            question_type, _LOCOMO_DEFAULT_INSTRUCTION
        )

        user_msg = (
            f"## Category-Specific Guidance\n{category_instruction}\n\n"
            f"## Question:\n{question}\n\n"
            f"## {'Explanation (why this question is unanswerable)' if is_adversarial else 'Reference Answer'}:\n{reference}\n\n"
            f"## Model Response to Evaluate:\n{prediction}\n\n"
            "Please evaluate the model response on all 4 dimensions. "
            "Output your ratings in the exact format specified."
        )

        try:
            response = await self.llm.generate(
                LOCOMO_JUDGE_SYSTEM_PROMPT,
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=512,
            )
            return self._parse_locomo_output(response.content)
        except Exception as e:
            logger.warning("LOCOMO multi-dim judge failed: %s", e)
            return LocomoJudgeResult(reasoning=f"Error: {e}", raw_output="")

    async def score_locomo_batch(
        self,
        questions: list[str],
        references: list[str],
        predictions: list[str],
        question_types: list[str],
        semaphore: asyncio.Semaphore | None = None,
        progress_callback=None,
    ) -> list[LocomoJudgeResult]:
        """Score a batch of LOCOMO predictions with multi-dimensional judging.

        Args:
            semaphore: Optional semaphore to limit concurrent LLM calls.
                       If None, runs sequentially for backward compatibility.
            progress_callback: Optional callable(done_count) called after each item completes.
        """
        total = len(questions)
        done_count = 0

        if semaphore is not None:
            async def _judge_one(idx: int, q: str, r: str, p: str, qt: str) -> tuple[int, LocomoJudgeResult]:
                nonlocal done_count
                async with semaphore:
                    logger.info("  LOCOMO Judge %d/%d: type=%s, question=%s...", idx + 1, total, qt, q[:50])
                    result = await self.score_locomo(q, r, p, qt)
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count)
                    return idx, result

            tasks = [
                _judge_one(i, q, r, p, qt)
                for i, (q, r, p, qt) in enumerate(zip(questions, references, predictions, question_types))
            ]
            indexed_results = await asyncio.gather(*tasks)
            indexed_results.sort(key=lambda x: x[0])
            return [r for _, r in indexed_results]
        else:
            results = []
            for i, (q, r, p, qt) in enumerate(zip(questions, references, predictions, question_types)):
                logger.info("  LOCOMO Judge %d/%d: type=%s, question=%s...", i + 1, total, qt, q[:50])
                result = await self.score_locomo(q, r, p, qt)
                results.append(result)
                done_count += 1
                if progress_callback:
                    progress_callback(done_count)
            return results

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score_output(output: str) -> JudgeResult:
        """Parse the judge's 1-5 score output."""
        score = 0.0
        reasoning = ""

        score_match = re.search(r"Score:\s*(\d(?:\.\d)?)", output)
        if score_match:
            score = float(score_match.group(1))
            score = max(1.0, min(5.0, score))

        reason_match = re.search(r"Reasoning:\s*(.+)", output, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return JudgeResult(score=score, label=score >= 4.0, reasoning=reasoning, raw_output=output)

    @staticmethod
    def _parse_locomo_output(output: str) -> LocomoJudgeResult:
        """Parse the LOCOMO multi-dimensional judge output.

        Expected format:
            Factual Accuracy: <1-5>
            Relevance: <1-5>
            Completeness: <1-5>
            Contextual Appropriateness: <1-5>
            Reasoning: <text>
        """
        def _extract(pattern: str, text: str) -> float:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                return max(1.0, min(5.0, val))
            return 0.0

        fa = _extract(r"Factual\s*Accuracy\s*:\s*(\d(?:\.\d)?)", output)
        rel = _extract(r"Relevance\s*:\s*(\d(?:\.\d)?)", output)
        comp = _extract(r"Completeness\s*:\s*(\d(?:\.\d)?)", output)
        ca = _extract(r"Contextual\s*Appropriateness\s*:\s*(\d(?:\.\d)?)", output)

        # 如果解析完全失败，尝试回退到通用 Score 解析
        dims = [fa, rel, comp, ca]
        valid_dims = [d for d in dims if d > 0]

        if valid_dims:
            overall = sum(valid_dims) / len(valid_dims)
        else:
            # 回退：尝试解析单一 Score 字段
            fallback = re.search(r"Score\s*:\s*(\d(?:\.\d)?)", output)
            overall = float(fallback.group(1)) if fallback else 0.0
            overall = max(0.0, min(5.0, overall))

        # label=True if overall >= 4.0 (对应"正确")
        label = overall >= 4.0
        # score 归一化到 0/1 用于准确率计算
        score = 1.0 if label else 0.0

        reasoning = ""
        reason_match = re.search(r"Reasoning\s*:\s*(.+)", output, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return LocomoJudgeResult(
            factual_accuracy=fa,
            relevance=rel,
            completeness=comp,
            contextual_appropriateness=ca,
            overall=overall,
            score=score,
            label=label,
            reasoning=reasoning,
            raw_output=output,
        )


