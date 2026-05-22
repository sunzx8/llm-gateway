"""PersonaMem benchmark adapter.

PersonaMem: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses
- 20 user personas, 5990 QA pairs across 3 context lengths (32k / 128k / 1M)
- MCQ format (4-option: a/b/c/d)
- 7 question types covering recall, preference tracking, recommendation, generalization
- HuggingFace: https://huggingface.co/datasets/bowen-upenn/PersonaMem
- Paper: arXiv:2504.14225

Data format:
    data_dir/
    ├── questions_32k.csv          # 589 questions (shortest context)
    ├── shared_contexts_32k.jsonl  # Shared conversation contexts
    ├── questions_128k.csv         # 2730 questions
    ├── shared_contexts_128k.jsonl
    ├── questions_1M.csv           # 2670 questions
    └── shared_contexts_1M.jsonl

CSV fields:
    persona_id, question_id, question_type, topic, context_length_in_tokens,
    user_question_or_message, correct_answer (a/b/c/d), all_options (JSON list),
    shared_context_id, end_index_in_shared_context, ...
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from benchmarks.base_benchmark import BaseBenchmark, Conversation, Session, UserTrajectory, Message, QAPair

logger = logging.getLogger(__name__)

DEFAULT_SPLIT = "32k"
VALID_SPLITS = ("32k", "128k", "1M")


class PersonaMemAdapter(BaseBenchmark):
    """Adapter for the PersonaMem benchmark dataset.

    PersonaMem evaluates LLMs' ability to:
    1. Remember user-shared facts
    2. Track preference evolution
    3. Generate personalized recommendations
    4. Generalize to new scenarios

    Args:
        data_dir: Path to the PersonaMem data directory.
        split: Context length split to use: "32k", "128k", or "1M".
            Defaults to "32k" (smallest, fastest for testing).
    """

    name = "personamem"

    def __init__(self, data_dir: str, split: str = DEFAULT_SPLIT):
        super().__init__(data_dir)
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}', must be one of {VALID_SPLITS}")
        self.split = split

    def load_data(self) -> list[Conversation]:
        """Load PersonaMem dataset.

        Loads questions from CSV and shared contexts from JSONL,
        then groups by persona_id into Conversations.
        """
        data_path = Path(self.data_dir)

        # --- Load questions ---
        questions_file = data_path / f"questions_{self.split}.csv"
        if not questions_file.exists():
            # Fallback: try legacy combined format
            legacy = data_path / "personamem.json"
            if legacy.exists():
                logger.info("Using legacy personamem.json format")
                with open(legacy) as f:
                    return self._parse_legacy_combined(json.load(f))
            raise FileNotFoundError(
                f"PersonaMem data not found: {questions_file}\n"
                f"Run 'bash scripts/setup_benchmarks.sh' to download, "
                f"or see https://huggingface.co/datasets/bowen-upenn/PersonaMem"
            )

        questions_by_persona: dict[str, list[dict]] = {}
        with open(questions_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("persona_id", ""))
                questions_by_persona.setdefault(pid, []).append(row)

        logger.info(
            "Loaded %d questions for %d personas from %s",
            sum(len(qs) for qs in questions_by_persona.values()),
            len(questions_by_persona),
            questions_file.name,
        )

        # --- Load shared contexts ---
        # Format: each JSONL line is {"<sha256_hash>": [{"role": "...", "content": "..."}, ...]}
        # The hash key is the shared_context_id referenced in the CSV
        contexts_file = data_path / f"shared_contexts_{self.split}.jsonl"
        shared_contexts: dict[str, Any] = {}
        if contexts_file.exists():
            with open(contexts_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        for ctx_id, ctx_value in item.items():
                            # ctx_id is the sha256 hash, ctx_value is the message list
                            shared_contexts[str(ctx_id)] = ctx_value
            logger.info(
                "Loaded %d shared contexts from %s",
                len(shared_contexts), contexts_file.name,
            )
        else:
            logger.warning("Shared contexts file not found: %s", contexts_file)

        # --- Build Conversations (one per persona × context) ---
        # Each shared_context_id becomes an independent Conversation to prevent
        # cross-context memory pollution. Different contexts for the same persona
        # are "forked" parallel worlds (they share a prefix but diverge), so their
        # memories must NOT accumulate.
        conversations: list[Conversation] = []
        for persona_id, question_rows in sorted(questions_by_persona.items()):
            convs = self._build_conversations(persona_id, question_rows, shared_contexts)
            conversations.extend(convs)

        logger.info("Built %d PersonaMem conversations (from %d personas)",
                    len(conversations), len(questions_by_persona))
        return conversations

    def _build_conversations(
        self,
        persona_id: str,
        question_rows: list[dict],
        shared_contexts: dict[str, Any],
    ) -> list[Conversation]:
        """Build Conversations for one persona — one per shared_context_id.

        PersonaMem data structure:
        - Each persona may have 1-2 shared contexts ("forked" parallel worlds)
        - Same persona's different contexts share a prefix but diverge at some point
        - Memory from one context must NOT pollute another context's answers

        Therefore each (persona_id, shared_context_id) pair becomes an independent
        Conversation with its own user_id, ensuring complete memory isolation.

        Within each context:
        - Questions have `end_index_in_shared_context` indicating how much context to see
        - We split the context into incremental sessions by unique end_index
        - Each session contains INCREMENTAL messages (prev_end_index:current_end_index)
        - Questions are attached to the session corresponding to their end_index
        - The runner ingests sessions in order, answering attached questions after each
        """
        # Group questions by shared_context_id
        questions_by_ctx: dict[str, list[dict]] = {}
        for row in question_rows:
            ctx_id = str(row.get("shared_context_id", ""))
            if ctx_id:
                questions_by_ctx.setdefault(ctx_id, []).append(row)

        conversations: list[Conversation] = []
        ctx_counter = 0

        for ctx_id, ctx_questions in questions_by_ctx.items():
            ctx_data = shared_contexts.get(ctx_id)
            if not ctx_data:
                logger.warning("Shared context %s not found for persona %s", ctx_id[:16], persona_id)
                continue

            # Each (persona, context) gets isolated memory via the three-level
            # branch structure: users/{user_id}/{context_id}/master
            # user_id = persona_id (shared across contexts for the same persona)
            # context_id = ctx{N} (unique per context, ensures isolation)
            ctx_short = ctx_id[:8]
            conv_id = f"persona_{persona_id}_ctx{ctx_counter}"
            user_id = str(persona_id)
            context_id = f"ctx{ctx_counter}"

            conv = Conversation(
                conv_id=conv_id,
                metadata={
                    "user_id": user_id,
                    "context_id": context_id,
                    "persona_id": persona_id,
                    "shared_context_id": ctx_id,
                    "context_index": ctx_counter,
                },
            )

            # Group questions by end_index within this context
            endidx_questions: dict[int, list[dict]] = {}
            for row in ctx_questions:
                end_idx = int(row.get("end_index_in_shared_context", 0) or 0)
                endidx_questions.setdefault(end_idx, []).append(row)

            # Get all unique end_indices, sorted ascending
            sorted_end_indices = sorted(endidx_questions.keys())

            # Build incremental sessions: each session covers [prev_end_index, current_end_index)
            prev_end_idx = 0
            sess_counter = 0
            for end_idx in sorted_end_indices:
                # Extract incremental messages for this segment
                messages = self._extract_messages_range(ctx_data, prev_end_idx, end_idx)
                if not messages and prev_end_idx == 0:
                    # First segment with no messages — skip
                    continue

                session = Session(
                    session_id=f"{persona_id}_ctx{ctx_counter}_s{sess_counter}",
                    metadata={
                        "shared_context_id": ctx_id,
                        "end_index": end_idx,
                        "prev_end_index": prev_end_idx,
                        "is_incremental": prev_end_idx > 0,
                    },
                )
                session.messages = messages

                # Build QA pairs for this end_index and attach to session
                for row in endidx_questions[end_idx]:
                    qa = self._build_qa_pair(row, persona_id)
                    conv.questions.append(qa)
                    session.questions.append(qa)

                conv.sessions.append(session)
                prev_end_idx = end_idx
                sess_counter += 1

            # If no sessions were created, create a dummy
            if not conv.sessions:
                session = Session(session_id=f"{persona_id}_ctx{ctx_counter}_s0")
                session.messages = [Message(
                    role="user",
                    content=f"(Persona {persona_id} context {ctx_short} not loaded)",
                )]
                conv.sessions.append(session)
                for row in ctx_questions:
                    qa = self._build_qa_pair(row, persona_id)
                    conv.questions.append(qa)
                    session.questions.append(qa)

            if conv.questions:
                conversations.append(conv)
                logger.debug(
                    "Persona %s ctx%d (%s): %d sessions, %d questions",
                    persona_id, ctx_counter, ctx_short,
                    len(conv.sessions), len(conv.questions),
                )

            ctx_counter += 1

        # Fallback: if no contexts found at all, create a dummy conversation
        if not conversations:
            conv = Conversation(
                conv_id=f"persona_{persona_id}",
                metadata={"user_id": persona_id, "persona_id": persona_id},
            )
            session = Session(session_id=f"{persona_id}_s0")
            session.messages = [Message(
                role="user",
                content=f"(Persona {persona_id} conversation context not loaded)",
            )]
            conv.sessions.append(session)
            for row in question_rows:
                qa = self._build_qa_pair(row, persona_id)
                conv.questions.append(qa)
                session.questions.append(qa)
            if conv.questions:
                conversations.append(conv)

        return conversations

    def _build_qa_pair(self, row: dict, persona_id: str) -> QAPair:
        """Build a QAPair from a CSV row.

        Aligned with official PersonaMem evaluation protocol:
        - question = user_question_or_message + all_options (raw CSV string)
        - The official code combines: question + '\\n\\n' + instructions + '\\n\\n' + all_options
        - We put question + all_options in the question field; instructions are added by each baseline
        - all_options is kept as the raw CSV string (JSON list format), not parsed/reformatted
        """
        question_text = row.get("user_question_or_message", "")
        correct = row.get("correct_answer", "")
        # Keep all_options as the raw CSV string — official code passes it directly
        # to the LLM without JSON parsing or reformatting.
        options_raw = row.get("all_options", "[]")

        # Combine question + options in the official format:
        #   question + '\n\n' + all_options
        # Instructions are NOT included here — each baseline adds its own.
        formatted_q = f"{question_text}\n\n{options_raw}"

        return QAPair(
            question=formatted_q,
            reference_answer=correct,
            question_type=row.get("question_type", "mcq"),
            metadata={
                "question_id": row.get("question_id", ""),
                "topic": row.get("topic", ""),
                "persona_id": persona_id,
                "correct_answer": correct,
                "all_options_raw": options_raw,  # Raw CSV string for official protocol
                "context_length_in_tokens": int(row.get("context_length_in_tokens", 0) or 0),
                "shared_context_id": row.get("shared_context_id", ""),
                "end_index_in_shared_context": int(row.get("end_index_in_shared_context", 0) or 0),
            },
            # Pure question text without options — used for memory retrieval
            # to avoid option text polluting embedding/BM25/entity search signals.
            question_for_retrieval=question_text,
        )

    @staticmethod
    def _extract_messages_range(ctx_data: Any, start_idx: int, end_idx: int) -> list[Message]:
        """Extract messages from a shared context entry for a specific range.

        Preserves the original message roles (system/user/assistant) as-is,
        matching the official PersonaMem evaluation protocol where context
        messages are passed directly as chat messages to the LLM.

        Args:
            ctx_data: The context data — a list of {role, content} message dicts.
            start_idx: Start index (inclusive).
            end_idx: End index (exclusive). The context for this segment is
                ctx_data[start_idx:end_idx].
        """
        messages: list[Message] = []

        # The context is a list of {role, content} message dicts
        if isinstance(ctx_data, list):
            raw_messages = ctx_data
        elif isinstance(ctx_data, dict):
            raw_messages = (
                ctx_data.get("messages")
                or ctx_data.get("conversations")
                or ctx_data.get("context")
                or ctx_data.get("dialogue")
                or []
            )
            if isinstance(raw_messages, str):
                text = raw_messages[start_idx:end_idx] if end_idx > 0 else raw_messages
                return [Message(role="user", content=text)]
        else:
            return messages

        # Slice message list by range
        if end_idx > 0:
            raw_messages = raw_messages[start_idx:end_idx]
        elif start_idx > 0:
            raw_messages = raw_messages[start_idx:]

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content", msg.get("text", ""))

            # Preserve original roles — system messages contain persona info
            # that is important for the model to understand the user.
            # The official code keeps system messages as-is (and for o-series
            # models, converts them to user messages via convert_role_system_to_user).
            if role.lower() == "system":
                messages.append(Message(role="system", content=content))
            elif role.lower() in ("user", "human"):
                messages.append(Message(role="user", content=content))
            else:
                messages.append(Message(role="assistant", content=content))

        return messages

    @staticmethod
    def _format_options(options: list) -> str:
        """Format MCQ options as labeled list.

        If options already have labels like '(a) ...', use them as-is.
        Otherwise, add labels.

        Note: This is only used by legacy format. The official protocol
        passes all_options as a raw string directly.
        """
        import re
        # Check if first option already has a label prefix like "(a) " or "a) "
        if options and re.match(r"^\(?[a-d]\)?[\.\.\)\s]", str(options[0]).strip()):
            return "\n".join(f"  {opt}" for opt in options)

        labels = "abcdefgh"
        parts = []
        for i, opt in enumerate(options):
            label = labels[i] if i < len(labels) else str(i)
            parts.append(f"  ({label}) {opt}")
        return "\n".join(parts)
    def to_trajectories(self, conversations: list[Conversation]) -> list[UserTrajectory]:
        """PersonaMem 特殊处理：每个 (persona_id, context_id) 保持独立 trajectory。

        PersonaMem 中同一 persona 的不同 context 是「分叉的平行世界」，
        它们共享前缀但在某个点分叉，因此记忆不能互相污染。
        每个 conversation（= persona × context）直接映射为一个独立的 UserTrajectory。
        """
        from collections import OrderedDict

        trajectories: list[UserTrajectory] = []
        for conv in conversations:
            persona_id = conv.metadata.get("persona_id", "")
            context_id = conv.metadata.get("context_id", "default")
            # 每个 (persona, context) 对应一个独立的 trajectory
            traj = UserTrajectory(
                user_id=conv.metadata.get("user_id", conv.conv_id),
                sessions=list(conv.sessions),
                questions=list(conv.questions),
                metadata={
                    "benchmark": self.name,
                    "persona_id": persona_id,
                    "context_id": context_id,
                    "shared_context_id": conv.metadata.get("shared_context_id", ""),
                    "num_conversations": 1,
                    "conv_ids": [conv.conv_id],
                },
                conversations=[conv],
            )
            trajectories.append(traj)

        return trajectories

    def get_questions(self, conversation: Conversation) -> list[QAPair]:
        return conversation.questions

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        qa_pairs: list[QAPair] | None = None,
    ) -> dict[str, float]:
        """Compute PersonaMem metrics — aligned with official MCQ accuracy.

        Reports overall + per question_type accuracy.
        """
        if not predictions:
            return {"mcq_accuracy": 0.0}

        correct = 0
        total = len(predictions)
        type_correct: dict[str, int] = {}
        type_total: dict[str, int] = {}

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_answer = self._extract_mcq_answer(pred)
            ref_answer = self._extract_mcq_answer(ref)
            is_correct = pred_answer == ref_answer

            if is_correct:
                correct += 1

            if qa_pairs and i < len(qa_pairs):
                qtype = qa_pairs[i].question_type
                type_total[qtype] = type_total.get(qtype, 0) + 1
                if is_correct:
                    type_correct[qtype] = type_correct.get(qtype, 0) + 1

        metrics: dict[str, float] = {
            "mcq_accuracy": correct / total if total > 0 else 0.0,
        }

        for qtype in sorted(type_total.keys()):
            n = type_total[qtype]
            c = type_correct.get(qtype, 0)
            metrics[f"acc_{qtype}"] = c / n if n > 0 else 0.0
            metrics[f"n_{qtype}"] = float(n)

        return metrics

    def check_correctness(
        self,
        predictions: list[str],
        references: list[str],
        qa_pairs: list[QAPair] | None = None,
    ) -> list[bool]:
        """判断每个 QA 是否回答正确（MCQ 答案字母匹配）。

        Args:
            predictions: 模型生成的答案列表。
            references: 参考答案列表。
            qa_pairs: 可选的 QAPair 对象列表（此处未使用）。

        Returns:
            与 predictions 等长的布尔列表，True 表示该 QA 回答正确。
        """
        results: list[bool] = []
        for pred, ref in zip(predictions, references):
            pred_answer = self._extract_mcq_answer(pred)
            ref_answer = self._extract_mcq_answer(ref)
            results.append(pred_answer == ref_answer)
        return results

    @staticmethod
    def _extract_mcq_answer(text: str) -> str:
        """Extract MCQ answer letter from model output.

        Aligned with the official PersonaMem extract_answer logic:
        1. Look for <final_answer> tag and extract from there
        2. Extract all (a)-(d) options from the answer portion
        3. If exactly one option found and it matches correct → True
        4. Fallback to full response text

        For our evaluation, we just need to extract the predicted letter.
        Priority order:
        1. <final_answer> tag content
        2. **Correct Answer: (X)** format (for non-full_context baselines)
        3. "The answer is X" / "Answer: X" pattern
        4. Last (X) occurrence in text
        5. Single letter fallback
        """
        import re

        raw_text = text.strip()
        text_lower = raw_text.lower()

        # Direct single letter
        if len(text_lower) == 1 and text_lower in "abcd":
            return text_lower

        # Priority 1: <final_answer> tag — official PersonaMem format
        if "<final_answer>" in text_lower:
            after_tag = text_lower.split("<final_answer>")[-1].strip()
            if after_tag.endswith("</final_answer>"):
                after_tag = after_tag[:-len("</final_answer>")].strip()
            # Extract options from the tag content
            in_parens = re.findall(r'\(([a-d])\)', after_tag)
            if in_parens:
                return in_parens[-1]
            bare = re.findall(r'\b([a-d])\b', after_tag)
            if bare:
                return bare[-1]

        # Priority 2: **Correct Answer: (X)** — structured format for memory baselines
        m = re.search(r"\*{0,2}correct answer:?\s*\*{0,2}\s*\(?([a-d])\)?", text_lower)
        if m:
            return m.group(1)

        # Priority 3: "The answer is X" / "Answer: X" pattern
        m = re.search(r"(?:the answer is|answer:?)\s*\(?([a-d])\)?", text_lower)
        if m:
            return m.group(1)

        # Priority 4: LAST occurrence of standalone (X) — more reliable for long answers
        matches = list(re.finditer(r"\(([a-d])\)", text_lower))
        if matches:
            return matches[-1].group(1)

        # Priority 5: Starts with (X) or X) pattern
        m = re.match(r"\s*\(?([a-d])\)?[\.\.\)\s:,]", text_lower)
        if m:
            return m.group(1)

        # Priority 6: Last bare a-d letter (fallback)
        bare_matches = list(re.finditer(r'\b([a-d])\b', text_lower))
        if bare_matches:
            return bare_matches[-1].group(1)

        return text_lower[:1] if text_lower else ""
    # ------------------------------------------------------------------
    # Legacy format support (for backward compatibility)
    # ------------------------------------------------------------------

    def _parse_legacy_combined(self, raw: Any) -> list[Conversation]:
        """Parse legacy combined PersonaMem format (personamem.json)."""
        conversations = []
        items = raw if isinstance(raw, list) else raw.get("data", [])

        for item in items:
            persona_id = item.get("persona_id", item.get("id", ""))
            conv = Conversation(
                conv_id=str(persona_id),
                metadata={"user_id": str(persona_id)},
            )

            for j, sess in enumerate(item.get("sessions", [])):
                session = Session(session_id=f"{persona_id}_s{j}")
                for msg in sess.get("messages", []):
                    session.messages.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    ))
                if session.messages:
                    conv.sessions.append(session)

            for qa in item.get("questions", item.get("qa", [])):
                qtype = "mcq" if qa.get("options") else "open"
                conv.questions.append(QAPair(
                    question=qa.get("question", ""),
                    reference_answer=qa.get("answer", ""),
                    question_type=qtype,
                    metadata=qa,
                ))

            if conv.sessions:
                conversations.append(conv)

        return conversations
