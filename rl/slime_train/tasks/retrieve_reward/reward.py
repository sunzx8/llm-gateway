"""Canonical Retrieve RLVR custom reward entrypoint.

Use ``llm_gateway.rl.slime_train.tasks.retrieve_reward.reward.reward_func`` as
the Slime ``--custom-rm-path``. The implementation is split across focused
modules in this package; this file re-exports the public and legacy debug
helpers.
"""

from __future__ import annotations

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

try:  # pragma: no cover - fallback keeps direct script-style imports working
    from .answer_scorer import build_qa_prompt as _build_qa_prompt
    from .answer_scorer import extract_answer_tag as _extract_answer_tag
    from .answer_scorer import extract_content_words as _extract_content_words
    from .answer_scorer import score_fill as _score_fill
    from .answer_scorer import score_mcq as _score_mcq
    from .entry import compute_r_context_payload
    from .entry import compute_single_reward_async as _compute_single_reward_async
    from .entry import reward_func
    from .format_reward import compute_r_format
    from .frozen_qa_client import call_frozen_model as _call_frozen_model
    from .frozen_qa_client import call_frozen_model_async as _call_frozen_model_async
    from .frozen_qa_client import get_aiohttp_session as _get_aiohttp_session
    from .metrics import log_debug_response as _log_debug_response
    from .metrics import log_rollout_record as _log_rollout_record
    from .metrics import log_single_metric as _log_single_metric
    from .parser import strip_think_wrapper as _strip_think_wrapper
    from .parser import try_parse_json as _try_parse_json
    from .query_quality import compute_r_query_quality
    from .retrieval_hit import compute_r_retrieval_hit
    from .retrieval_hit import execute_real_retrieval as _execute_real_retrieval
    from .retrieval_hit import run_retrieval_with_queries as _run_retrieval_with_queries
    from .retrieval_hit import score_context_against_gold_async as _score_context_against_gold_async
    from .snapshot_cache import get_loaded_env as _get_loaded_env
    from .snapshot_cache import get_snapshot_session as _get_snapshot_session
except ImportError:  # pragma: no cover
    from answer_scorer import build_qa_prompt as _build_qa_prompt
    from answer_scorer import extract_answer_tag as _extract_answer_tag
    from answer_scorer import extract_content_words as _extract_content_words
    from answer_scorer import score_fill as _score_fill
    from answer_scorer import score_mcq as _score_mcq
    from entry import compute_r_context_payload
    from entry import compute_single_reward_async as _compute_single_reward_async
    from entry import reward_func
    from format_reward import compute_r_format
    from frozen_qa_client import call_frozen_model as _call_frozen_model
    from frozen_qa_client import call_frozen_model_async as _call_frozen_model_async
    from frozen_qa_client import get_aiohttp_session as _get_aiohttp_session
    from metrics import log_debug_response as _log_debug_response
    from metrics import log_rollout_record as _log_rollout_record
    from metrics import log_single_metric as _log_single_metric
    from parser import strip_think_wrapper as _strip_think_wrapper
    from parser import try_parse_json as _try_parse_json
    from query_quality import compute_r_query_quality
    from retrieval_hit import compute_r_retrieval_hit
    from retrieval_hit import execute_real_retrieval as _execute_real_retrieval
    from retrieval_hit import run_retrieval_with_queries as _run_retrieval_with_queries
    from retrieval_hit import score_context_against_gold_async as _score_context_against_gold_async
    from snapshot_cache import get_loaded_env as _get_loaded_env
    from snapshot_cache import get_snapshot_session as _get_snapshot_session

__all__ = [
    "reward_func",
    "compute_r_context_payload",
    "compute_r_format",
    "compute_r_query_quality",
    "compute_r_retrieval_hit",
    "_build_qa_prompt",
    "_call_frozen_model",
    "_call_frozen_model_async",
    "_compute_single_reward_async",
    "_execute_real_retrieval",
    "_extract_answer_tag",
    "_extract_content_words",
    "_get_aiohttp_session",
    "_get_loaded_env",
    "_get_snapshot_session",
    "_log_debug_response",
    "_log_rollout_record",
    "_log_single_metric",
    "_run_retrieval_with_queries",
    "_score_context_against_gold_async",
    "_score_fill",
    "_score_mcq",
    "_strip_think_wrapper",
    "_try_parse_json",
]
