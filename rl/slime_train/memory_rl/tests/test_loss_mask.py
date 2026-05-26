"""End-to-end tests for custom_generate.py loss-mask correctness.

What we verify (each as an independent test):

  L1  token alignment        : prompt_token_ids + response_token_ids equals
                               tokenize(final_concatenated_string), i.e. no
                               BPE drift across segment boundaries.

  L2  mask values             : assistant-generated tokens get loss_mask=1,
                               template/observation tokens get loss_mask=0,
                               with NO off-by-one against the chat_template.

  L3  chat_template equivalence:
                               the final tokens reconstructed by custom_generate
                               equal what tokenizer.apply_chat_template would
                               produce for the FULL multi-turn conversation
                               (with assistant<|im_end|> on every turn).

  L4  mock training step      : we build a tiny LM head on top of one-hot token
                               embeddings, compute the slime-style shift-by-one
                               cross-entropy loss masked by sample.loss_mask, and
                               check that:
                                 (a) gradients on assistant tokens are nonzero,
                                 (b) gradients on observation tokens are zero,
                                 (c) deliberately mis-shifting the mask by 1
                                     produces a *different* loss
                                     (proves mask actually moves where we think).

The test deliberately mocks SGLang. We *replace* `_post_generate` with a fake
that returns a deterministic assistant_text whose finish_reason is "stop".
This isolates the question we care about: given correct generations, does
custom_generate build a (tokens, response_length, loss_mask) triple that is
self-consistent and semantically right?

Run with:
    cd /data/cloud_disk_1/erenpeng/llm-gateway
    pytest -xvs rl/slime_train/memory_rl/tests/test_loss_mask.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]                                       # /data/cloud_disk_1/erenpeng/llm-gateway
_PKG  = _REPO / "llm_gateway"                                  # the inner llm_gateway/ package root
for _p in (str(_REPO), str(_PKG.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOKENIZER_PATH = os.environ.get(
    "MEMORY_RL_TEST_TOKENIZER",
    "/data/cloud_disk_1/erenpeng/models/Qwen/Qwen3.6-27B",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    if not Path(TOKENIZER_PATH).is_dir():
        pytest.skip(f"tokenizer dir not found: {TOKENIZER_PATH}")
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


@pytest.fixture
def cg_module(tokenizer, monkeypatch):
    """Import custom_generate AFTER monkey-patching its TOKENIZER + network."""
    # Make sure response_parser.set_tool_schemas / get_tool_schemas don't blow up
    # when tools is None.
    from llm_gateway.rl.slime_train.memory_rl import custom_generate as cg
    # Inject our cached tokenizer.
    monkeypatch.setattr(cg, "TOKENIZER", tokenizer, raising=False)
    return cg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSample:
    """Stand-in for slime.utils.types.Sample (only fields used by custom_generate)."""

    class Status:
        PENDING   = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED   = "aborted"
        FAILED    = "failed"

    def __init__(self, prompt, metadata=None):
        self.prompt = prompt
        self.metadata = metadata or {}
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.loss_mask = None
        self.status = self.Status.PENDING


class FakeArgs:
    sglang_router_url = "http://127.0.0.1:9999"
    hf_checkpoint = TOKENIZER_PATH


def _install_fake_post_generate(cg, scripted_outputs):
    """Replace cg._post_generate with a deterministic queue-driven fake.

    `scripted_outputs` is a list of (text, finish_reason) tuples; one entry
    is consumed per LLM call.
    """
    queue = list(scripted_outputs)

    async def fake_post_generate(args, text, sampling_params):
        if not queue:
            raise RuntimeError("scripted SGLang queue exhausted")
        out_text, finish = queue.pop(0)
        return {
            "text": out_text,
            "meta_info": {"finish_reason": {"type": finish}},
        }

    cg._post_generate = fake_post_generate
    return queue  # caller can inspect


def _install_fake_tool_executor(cg, observations_per_call):
    """Replace _build_tool_executor with one that yields scripted observations."""
    queue = list(observations_per_call)

    @asynccontextmanager
    async def fake_builder(task, metadata, args):
        async def executor(calls):
            if not queue:
                return [{"id": c.get("id", ""), "tool": c.get("tool", ""), "result": ""} for c in calls]
            return queue.pop(0)
        yield executor

    cg._build_tool_executor = fake_builder
    return queue


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Scripted multi-turn rollout that we'll reuse across L1/L2/L3
# ---------------------------------------------------------------------------


@pytest.fixture
def scripted_rollout(tokenizer):
    """A 2-turn rollout: turn0 calls a tool, turn1 calls finish."""
    initial_messages = [
        {"role": "system", "content": "You are a memory agent."},
        {"role": "user",   "content": "Please ingest the latest session and finish."},
    ]

    # Turn 0: model emits a tool_call. NOTE: Qwen3.6 chat_template injects
    # `<think>\n` as part of the generation prompt, so SGLang's returned
    # `text` should *not* start with `<think>`. The model's own output
    # begins with the *thinking content*, then `</think>`, then the answer.
    turn0_text = (
        "I need to ingest first.\n</think>\n\n"
        "<tool_call>\n<function=add_memory>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>"
    )
    # Tool returns a result
    turn0_obs = [{"id": "slime_call_0", "tool": "add_memory", "result": "ok: stored mem#1"}]

    # Turn 1: model finishes (same convention - no leading <think>)
    turn1_text = (
        "Done.\n</think>\n\n"
        "<tool_call>\n<function=finish>\n<parameter=summary>\ningested 1 memory\n</parameter>\n</function>\n</tool_call>"
    )

    return {
        "messages":  initial_messages,
        "scripted":  [(turn0_text, "stop"), (turn1_text, "stop")],
        "obs":       [turn0_obs],
        "turn_texts": [turn0_text, turn1_text],
    }


def _run_custom_generate(cg, scripted, tokenizer):
    """Drive cg.custom_generate with the scripted rollout."""
    _install_fake_post_generate(cg, scripted["scripted"])
    _install_fake_tool_executor(cg, scripted["obs"])

    sample = FakeSample(
        prompt=scripted["messages"],
        metadata={
            "task": "ingest",
            "snapshot_id": "snap-test",        # forces multi-turn branch
            "session_id": "sess-1",
            "tools": [],                       # avoid hitting tool_schemas registry
        },
    )

    # Make sure tool_schemas / response_parser don't reach the network or
    # blow up; provide a minimal stub that returns no schemas so set_tool_schemas
    # is a no-op.
    import llm_gateway.rl.slime_train.memory_rl.tool_schemas as ts
    if not hasattr(ts, "_orig_get_tool_schemas"):
        ts._orig_get_tool_schemas = ts.get_tool_schemas
    ts.get_tool_schemas = lambda task: None  # type: ignore

    # Disable native task_loop branch
    os.environ.pop("MEMORY_RL_APPLY_MODE", None)

    result = asyncio.run(cg.custom_generate(FakeArgs(), sample, sampling_params={"temperature": 0.0}))
    return result


# ---------------------------------------------------------------------------
# L1: token alignment - prompt_ids + response_ids must equal tokenize(full_string)
# ---------------------------------------------------------------------------


def test_L1_token_alignment_no_bpe_drift(cg_module, tokenizer, scripted_rollout):
    """The hard slime contract: sample.tokens == tokenizer(final_string).

    custom_generate independently encodes prompt, each assistant chunk, and each
    observation chunk. If BPE merges across segment boundaries, the concatenated
    ids won't equal a single tokenization of the joined string. That would make
    sample.tokens silently mis-aligned with what slime actually trains on.
    """
    cg = cg_module
    sample = _run_custom_generate(cg, scripted_rollout, tokenizer)

    # Reconstruct what custom_generate THINKS the model saw end-to-end.
    # We mirror cg.custom_generate's own concatenation rules.
    prompt_text = cg._render_prompt(scripted_rollout["messages"], tokenizer, tools=None)
    expected_text = prompt_text
    expected_text += scripted_rollout["turn_texts"][0]
    obs_block = "\n".join(
        f"<tool_response>\n{o['result']}\n</tool_response>" for o in scripted_rollout["obs"][0]
    )
    expected_text += (
        f"<|im_end|>\n<|im_start|>user\n{obs_block}<|im_end|>"
        f"\n<|im_start|>assistant\n<think>\n"
    )
    expected_text += scripted_rollout["turn_texts"][1]

    expected_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    assert sample.tokens == expected_ids, (
        "sample.tokens drifts vs single-shot tokenization.\n"
        f"  len(sample.tokens) = {len(sample.tokens)}\n"
        f"  len(expected_ids) = {len(expected_ids)}\n"
        f"  first diff at = {next((i for i,(a,b) in enumerate(zip(sample.tokens, expected_ids)) if a!=b), 'tail')}\n"
        f"  decoded(sample.tokens)[-200:] = {repr(_decode(tokenizer, sample.tokens)[-200:])}\n"
        f"  expected_text[-200:]          = {repr(expected_text[-200:])}"
    )


# ---------------------------------------------------------------------------
# L2: mask values - 1 on assistant content, 0 on template / observation
# ---------------------------------------------------------------------------


def test_L2_mask_invariants(cg_module, tokenizer, scripted_rollout):
    """Slime's hard contract: len(loss_mask) == response_length.
    Plus: mask must be 1 exactly on assistant-generated tokens, 0 elsewhere
    inside the response segment.
    """
    cg = cg_module
    sample = _run_custom_generate(cg, scripted_rollout, tokenizer)

    # ---- Hard contract from ray/rollout.py:713-715 -----------------------
    assert isinstance(sample.loss_mask, list)
    assert len(sample.loss_mask) == sample.response_length, (
        f"slime asserts len(loss_mask)==response_length, "
        f"got {len(sample.loss_mask)} vs {sample.response_length}"
    )
    assert sample.response_length + (len(sample.tokens) - sample.response_length) == len(sample.tokens)

    # ---- Compute the *expected* mask from segment plan --------------------
    response_ids = sample.tokens[-sample.response_length:]

    a0 = tokenizer(scripted_rollout["turn_texts"][0], add_special_tokens=False)["input_ids"]
    obs_block = "\n".join(
        f"<tool_response>\n{o['result']}\n</tool_response>" for o in scripted_rollout["obs"][0]
    )
    obs_seg_text = (
        f"<|im_end|>\n<|im_start|>user\n{obs_block}<|im_end|>"
        f"\n<|im_start|>assistant\n<think>\n"
    )
    obs_ids = tokenizer(obs_seg_text, add_special_tokens=False)["input_ids"]
    a1 = tokenizer(scripted_rollout["turn_texts"][1], add_special_tokens=False)["input_ids"]

    # custom_generate's per-segment encoding => expected mask.
    expected_mask = [1]*len(a0) + [0]*len(obs_ids) + [1]*len(a1)
    expected_response_ids = a0 + obs_ids + a1

    assert response_ids == expected_response_ids, "response_ids segmentation mismatch"
    assert sample.loss_mask == expected_mask, (
        "loss_mask values disagree with the documented segment plan.\n"
        f"  sum(mask)={sum(sample.loss_mask)} expected_sum={sum(expected_mask)}\n"
        f"  first_diff={next((i for i,(a,b) in enumerate(zip(sample.loss_mask, expected_mask)) if a!=b), None)}"
    )


# ---------------------------------------------------------------------------
# L3: chat_template equivalence - we are training on a *legal* multi-turn chat
# ---------------------------------------------------------------------------


def test_L3_matches_chat_template(cg_module, tokenizer, scripted_rollout):
    """The token stream we feed to slime must equal what apply_chat_template
    would produce for the same conversation. If it doesn't, the model sees a
    different multi-turn format at train vs serve time.

    NOTE on Qwen3.6 chat_template specifics:
      - assistant turns end with <|im_end|>\\n which the model is expected
        to emit. SGLang strips that token in the returned `text`.
      - tool observations are wrapped as a "user" message containing
        <tool_response>...</tool_response>.
      - generation prompt ends in <|im_start|>assistant\\n<think>\\n.

    We rebuild the conversation as messages list and apply_chat_template it,
    then compare to sample.tokens.
    """
    cg = cg_module
    sample = _run_custom_generate(cg, scripted_rollout, tokenizer)

    full_messages = list(scripted_rollout["messages"])
    full_messages.append({"role": "assistant", "content": scripted_rollout["turn_texts"][0]})
    full_messages.append({"role": "user", "content":
        "\n".join(f"<tool_response>\n{o['result']}\n</tool_response>" for o in scripted_rollout["obs"][0])
    })
    full_messages.append({"role": "assistant", "content": scripted_rollout["turn_texts"][1]})

    # Render WITHOUT add_generation_prompt (this is the final closed conversation).
    gold = tokenizer.apply_chat_template(
        full_messages, tools=None, tokenize=False, add_generation_prompt=False
    )
    gold_ids = tokenizer(gold, add_special_tokens=False)["input_ids"]
    sample_decoded = tokenizer.decode(sample.tokens, skip_special_tokens=False)

    # We don't expect *equality* of decoded strings here -- custom_generate inserts
    # a `<think>\n` lead-in before turn1's assistant content, while
    # apply_chat_template may not (depends on the template). Instead we report
    # the diff so the human reviewer can decide.
    if sample.tokens != gold_ids:
        # Find the first divergence and report context around it
        first = next((i for i,(a,b) in enumerate(zip(sample.tokens, gold_ids)) if a != b), min(len(sample.tokens), len(gold_ids)))
        ctx_lo = max(0, first - 8)
        sample_ctx = tokenizer.decode(sample.tokens[ctx_lo:first+8], skip_special_tokens=False)
        gold_ctx   = tokenizer.decode(gold_ids[ctx_lo:first+8],     skip_special_tokens=False)
        pytest.fail(
            "custom_generate's tokens != apply_chat_template(full conv).\n"
            f"  sample_len={len(sample.tokens)}, gold_len={len(gold_ids)}, first_diff_at={first}\n"
            f"  sample_ctx[{ctx_lo}:{first+8}] = {sample_ctx!r}\n"
            f"  gold_ctx  [{ctx_lo}:{first+8}] = {gold_ctx!r}\n"
            f"  -- This means train-time format != serve-time format."
        )


# ---------------------------------------------------------------------------
# L4: mock training step - prove the mask actually selects the right tokens
# ---------------------------------------------------------------------------


def _slime_style_loss(tokens, response_length, loss_mask, model, vocab_size, hidden=64):
    """Replicate the slime-side loss computation, simplified to one sample.

    Per backends/megatron_utils/loss.py:88-95 :
        end   = total_length
        start = end - response_length
        logits_chunk  = logits[start-1 : end-1]   # (response_length, V)
        tokens_chunk  = tokens[-response_length:]  # (response_length,)
        per_token_ce  = CE(logits_chunk, tokens_chunk)
        masked_loss   = (per_token_ce * loss_mask).sum() / max(loss_mask.sum(), 1)

    `model` is a (embed, lm_head) tuple producing logits of shape (T, V).
    Using a real (low-dim) embedding makes per-position logits genuinely
    distinct, so a 1-step mask shift produces a different loss.
    """
    embed, lm_head = model
    tokens_t = torch.tensor(tokens, dtype=torch.long)            # (T,)
    mask_t   = torch.tensor(loss_mask, dtype=torch.float32)      # (R,)
    T = tokens_t.shape[0]
    h = embed(tokens_t)                                          # (T, H)
    logits = lm_head(h)                                          # (T, V)

    end   = T
    start = end - response_length
    logits_chunk  = logits[start - 1 : end - 1]                   # (R, V)
    targets_chunk = tokens_t[-response_length:]                   # (R,)

    per_tok = F.cross_entropy(logits_chunk, targets_chunk, reduction="none")  # (R,)
    masked_loss = (per_tok * mask_t).sum() / mask_t.sum().clamp_min(1.0)
    return masked_loss, per_tok.detach(), logits


def test_L4_mock_train_step_grad_flows_only_through_mask(cg_module, tokenizer, scripted_rollout):
    """End-to-end smoke: construct loss exactly like slime does, and check
    that gradients are nonzero ONLY on response positions where loss_mask=1.
    Also assert that off-by-one shifting the mask changes the loss value
    (proving the mask is wired to the right positions, not no-ops)."""
    cg = cg_module
    sample = _run_custom_generate(cg, scripted_rollout, tokenizer)

    V = max(tokenizer.vocab_size + 256, max(sample.tokens) + 1)
    H = 64

    torch.manual_seed(0)
    embed   = torch.nn.Embedding(V, H)
    lm_head = torch.nn.Linear(H, V, bias=False)
    model   = (embed, lm_head)

    # 1) Real masked loss
    loss, _, _ = _slime_style_loss(
        sample.tokens, sample.response_length, sample.loss_mask, model, V, hidden=H
    )

    # 2) Re-compute with grad-on-embedding-rows to inspect WHICH input positions
    #    contributed to the loss. We do this by tracking gradient on the embed
    #    output, not on the input ids.
    embed.zero_grad(set_to_none=True); lm_head.zero_grad(set_to_none=True)
    tokens_t = torch.tensor(sample.tokens, dtype=torch.long)
    h = embed(tokens_t)
    h.retain_grad()
    logits = lm_head(h)
    R = sample.response_length
    end = len(sample.tokens); start = end - R
    logits_chunk = logits[start-1:end-1]
    targets = tokens_t[-R:]
    per_tok = F.cross_entropy(logits_chunk, targets, reduction="none")
    mask_t = torch.tensor(sample.loss_mask, dtype=torch.float32)
    ((per_tok * mask_t).sum() / mask_t.sum().clamp_min(1.0)).backward()

    grad_norm_per_pos = h.grad.abs().sum(dim=-1)                  # (T,)
    contributes = (grad_norm_per_pos > 1e-9).tolist()

    # CE on response position r reads logits[start-1+r]; that consumes h[start-1+r].
    # So input position p contributes iff p in [start-1, end-2] AND
    # loss_mask[p - (start-1)] == 1.
    expected_contrib = [False] * len(sample.tokens)
    for r in range(R):
        if sample.loss_mask[r] == 1:
            expected_contrib[start - 1 + r] = True

    for i, (got, want) in enumerate(zip(contributes, expected_contrib)):
        if i < start - 1 or i > end - 2:
            assert not got, (
                f"non-response position {i} should have zero grad but got "
                f"{grad_norm_per_pos[i].item()}"
            )
        else:
            r = i - (start - 1)
            mask_here = sample.loss_mask[r]
            assert got == want, (
                f"mismatch at input_pos={i} (response_pos={r}): "
                f"contributes={got}, expected={want}, mask_here={mask_here}"
            )

    # ---- Sanity: shifting the mask by 1 must change the loss (not no-op) ----
    shifted_mask = [0] + sample.loss_mask[:-1]
    if any(a != b for a, b in zip(shifted_mask, sample.loss_mask)):
        loss_shift, _, _ = _slime_style_loss(
            sample.tokens, sample.response_length, shifted_mask, model, V, hidden=H
        )
        assert not torch.isclose(loss, loss_shift, atol=1e-4), (
            f"loss invariant to mask shift: orig={loss.item():.6f} shifted={loss_shift.item():.6f}"
        )


# ---------------------------------------------------------------------------
# L5: extra paranoia -- check that <|im_end|> learning signal is present.
# This is the regression I flagged earlier.
# ---------------------------------------------------------------------------


def test_L5_im_end_loss_signal(cg_module, tokenizer, scripted_rollout):
    """Each assistant turn should END with <|im_end|> AND that <|im_end|>
    must have loss_mask=1 -- otherwise the model never learns to stop.

    Current implementation: SGLang strips <|im_end|> from `text`, and
    custom_generate prepends it as part of the *observation* segment with
    mask=0. Therefore <|im_end|> for turn 0 should have mask=0 (BUG).
    This test documents that bug and will start passing once it's fixed.
    """
    cg = cg_module
    sample = _run_custom_generate(cg, scripted_rollout, tokenizer)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    response_ids = sample.tokens[-sample.response_length:]
    response_mask = sample.loss_mask

    # Find every <|im_end|> position inside the response segment.
    im_end_positions = [i for i, t in enumerate(response_ids) if t == im_end_id]
    assert im_end_positions, "no <|im_end|> found in response_ids -- assistant never stops?"

    masks_at_im_end = [response_mask[i] for i in im_end_positions]
    # We *want* every assistant-emitted <|im_end|> to be mask=1.
    # If even one is mask=0, EOS learning is degraded.
    if any(m == 0 for m in masks_at_im_end):
        pytest.xfail(
            f"BUG: at least one <|im_end|> has loss_mask=0 "
            f"(positions={im_end_positions}, masks={masks_at_im_end}). "
            f"Model won't learn when to stop on those turns."
        )
