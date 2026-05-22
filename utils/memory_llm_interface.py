"""Unified LLM interface supporting OpenAI, Anthropic, and OpenAI-compatible APIs.

Supports:
- OpenAI (GPT-4o, GPT-5.4, etc.)
- Anthropic (Claude)
- OpenAI-compatible APIs (self-deployed GLM, Qwen, DeepSeek, LLaMA, etc.)

Uses model-native tool calling (function calling) instead of custom XML tags.
All providers are normalized to the OpenAI tool calling format.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
import logger.logger as logger


@dataclass
class ToolCall:
    """A single tool call from the model (OpenAI function calling format)."""

    id: str
    name: str  # "execute_bash" | "finish"
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # original API object for provider-specific use

    @classmethod
    def from_openai(cls, tc) -> ToolCall:
        """Parse from OpenAI tool_call object.

        Handles two formats:
        - Standard OpenAI: arguments is a JSON string (str)
        - Qwen3.5/vLLM native tool call: arguments is already a dict
        """
        args = {}
        if tc.function.arguments:
            raw_args = tc.function.arguments
            if isinstance(raw_args, dict):
                # Qwen3.5 / vLLM returns arguments as a dict directly
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {"command": raw_args}
        return cls(id=tc.id, name=tc.function.name, arguments=args, raw=tc)


@dataclass
class LLMResponse:
    """Response from an LLM API call, with native tool calling support."""

    content: str = ""  # text content (reasoning / final answer)
    reasoning_content: str = ""  # 推理模型的思维链内容（DeepSeek-R1, GLM-5.1 等）
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    raw_response: Any = None

    # --- Helpers for building multi-turn tool use conversations ---

    def to_message(self) -> dict:
        """Convert to an OpenAI-style assistant message (for appending to conversation)."""
        msg: dict[str, Any] = {"role": "assistant"}
        if self.content:
            msg["content"] = self.content
        else:
            msg["content"] = None  # OpenAI requires content field even if null
        # 保留推理模型的思维链内容（或客户端提取的推理链）
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# ---------------------------------------------------------------------------
# Embedding interface
# ---------------------------------------------------------------------------


class EmbeddingInterface:
    """统一的 Embedding 接口，支持 OpenAI 和 OpenAI 兼容 API。

    使用 OpenAI SDK 的 embeddings.create 接口，兼容所有 OpenAI-compatible
    embedding 服务（如 vLLM、TEI、Infinity、Ollama 等）。
    """

    def __init__(self, config: dict[str, Any]):
        """初始化 Embedding 接口。

        Args:
            config: Embedding 配置字典，包含 provider/model/api_key/base_url 等。
        """
        # 从配置字典读取，回退到环境变量，最终使用默认值
        self.provider = config.get("provider") or os.environ.get("EMBEDDING_PROVIDER", "openai")
        self.model = config.get("model") or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = config.get("api_key") or os.environ.get(
            "EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = config.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")
        self.timeout = float(config.get("timeout", 120.0))
        self.max_retries = int(config.get("max_retries", 3))

        _batch_size = config.get("batch_size")
        if _batch_size is not None:
            self.batch_size = int(_batch_size)
        else:
            self.batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))

        _dimensions = config.get("dimensions")
        if _dimensions is not None:
            self.dimensions = int(_dimensions)
        else:
            dim_str = os.environ.get("EMBEDDING_DIMENSIONS")
            self.dimensions = int(dim_str) if dim_str else None

        # 自动检测 provider：如果设置了非 OpenAI 的 base_url，切换到 openai_compat
        if (
            self.base_url
            and self.provider == "openai"
            and "api.openai.com" not in self.base_url
        ):
            logger.info(
                "EmbeddingInterface: auto-detected non-OpenAI base_url (%s), "
                "switching provider to openai_compat",
                self.base_url,
            )
            self.provider = "openai_compat"

        # openai_compat 模式下提供默认 api_key
        if self.provider == "openai_compat" and not self.api_key:
            self.api_key = "none"

        self._total_tokens = 0
        self._call_count = 0
        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._client = AsyncOpenAI(
                api_key=self.api_key or "none",
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def serialize(self) -> dict[str, Any]:
        """将 EmbeddingInterface 序列化为可 JSON 化的配置字典。

        序列化所有初始化参数，以便在子进程中重建实例。

        Returns:
            包含重建实例所需全部参数的字典。
        """
        config: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
        }
        if self.dimensions is not None:
            config["dimensions"] = self.dimensions
        return config

    @property
    def stats(self) -> dict[str, Any]:
        """返回累计使用统计。"""
        return {
            "total_tokens": self._total_tokens,
            "total_calls": self._call_count,
        }

    async def aclose(self) -> None:
        """Explicitly close the underlying AsyncOpenAI client.

        Tolerates the known openai/httpx version-mismatch bug where
        ``AsyncHttpxClientWrapper.aclose()`` raises ``AttributeError``
        because ``_transport`` is missing.
        """
        client = self._client
        if client is None:
            return
        try:
            await client.close()
        except AttributeError as e:
            logger.debug("Ignored AttributeError while closing embedding client: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.debug("Ignored error while closing embedding client: %s", e)
        self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。

        自动按 batch_size 分批请求，避免超出 API 限制。

        Args:
            texts: 待编码的文本列表。

        Returns:
            与 texts 等长的向量列表，每个向量是 float 列表。
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: list[list[float]] = []
        batch_size = self.batch_size or 64

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": batch,
            }
            # 部分模型支持 dimensions 参数（如 text-embedding-3-small/large）
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            try:
                response = await client.embeddings.create(**kwargs)

                # 按 index 排序确保顺序正确
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

                # 统计 token 用量
                if hasattr(response, "usage") and response.usage:
                    self._total_tokens += getattr(response.usage, "total_tokens", 0) or 0
                self._call_count += 1

                logger.debug(
                    "Embedding batch %d/%d: %d texts, model=%s",
                    i // batch_size + 1,
                    (len(texts) + batch_size - 1) // batch_size,
                    len(batch),
                    self.model,
                )

            except Exception as e:
                logger.error("Embedding API call failed: %s", e)
                raise

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """编码单个文本。"""
        results = await self.embed([text])
        return results[0]


# ---------------------------------------------------------------------------
# Tool definitions — the two tools our Memory Model can use
# ---------------------------------------------------------------------------

MEMORY_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": (
                "Execute bash commands in the memory repo sandbox. "
                "PREFERRED: Use mem_* helper commands for common operations:\n"
                "  - mem_write <file> <<'EOF'\\n...\\nEOF  (multi-line write, auto-mkdir)\n"
                "  - mem_write -a <file> <<'EOF'\\n...\\nEOF  (multi-line APPEND)\n"
                "  - mem_read <file> [start] [end] (read file with line numbers: mem_read f.md = full, mem_read f.md 1 50 = lines 1-50)\n"
                "  - mem_ls (list memory files), mem_cat_all (view all memory contents)\n"
"  - mem_search \"kw1\" \"kw2\" ... (hybrid batch search: BM25 + vector — input concise keywords only, not full sentences. Output grouped by query then by file: == file_path (score: X.XX) == + L<line>: text. Fallback: grep -rn \"keyword\" knowledge/ skills/)\n"
                "  - mem_search_lt \"q1\" \"q2\" ... (long-term only hybrid batch, same output format)\n"
                "  - echo \"content\" | mem_create <file> (create with stdin)\n"
                "  - echo \"content\" | mem_append <file> (append single line)\n"
                "  - mem_commit \"message\" (git add -A && git commit)\n"
                "  - mem_tags \"tag\", mem_type \"type\" (search by frontmatter)\n"
                "  - mem_log [N] (git history), mem_history <file> (file history)\n"
                "Also available: ls, cat, grep, find, sed, mkdir, rm, mv, cp, "
                "touch, head, tail, wc, sort, uniq, tee, awk, tr, cut, diff, git, echo. "
                "Environment variables (ALREADY SET — do NOT compute these yourself): "
                "$CURRENT_TIME (current timestamp string), $CURRENT_SESSION_ID (active session ID), "
                "$MEMORY_REPO_PATH (repo root). "
                "Working directory is the memory repo root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute. Use $CURRENT_TIME and $CURRENT_SESSION_ID variables directly — do NOT call date or datetime to get the current time.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in the memory repo sandbox. "
                "Use ONLY for complex file processing, YAML/frontmatter manipulation, "
                "batch operations, or data parsing that cannot be done easily in bash. "
                "DO NOT use this to get the current time — use $CURRENT_TIME env var in execute_bash instead. "
                "Environment variables: MEMORY_REPO_PATH, CURRENT_TIME, CURRENT_SESSION_ID. "
                "Working directory is the memory repo root. "
                "Available stdlib: os, pathlib, json, yaml, re, glob, datetime, subprocess, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Do NOT use this for getting timestamps — use execute_bash with $CURRENT_TIME instead.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal task completion. Optionally return a result "
                "(for retrieval tasks: return formatted memory context; "
                "for write tasks: return a brief summary of actions taken)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Task result content. Leave empty if no result needed.",
                    }
                },
            },
        },
    },
]

# Anthropic format (converted from OpenAI format)
MEMORY_TOOLS_ANTHROPIC = [
    {
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "input_schema": tool["function"]["parameters"],
    }
    for tool in MEMORY_TOOLS_OPENAI
]


class LLMInterface:
    """Unified interface for calling LLM APIs with native tool calling.

    Supports:
    - OpenAI (GPT-5.4, GPT-4o, etc.)
    - Anthropic (Claude)
    - OpenAI-compatible APIs (self-deployed GLM, Qwen, DeepSeek, LLaMA via vLLM/SGLang/etc.)

    All responses are normalized to the same ToolCall format.
    """

    # Endpoints that require 'max_completion_tokens' instead of 'max_tokens'.
    _MAX_COMPLETION_TOKENS_URLS = frozenset({
        "https://vdbteam.cognitiveservices.azure.com/openai/v1/",
    })

    def _needs_max_completion_tokens(self) -> bool:
        """Check if the configured base_url requires ``max_completion_tokens``."""
        return self.base_url in self._MAX_COMPLETION_TOKENS_URLS

    def __init__(self, config: dict[str, Any]):
        """初始化 LLM 接口。

        Args:
            config: LLM 配置字典，包含 provider/model/api_key/base_url 等。
        """
        # 从配置字典读取，使用默认值
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.temperature = float(config.get("temperature", 1.0))
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.timeout = float(config.get("timeout", 600.0))
        self.max_retries = int(config.get("max_retries", 3))

        # 自动检测：如果设置了非 OpenAI 的 base_url，切换到 openai_compat
        if (
            self.base_url
            and self.provider == "openai"
            and "api.openai.com" not in self.base_url
        ):
            logger.info(
                "Auto-detected non-OpenAI base_url (%s), switching provider to openai_compat",
                self.base_url,
            )
            self.provider = "openai_compat"

        # openai_compat 模式下提供默认 api_key
        if self.provider == "openai_compat" and not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "none")

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._total_tool_calls = 0       # 累计 tool_calls 次数（LLM 返回的 tool_call 个数）
        self._total_llm_latency_ms = 0.0  # 累计 LLM 调用耗时（毫秒）
        # Reuse a single AsyncOpenAI / AsyncAnthropic client across calls.
        # Creating a new client per request leaks httpx connections and
        # triggers 'AsyncHttpxClientWrapper has no _transport' warnings
        # during GC-triggered aclose().
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    def _get_openai_client(self) -> Any:
        """Lazily create and cache the AsyncOpenAI client (works for both
        'openai' and 'openai_compat' providers).
        """
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        if self.provider == "openai":
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        else:
            # openai_compat: allow placeholder key
            api_key = self.api_key or "none"

        self._openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        # Opik 追踪：wrap client 让所有 LLM 调用自动上报
        try:
            from opik.integrations.openai import track_openai
            self._openai_client = track_openai(self._openai_client)
            logger.info("Opik: track_openai 已启用")
        except Exception as e:
            logger.warning(f"Opik: track_openai 失败: {e}")

        return self._openai_client

    def _get_anthropic_client(self) -> Any:
        """Lazily create and cache the AsyncAnthropic client."""
        if self._anthropic_client is not None:
            return self._anthropic_client
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self._anthropic_client = AsyncAnthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._anthropic_client

    async def aclose(self) -> None:
        """Explicitly close underlying HTTP clients.

        Tolerates the known openai SDK issue where
        ``AsyncHttpxClientWrapper.aclose()`` raises ``AttributeError``
        because of a version mismatch with httpx.
        """
        for attr in ("_openai_client", "_anthropic_client"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            try:
                await client.close()
            except AttributeError as e:
                # Known SDK bug: httpx wrapper missing '_transport'
                logger.debug("Ignored AttributeError while closing %s: %s", attr, e)
            except Exception as e:  # noqa: BLE001
                logger.debug("Ignored error while closing %s: %s", attr, e)
            setattr(self, attr, None)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._call_count,
        }

    async def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Call the LLM and return the response with tool calls.

        Args:
            system: System prompt.
            messages: Conversation messages (OpenAI format).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            tools: Tool definitions (OpenAI format). None = no tool calling.

        Returns:
            LLMResponse with content, tool_calls, and usage stats.
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens or self.max_tokens

        logger.info(
            "LLM request: provider=%s, model=%s, messages=%d, tools=%s, max_tokens=%d",
            self.provider, self.model, len(messages),
            len(tools) if tools else "none", max_tok,
        )

        start = time.monotonic()

        if self.provider == "openai":
            resp = await self._call_openai(system, messages, temp, max_tok, tools)
        elif self.provider == "openai_compat":
            resp = await self._call_openai_compat(system, messages, temp, max_tok, tools)
        elif self.provider == "anthropic":
            resp = await self._call_anthropic(system, messages, temp, max_tok, tools)
        else:
            logger.error("Unsupported provider: %s", self.provider)
            raise ValueError(f"Unsupported provider: {self.provider}")

        resp.latency_ms = (time.monotonic() - start) * 1000
        self._total_input_tokens += resp.input_tokens
        self._total_output_tokens += resp.output_tokens
        self._call_count += 1
        self._total_tool_calls += len(resp.tool_calls)
        self._total_llm_latency_ms += resp.latency_ms

        logger.info(
            "LLM response #%d: model=%s, in=%d, out=%d, tool_calls=%d, latency=%.0fms, content_len=%d",
            self._call_count, resp.model, resp.input_tokens, resp.output_tokens,
            len(resp.tool_calls), resp.latency_ms, len(resp.content),
        )

        # Write structured event if the eval runner's event logger is available
        try:
            from agent_memory.eval.runner import get_event_logger
            evt = get_event_logger()
            if evt is not None:
                evt.log(
                    "llm_call",
                    call_id=self._call_count,
                    provider=self.provider,
                    model=resp.model or self.model,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    tool_calls=len(resp.tool_calls),
                    latency_ms=round(resp.latency_ms),
                    content_len=len(resp.content),
                    num_messages=len(messages),
                )
        except ImportError:
            pass

        return resp

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    async def _call_openai(
        self, system: str, messages: list[dict], temperature: float,
        max_tokens: int, tools: list[dict] | None,
    ) -> LLMResponse:
        client = self._get_openai_client()

        all_messages = ([{"role": "system", "content": system}] if system else []) + messages

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
        }

        # Certain endpoints (e.g. Azure OpenAI) require
        # 'max_completion_tokens' instead of the legacy 'max_tokens'.
        if self._needs_max_completion_tokens():
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage
        msg = choice.message

        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall.from_openai(tc))

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # OpenAI-Compatible (self-deployed GLM, Qwen, DeepSeek, etc.)
    # ------------------------------------------------------------------

    async def _call_openai_compat(
        self, system: str, messages: list[dict], temperature: float,
        max_tokens: int, tools: list[dict] | None,
    ) -> LLMResponse:
        """Call an OpenAI-compatible API (vLLM, SGLang, Ollama, self-deployed GLM/Qwen/etc.).

        Handles common compatibility issues:
        - Tool calling may not be supported → graceful fallback to text-only
        - Usage stats may be missing or differently structured
        - Some models return slightly different response formats
        """
        client = self._get_openai_client()

        all_messages = ([{"role": "system", "content": system}] if system else []) + messages

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
        }

        if self._needs_max_completion_tokens():
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        # Try with tools first; fall back to no-tools if the server doesn't support it
        use_tools = bool(tools)
        if use_tools:
            kwargs["tools"] = tools

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Common errors when server doesn't support tool calling
            if use_tools and any(
                kw in error_str
                for kw in ("tool", "function", "not supported", "unrecognized", "invalid", "400")
            ):
                logger.warning(
                    "OpenAI-compat server does not support tool calling, "
                    "falling back to text-only mode: %s", e,
                )
                kwargs.pop("tools", None)
                response = await client.chat.completions.create(**kwargs)
            else:
                logger.error("OpenAI-compat API call failed: %s", e)
                raise

        choice = response.choices[0]
        usage = response.usage
        msg = choice.message

        # Handle reasoning models (GLM-5.1, DeepSeek-R1, etc.)
        # These models return reasoning in `reasoning_content` and final answer in `content`.
        # When content is null but reasoning_content exists, the model may have been
        # truncated (max_tokens too low) — log a warning.
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or ""

        if not content and reasoning_content:
            logger.warning(
                "OpenAI-compat: content is empty but reasoning_content has %d chars. "
                "This usually means max_tokens is too low for this reasoning model. "
                "Falling back to reasoning_content.",
                len(reasoning_content),
            )
            content = reasoning_content
        if reasoning_content and content:
            logger.debug(
                "OpenAI-compat: reasoning model detected (reasoning=%d chars, content=%d chars)",
                len(reasoning_content), len(content),
            )

        # Parse tool calls (may not be present on compat servers)
        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall.from_openai(tc))

        # Fallback: parse XML-style <tool_call> tags from content.
        # Some models (Qwen3, Hermes-format, etc.) output tool calls as XML
        # in the content field when the serving framework (vLLM/SGLang) does
        # not have --enable-auto-tool-call enabled.
        if not tool_calls and content and "<tool_call>" in content:
            parsed_tcs, cleaned_content = self._parse_xml_tool_calls(content)
            if parsed_tcs:
                tool_calls = parsed_tcs
                # Remove the tool_call XML from content so downstream sees
                # only the reasoning/thinking text.
                content = cleaned_content
                logger.info(
                    "OpenAI-compat: parsed %d tool call(s) from XML tags in content",
                    len(tool_calls),
                )

        # Handle potentially missing usage fields
        input_tokens = 0
        output_tokens = 0
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            model=getattr(response, "model", self.model) or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # XML tool call parsing (Qwen3 / Hermes fallback)
    # ------------------------------------------------------------------

    _XML_TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL
    )
    _XML_FUNCTION_RE = re.compile(
        r"<function=([^>]+)>\s*(.+?)\s*</function>", re.DOTALL
    )
    _XML_PARAMETER_RE = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
    )

    def _parse_xml_tool_calls(self, content: str) -> tuple[list[ToolCall], str]:
        """Parse XML-style <tool_call> blocks from model content.

        Format (used by Qwen3/Hermes when vLLM auto-tool-call is disabled):
            <tool_call>
            <function=FUNCTION_NAME>
            <parameter=PARAM_NAME>
            PARAM_VALUE
            </parameter>
            </function>
            </tool_call>

        Returns:
            Tuple of (parsed_tool_calls, cleaned_content_without_xml_blocks).
        """
        tool_calls: list[ToolCall] = []
        call_id_counter = 0

        for match in self._XML_TOOL_CALL_RE.finditer(content):
            block = match.group(1)
            func_match = self._XML_FUNCTION_RE.search(block)
            if not func_match:
                # Try a simpler pattern: just function name without closing tag
                # e.g. <function=mem_ls\n</parameter>\n</function>
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            # Extract all parameters
            arguments: dict[str, Any] = {}
            for param_match in self._XML_PARAMETER_RE.finditer(func_body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                arguments[param_name] = param_value

            call_id_counter += 1
            tool_calls.append(ToolCall(
                id=f"xmlcall_{call_id_counter}",
                name=func_name,
                arguments=arguments,
            ))

        # Remove all <tool_call>...</tool_call> blocks from content
        cleaned = self._XML_TOOL_CALL_RE.sub("", content).strip()
        # Also remove trailing </think> tag if present (model may output thinking)
        if cleaned.endswith("</think>"):
            cleaned = cleaned[:-len("</think>")].strip()
        # Remove <think> wrapper if it's the only remaining content
        if cleaned.startswith("<think>") and "</think>" not in cleaned:
            cleaned = cleaned[len("<think>"):].strip()

        return tool_calls, cleaned

    # ------------------------------------------------------------------
    # Anthropic — adapted to OpenAI tool call format
    # ------------------------------------------------------------------

    async def _call_anthropic(
        self, system: str, messages: list[dict], temperature: float,
        max_tokens: int, tools: list[dict] | None,
    ) -> LLMResponse:
        client = self._get_anthropic_client()

        # Convert OpenAI-format messages to Anthropic format (strip tool_calls)
        anthropic_messages = self._convert_messages_for_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = MEMORY_TOOLS_ANTHROPIC

        response = await client.messages.create(**kwargs)

        # Parse content blocks → extract text + tool_use
        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                    raw=block,
                ))

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response,
        )

    @staticmethod
    def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages (with tool_calls/tool results) to Anthropic format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "tool":
                # OpenAI tool result → Anthropic user message with tool_result block
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                })
            elif role == "assistant" and "tool_calls" in msg:
                # OpenAI assistant with tool_calls → Anthropic assistant with tool_use blocks
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"command": args}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                result.append({"role": "assistant", "content": content_blocks})
            else:
                # Normal user/assistant message
                result.append({"role": role, "content": msg.get("content", "")})

        return result

    # ------------------------------------------------------------------
    # Streaming with TTFT (Time To First Token) measurement
    # ------------------------------------------------------------------

    async def generate_with_ttft(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[LLMResponse, float]:
        """使用 streaming 模式调用 LLM，返回完整响应和 TTFT（首 token 时延）。

        通过 streaming 模式获取真实的首 token 时延（Time To First Token），
        同时拼接完整响应内容，最终返回与 generate() 相同格式的 LLMResponse。

        注意：此方法不支持 tool calling，仅用于评测场景的纯文本生成。

        Args:
            system: System prompt.
            messages: Conversation messages (OpenAI format).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            元组 (LLMResponse, ttft_seconds)。
            ttft_seconds: 从请求发出到收到第一个内容 token 的时间（秒）。
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens or self.max_tokens

        start = time.monotonic()
        ttft = 0.0
        first_token_received = False
        content_parts: list[str] = []
        model_name = ""
        input_tokens = 0
        output_tokens = 0

        if self.provider in ("openai", "openai_compat"):
            client = self._get_openai_client()
            all_messages = ([{"role": "system", "content": system}] if system else []) + messages

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": all_messages,
                "temperature": temp,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if self._needs_max_completion_tokens():
                kwargs["max_completion_tokens"] = max_tok
            else:
                kwargs["max_tokens"] = max_tok

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                # 提取 usage（通常在最后一个 chunk 中）
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                model_name = getattr(chunk, "model", "") or ""

                if delta and delta.content:
                    if not first_token_received:
                        ttft = time.monotonic() - start
                        first_token_received = True
                    content_parts.append(delta.content)

        elif self.provider == "anthropic":
            # Anthropic streaming
            client = self._get_anthropic_client()
            anthropic_messages = self._convert_messages_for_anthropic(messages)

            kwargs_ant: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }
            if system:
                kwargs_ant["system"] = system

            async with client.messages.stream(**kwargs_ant) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta" and hasattr(event, "delta"):
                            text = getattr(event.delta, "text", "")
                            if text:
                                if not first_token_received:
                                    ttft = time.monotonic() - start
                                    first_token_received = True
                                content_parts.append(text)

                # 获取最终消息的 usage
                final_message = await stream.get_final_message()
                model_name = final_message.model
                input_tokens = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens
        else:
            raise ValueError(f"Unsupported provider for streaming: {self.provider}")

        total_latency_ms = (time.monotonic() - start) * 1000
        content = "".join(content_parts)

        resp = LLMResponse(
            content=content,
            model=model_name or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=total_latency_ms,
        )

        # 更新累计统计
        self._total_input_tokens += resp.input_tokens
        self._total_output_tokens += resp.output_tokens
        self._call_count += 1
        self._total_llm_latency_ms += total_latency_ms

        logger.info(
            "LLM streaming response #%d: model=%s, in=%d, out=%d, "
            "latency=%.0fms, ttft=%.0fms, content_len=%d",
            self._call_count, resp.model, resp.input_tokens, resp.output_tokens,
            resp.latency_ms, ttft * 1000, len(resp.content),
        )

        return resp, ttft
