import litellm
from litellm import CustomLLM
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Union,
)

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import ModelResponse
from litellm import CustomStreamWrapper

# The mock response used by FakeLLM
MOCK_RESPONSE = "Hi! I'm FakeLLM, a mock model for testing."
FAKE_MODEL = "gpt-3.5-turbo"

class FakeLLM(CustomLLM):
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        return litellm.completion(
            model=FAKE_MODEL,
            messages=messages,
            mock_response=MOCK_RESPONSE,
        )  # type: ignore

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        # LiteLLM's CustomLLM framework expects GenericStreamingChunk dicts,
        # not ModelResponseStream objects.
        mock_chunks = MOCK_RESPONSE.split(" ")
        for i, word in enumerate(mock_chunks):
            is_last = i == len(mock_chunks) - 1
            text = word if i == 0 else " " + word
            yield GenericStreamingChunk(
                text=text,
                is_finished=is_last,
                finish_reason="stop" if is_last else None,
                usage=None,
                tool_use=None,
            )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[
        Coroutine[Any, Any, Union[ModelResponse, "CustomStreamWrapper"]],
        Union[ModelResponse, "CustomStreamWrapper"],
    ]:
        return litellm.completion(
            model=FAKE_MODEL,
            messages=messages,
            mock_response=MOCK_RESPONSE,
        )  # type: ignore

    async def astreaming(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        # IMPORTANT: This must be an async generator function (using yield),
        # NOT a regular async function that returns an async iterator.
        # LiteLLM's CustomLLM framework expects GenericStreamingChunk dicts,
        # not ModelResponseStream objects.
        mock_chunks = MOCK_RESPONSE.split(" ")
        for i, word in enumerate(mock_chunks):
            is_last = i == len(mock_chunks) - 1
            text = word if i == 0 else " " + word
            yield GenericStreamingChunk(
                text=text,
                is_finished=is_last,
                finish_reason="stop" if is_last else None,
                usage=None,
                tool_use=None,
            )


# Module-level instance used by litellm's custom_provider_map config.
# In config.yaml, reference as: gateway.fake_llm.fake_llm
fake_llm = FakeLLM()