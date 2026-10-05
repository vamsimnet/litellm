import json
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-5.6-luna:zdr"])
@pytest.mark.parametrize("explicit_provider", [False, True])
def test_fastrouter_resolves_nested_model(monkeypatch: pytest.MonkeyPatch, model: str, explicit_provider: bool) -> None:
    monkeypatch.setenv("FASTROUTER_API_KEY", "test-fastrouter-key")
    monkeypatch.delenv("FASTROUTER_API_BASE", raising=False)
    resolved: Final = get_llm_provider(
        model=model if explicit_provider else f"fastrouter/{model}",
        custom_llm_provider="fastrouter" if explicit_provider else None,
    )
    assert resolved == (model, "fastrouter", "test-fastrouter-key", "https://api.fastrouter.ai")


def test_fastrouter_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTROUTER_API_KEY", "env-key")
    monkeypatch.setenv("FASTROUTER_API_BASE", "https://env.example")
    assert get_llm_provider(model="fastrouter/openai/gpt-5.6-luna")[2:] == ("env-key", "https://env.example")
    assert get_llm_provider(
        model="fastrouter/openai/gpt-5.6-luna", api_key="explicit-key", api_base="https://explicit.example"
    )[2:] == ("explicit-key", "https://explicit.example")


def test_fastrouter_detected_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTROUTER_API_KEY", "test-fastrouter-key")
    resolved: Final = get_llm_provider(model="test-model", api_base="https://api.fastrouter.ai")
    assert resolved == ("test-model", "fastrouter", "test-fastrouter-key", "https://api.fastrouter.ai")


def _respond(request: httpx.Request) -> httpx.Response:
    body: Final = json.loads(request.content)
    assert str(request.url) == "https://api.fastrouter.ai/chat/completions"
    assert request.headers["authorization"] == "Bearer test-fastrouter-key"
    assert body["model"] == "openai/gpt-5.6-luna:zdr"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["max_completion_tokens"] == 32
    if body.get("stream"):
        chunks: Final = (
            {"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        )
        events: Final = tuple(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [chunk],
                }
            )
            + "\n\n"
            for chunk in chunks
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=":FASTROUTER PROCESSING\n\n" + "".join(events) + "data: [DONE]\n\n",
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "response",
            "model": "gpt-5.6-luna",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


@pytest.mark.parametrize("stream", [False, True])
def test_fastrouter_completion(stream: bool) -> None:
    with OpenAI(
        api_key="test-fastrouter-key",
        base_url="https://api.fastrouter.ai",
        http_client=httpx.Client(transport=httpx.MockTransport(_respond)),
    ) as client:
        response: Final = litellm.completion(
            model="fastrouter/openai/gpt-5.6-luna:zdr",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-fastrouter-key",
            max_completion_tokens=32,
            stream=stream,
            client=client,
        )
        if stream:
            assert isinstance(response, litellm.CustomStreamWrapper)
            chunks: Final = tuple(response)
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.usage.total_tokens == 3
            assert response.object == "chat.completion"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_fastrouter_async_completion(stream: bool) -> None:
    async with AsyncOpenAI(
        api_key="test-fastrouter-key",
        base_url="https://api.fastrouter.ai",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_respond)),
    ) as client:
        response: Final = await litellm.acompletion(
            model="fastrouter/openai/gpt-5.6-luna:zdr",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-fastrouter-key",
            max_completion_tokens=32,
            stream=stream,
            client=client,
        )
        if stream:
            assert isinstance(response, litellm.CustomStreamWrapper)
            chunks: Final = [chunk async for chunk in response]
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.usage.total_tokens == 3
            assert response.object == "chat.completion"
