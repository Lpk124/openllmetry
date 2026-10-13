"""Regression tests for cached Anthropic response accessors after uninstrumenting."""

import inspect

import anthropic
import anthropic._base_client as base_client
import pytest

from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

httpx = getattr(base_client, "httpx2", None) or base_client.httpx


MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
REQUEST = {"model": "claude-test", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}


@pytest.mark.parametrize("accessor", ["plain", "with_raw_response", "with_streaming_response"])
def test_cached_response_accessor_stops_tracing_after_uninstrument(accessor, tracer_provider, span_exporter):
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    client = anthropic.Anthropic(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE))),
    )

    def call():
        if accessor == "plain":
            client.messages.create(**REQUEST)
        elif accessor == "with_raw_response":
            client.messages.with_raw_response.create(**REQUEST).parse()
        else:
            with client.messages.with_streaming_response.create(**REQUEST) as response:
                response.parse()

    try:
        call()
        assert len(span_exporter.get_finished_spans()) == 1
        instrumentor.uninstrument()
        span_exporter.clear()
        call()
        assert len(span_exporter.get_finished_spans()) == 0
    finally:
        instrumentor.uninstrument()
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("accessor", ["plain", "with_raw_response", "with_streaming_response"])
async def test_cached_async_response_accessor_stops_tracing_after_uninstrument(
    accessor, tracer_provider, span_exporter
):
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    client = anthropic.AsyncAnthropic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE))),
    )

    async def call():
        if accessor == "plain":
            await client.messages.create(**REQUEST)
        elif accessor == "with_raw_response":
            response = await client.messages.with_raw_response.create(**REQUEST)
            parsed = response.parse()
            if inspect.isawaitable(parsed):
                await parsed
        else:
            async with client.messages.with_streaming_response.create(**REQUEST) as response:
                parsed = response.parse()
                if inspect.isawaitable(parsed):
                    await parsed

    try:
        await call()
        assert len(span_exporter.get_finished_spans()) == 1
        instrumentor.uninstrument()
        span_exporter.clear()
        await call()
        assert len(span_exporter.get_finished_spans()) == 0
    finally:
        instrumentor.uninstrument()
        await client.close()


def test_cached_accessor_stays_disabled_after_reinstrument(tracer_provider, span_exporter):
    instrumentor = AnthropicInstrumentor()
    client = anthropic.Anthropic(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE))),
    )

    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        cached_create = client.messages.with_raw_response.create
        cached_create(**REQUEST).parse()
        assert len(span_exporter.get_finished_spans()) == 1

        instrumentor.uninstrument()
        span_exporter.clear()
        instrumentor.instrument(tracer_provider=tracer_provider)
        cached_create(**REQUEST).parse()
        assert len(span_exporter.get_finished_spans()) == 0

        new_client = anthropic.Anthropic(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=MESSAGE))),
        )
        try:
            new_client.messages.with_raw_response.create(**REQUEST).parse()
            assert len(span_exporter.get_finished_spans()) == 1
        finally:
            new_client.close()
    finally:
        instrumentor.uninstrument()
        client.close()
