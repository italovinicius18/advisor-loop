import httpx
import json
import pytest
from advisor_loop.advisor.provider import OpenRouterClient


def _mock(payload):
    """Create a mock transport that validates request and returns payload."""
    def handler(request):
        body = json.loads(request.content)
        assert body["usage"] == {"include": True} and body["temperature"] == 0.2
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)


def test_chat_parses_cost_and_tool_calls():
    """Test that chat() parses cost and tool_calls from OpenRouter response."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "table_stats", "arguments": '{"table":"t"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00042},
    }
    c = OpenRouterClient(api_key="k", transport=_mock(payload))
    r = c.chat("deepseek/deepseek-v4-flash-0731", [{"role": "user", "content": "oi"}])
    assert r.cost_usd == 0.00042
    assert r.tool_calls[0]["function"]["name"] == "table_stats"


def test_chat_forwards_max_tokens_when_set():
    """max_tokens must be forwarded in the payload when the caller opts in."""
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured.update(body)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}],
                                          "usage": {"cost": 0.0}})

    c = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    c.chat("m", [{"role": "user", "content": "oi"}], max_tokens=512)
    assert captured["max_tokens"] == 512


def test_chat_omits_max_tokens_by_default():
    """Default (max_tokens=None) must not add the key - preserves old behavior."""
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured.update(body)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}],
                                          "usage": {"cost": 0.0}})

    c = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    c.chat("m", [{"role": "user", "content": "oi"}])
    assert "max_tokens" not in captured


def test_chat_forwards_tool_choice_and_reasoning():
    """tool_choice + reasoning must reach the payload when set (keeps a flash
    model fast and on-protocol); absent by default to preserve old behavior."""
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}],
                                          "usage": {"cost": 0.0}})

    c = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    c.chat("m", [{"role": "user", "content": "oi"}])
    assert "tool_choice" not in captured and "reasoning" not in captured

    c.chat("m", [{"role": "user", "content": "oi"}],
           tool_choice="required", reasoning={"enabled": False})
    assert captured["tool_choice"] == "required"
    assert captured["reasoning"] == {"enabled": False}


def test_chat_retries_transient_network_error():
    """A dropped connection (RemoteProtocolError) must be retried, not fatal.

    The OpenRouter sweep died mid-run because one incomplete chunked read
    killed the whole batch. chat() must survive transient blips."""
    calls = {"n": 0}
    ok = {"choices": [{"message": {"role": "assistant"}}], "usage": {"cost": 0.0}}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("peer closed connection")
        return httpx.Response(200, json=ok)

    c = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler),
                         max_retries=5, sleep=lambda _s: None)
    r = c.chat("m", [{"role": "user", "content": "oi"}])
    assert calls["n"] == 3          # failed twice, succeeded on the third
    assert r.cost_usd == 0.0


def test_chat_gives_up_after_max_retries():
    """After exhausting retries the transient error propagates (honest failure)."""
    def handler(request):
        raise httpx.RemoteProtocolError("peer closed connection")

    c = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler),
                         max_retries=3, sleep=lambda _s: None)
    with pytest.raises(httpx.RemoteProtocolError):
        c.chat("m", [{"role": "user", "content": "oi"}])
