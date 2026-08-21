import httpx
import json
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
