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
