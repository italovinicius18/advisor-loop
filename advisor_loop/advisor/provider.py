import os
import httpx
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    message: dict
    tool_calls: list = field(default_factory=list)
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)


class OpenRouterClient:
    """OpenRouter LLM client for chat completions with real-cost tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        transport: httpx.BaseTransport | None = None,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: API key (uses OPENROUTER_API_KEY env var if None)
            base_url: OpenRouter API base URL
            transport: Custom httpx transport (for testing)
        """
        load_dotenv()
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.http = httpx.Client(
            base_url=base_url,
            timeout=120,
            transport=transport,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to OpenRouter.

        Args:
            model: Model identifier (e.g. "deepseek/deepseek-v4-flash-0731")
            messages: Chat messages
            tools: Optional tool definitions for function calling
            temperature: Sampling temperature (default 0.2)
            max_tokens: Optional output token cap (default None = unbounded)

        Returns:
            LLMResponse with message, tool_calls, cost_usd, and usage
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "usage": {"include": True},
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = self.http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        return LLMResponse(
            message=msg,
            tool_calls=msg.get("tool_calls") or [],
            cost_usd=data.get("usage", {}).get("cost", 0.0),
            usage=data.get("usage", {}),
        )
