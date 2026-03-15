import time

import httpx

from telegram_llm_kit.llm.base import LLMProvider, LLMResponse


class ClaudeProvider(LLMProvider):
    """Claude LLM provider using the Anthropic Messages API via httpx."""

    BASE_URL = "https://api.anthropic.com"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str | None = None):
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Anthropic Messages API expects system as a top-level param, not in messages
        system_text = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        request_payload = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            request_payload["system"] = system_text

        start = time.monotonic()
        response = await self._client.post("/v1/messages", json=request_payload)
        latency_ms = int((time.monotonic() - start) * 1000)

        response.raise_for_status()
        data = response.json()

        # Extract text from content blocks
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]

        return LLMResponse(
            content=content,
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            model=data.get("model", self._model),
            raw_request=request_payload,
            raw_response=data,
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        await self._client.aclose()
