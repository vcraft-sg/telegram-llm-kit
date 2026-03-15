import time

import httpx

from telegram_llm_kit.llm.base import LLMProvider, LLMResponse


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider using the OpenAI-compatible API via httpx."""

    BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str, model: str | None = None):
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        request_payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.monotonic()
        response = await self._client.post("/chat/completions", json=request_payload)
        latency_ms = int((time.monotonic() - start) * 1000)

        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            model=data.get("model", self._model),
            raw_request=request_payload,
            raw_response=data,
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        await self._client.aclose()
