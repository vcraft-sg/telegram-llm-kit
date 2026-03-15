import pytest
import respx
from httpx import Response

from telegram_llm_kit.llm.deepseek import DeepSeekProvider


class TestDeepSeekProvider:
    def test_provider_name(self):
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.provider_name == "deepseek"

    def test_default_model(self):
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.model_name == "deepseek-chat"

    def test_custom_model(self):
        provider = DeepSeekProvider(api_key="test-key", model="deepseek-reasoner")
        assert provider.model_name == "deepseek-reasoner"

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete(self):
        """complete() sends correct request and parses response."""
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "model": "deepseek-chat",
                    "choices": [{"message": {"content": "Hi there!"}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 5},
                },
            )
        )

        provider = DeepSeekProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
        )

        assert result.content == "Hi there!"
        assert result.input_tokens == 15
        assert result.output_tokens == 5
        assert result.model == "deepseek-chat"
        assert result.latency_ms >= 0
        assert result.raw_request["model"] == "deepseek-chat"
        assert result.raw_request["temperature"] == 0.5
        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_error(self):
        """complete() raises on HTTP error."""
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=Response(429, json={"error": "rate limited"})
        )

        provider = DeepSeekProvider(api_key="test-key")
        with pytest.raises(Exception):
            await provider.complete(messages=[{"role": "user", "content": "Hello"}])
        await provider.close()
