import pytest
import respx
from httpx import Response

from telegram_llm_kit.llm.claude import ClaudeProvider


class TestClaudeProvider:
    def test_provider_name(self):
        provider = ClaudeProvider(api_key="test-key")
        assert provider.provider_name == "claude"

    def test_default_model(self):
        provider = ClaudeProvider(api_key="test-key")
        assert provider.model_name == "claude-sonnet-4-6"

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete(self):
        """complete() sends correct request and parses Anthropic response."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            )
        )

        provider = ClaudeProvider(api_key="test-key")
        result = await provider.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            temperature=0.5,
            max_tokens=200,
        )

        assert result.content == "Hello!"
        assert result.input_tokens == 12
        assert result.output_tokens == 3
        # System message should be extracted to top-level param
        assert result.raw_request["system"] == "You are helpful."
        assert len(result.raw_request["messages"]) == 1  # Only user message
        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_no_system(self):
        """complete() works without a system message."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "Response"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )
        )

        provider = ClaudeProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert result.content == "Response"
        assert "system" not in result.raw_request
        await provider.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_error(self):
        """complete() raises on HTTP error."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(401, json={"error": "unauthorized"})
        )

        provider = ClaudeProvider(api_key="test-key")
        with pytest.raises(Exception):
            await provider.complete(messages=[{"role": "user", "content": "Hi"}])
        await provider.close()
