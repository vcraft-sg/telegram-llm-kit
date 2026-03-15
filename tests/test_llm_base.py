from telegram_llm_kit.llm.base import LLMProvider, LLMResponse


class TestLLMResponse:
    def test_dataclass_fields(self):
        """LLMResponse holds all expected fields."""
        resp = LLMResponse(
            content="hello",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            raw_request={"messages": []},
            raw_response={"choices": []},
            latency_ms=100,
        )
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "test-model"
        assert resp.latency_ms == 100


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        """LLMProvider is abstract and cannot be instantiated directly."""
        import pytest

        with pytest.raises(TypeError):
            LLMProvider()
