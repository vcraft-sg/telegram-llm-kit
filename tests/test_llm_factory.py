import pytest

from telegram_llm_kit.llm.claude import ClaudeProvider
from telegram_llm_kit.llm.deepseek import DeepSeekProvider
from telegram_llm_kit.llm.factory import create_llm_provider


class TestFactory:
    def test_creates_deepseek(self):
        provider = create_llm_provider("deepseek", "key")
        assert isinstance(provider, DeepSeekProvider)

    def test_creates_claude(self):
        provider = create_llm_provider("claude", "key")
        assert isinstance(provider, ClaudeProvider)

    def test_passes_model(self):
        provider = create_llm_provider("deepseek", "key", model="deepseek-reasoner")
        assert provider.model_name == "deepseek-reasoner"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider("openai", "key")
