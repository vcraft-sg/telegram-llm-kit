from telegram_llm_kit.llm.base import LLMProvider
from telegram_llm_kit.llm.claude import ClaudeProvider
from telegram_llm_kit.llm.deepseek import DeepSeekProvider


def create_llm_provider(
    provider: str, api_key: str, model: str | None = None
) -> LLMProvider:
    """Create an LLM provider instance from a config string."""
    if provider == "deepseek":
        return DeepSeekProvider(api_key=api_key, model=model)
    elif provider == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Must be 'deepseek' or 'claude'.")
