
import pytest

from telegram_llm_kit.config import Settings


class TestSettings:
    def test_required_fields(self, monkeypatch):
        """Settings raises if required env vars are missing."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_defaults(self, monkeypatch):
        """Optional fields have sensible defaults."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("LLM_API_KEY", "key")
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        s = Settings(_env_file=None)
        assert s.telegram_bot_token == "tok"
        assert s.llm_api_key == "key"
        assert s.llm_provider == "deepseek"
        assert s.llm_temperature == 0.7
        assert s.llm_max_tokens == 1024
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.recency_count == 20
        assert s.semantic_count == 10
        assert s.sqlite_db_path == "data/bot.db"
        assert s.chroma_persist_dir == "data/chroma"
        assert s.log_level == "INFO"

    def test_provider_validation(self, monkeypatch):
        """Only 'deepseek' and 'claude' are valid providers."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("LLM_API_KEY", "key")
        monkeypatch.setenv("LLM_PROVIDER", "invalid")
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_custom_values(self, monkeypatch):
        """All fields can be overridden via env vars."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("LLM_API_KEY", "key456")
        monkeypatch.setenv("LLM_PROVIDER", "claude")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
        monkeypatch.setenv("RECENCY_COUNT", "30")
        monkeypatch.setenv("SEMANTIC_COUNT", "5")
        s = Settings(_env_file=None)
        assert s.llm_model == "claude-sonnet-4-6"
        assert s.llm_temperature == 0.5
        assert s.llm_max_tokens == 2048
        assert s.recency_count == 30
        assert s.semantic_count == 5
