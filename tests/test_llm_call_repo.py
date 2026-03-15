import pytest

from telegram_llm_kit.storage.llm_call_repo import LLMCallRepository
from telegram_llm_kit.storage.message_repo import MessageRepository
from telegram_llm_kit.storage.models import LLMCall, Message


class TestLLMCallRepository:
    @pytest.fixture
    def repos(self, tmp_db):
        return MessageRepository(tmp_db), LLMCallRepository(tmp_db)

    def test_save_and_retrieve(self, repos):
        """Saving an LLM call assigns id and created_at."""
        msg_repo, call_repo = repos
        msg = msg_repo.save(Message(role="assistant", content="response"))
        call = LLMCall(
            provider="deepseek",
            model="deepseek-chat",
            request_payload='{"messages": []}',
            response_payload='{"choices": []}',
            input_tokens=10,
            output_tokens=20,
            latency_ms=150,
            message_id=msg.id,
        )
        saved = call_repo.save(call)
        assert saved.id is not None
        assert saved.created_at is not None

    def test_get_by_message_id(self, repos):
        """get_by_message_id returns the associated call."""
        msg_repo, call_repo = repos
        msg = msg_repo.save(Message(role="assistant", content="resp"))
        call = LLMCall(
            provider="claude",
            model="claude-sonnet",
            request_payload="{}",
            response_payload="{}",
            input_tokens=5,
            output_tokens=15,
            latency_ms=200,
            message_id=msg.id,
        )
        call_repo.save(call)
        found = call_repo.get_by_message_id(msg.id)
        assert found is not None
        assert found.provider == "claude"
        assert found.input_tokens == 5

    def test_get_by_message_id_not_found(self, repos):
        """get_by_message_id returns None when no call exists."""
        _, call_repo = repos
        assert call_repo.get_by_message_id(999) is None

    def test_save_with_error(self, repos):
        """LLM calls can record errors."""
        _, call_repo = repos
        call = LLMCall(
            provider="deepseek",
            model="deepseek-chat",
            request_payload="{}",
            response_payload="",
            error="Connection timeout",
        )
        saved = call_repo.save(call)
        assert saved.error == "Connection timeout"
