from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram_llm_kit.bot.handlers import (
    HandlerDependencies,
    message_handler,
    search_handler,
    start_handler,
)
from telegram_llm_kit.llm.base import LLMResponse
from telegram_llm_kit.storage.models import Message


def _make_deps(**overrides):
    """Create a HandlerDependencies with all mocks."""
    defaults = {
        "message_repo": MagicMock(),
        "llm_call_repo": MagicMock(),
        "llm_provider": AsyncMock(),
        "retriever": MagicMock(),
        "vector_store": MagicMock(),
        "embedding_service": MagicMock(),
    }
    defaults.update(overrides)
    return HandlerDependencies(**defaults)


def _make_update(text="hello", chat_id=123, message_id=1):
    """Create a mock Telegram Update."""
    update = MagicMock()
    update.message.text = text
    update.message.chat_id = chat_id
    update.message.message_id = message_id
    update.message.reply_text = AsyncMock()
    return update


def _make_context(deps):
    """Create a mock context with deps in bot_data."""
    context = MagicMock()
    context.bot_data = {"deps": deps}
    return context


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_sends_welcome(self):
        update = _make_update()
        context = _make_context(_make_deps())
        await start_handler(update, context)
        update.message.reply_text.assert_called_once()
        assert "Hello" in update.message.reply_text.call_args[0][0]


class TestMessageHandler:
    @pytest.mark.asyncio
    async def test_full_flow(self):
        """message_handler saves, embeds, retrieves, calls LLM, and replies."""
        deps = _make_deps()
        # Configure mocks
        saved_user_msg = Message(id=1, role="user", content="hello")
        saved_assistant_msg = Message(id=2, role="assistant", content="Hi there!")
        deps.message_repo.save.side_effect = [saved_user_msg, saved_assistant_msg]
        deps.embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        deps.retriever.retrieve.return_value = ([], [])
        deps.llm_provider.complete.return_value = LLMResponse(
            content="Hi there!",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            raw_request={"messages": []},
            raw_response={"choices": []},
            latency_ms=100,
        )

        update = _make_update(text="hello")
        context = _make_context(deps)
        await message_handler(update, context)

        # Verify message was saved
        assert deps.message_repo.save.call_count == 2
        # Verify embedding was computed and stored
        assert deps.embedding_service.embed.call_count == 2  # user + assistant
        assert deps.vector_store.add.call_count == 2
        # Verify LLM was called
        deps.llm_provider.complete.assert_called_once()
        # Verify response was sent
        update.message.reply_text.assert_called_once_with("Hi there!")
        # Verify LLM call was logged
        deps.llm_call_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_error_handling(self):
        """message_handler handles LLM errors gracefully."""
        deps = _make_deps()
        deps.message_repo.save.return_value = Message(id=1, role="user", content="hi")
        deps.embedding_service.embed.return_value = [0.1, 0.2]
        deps.retriever.retrieve.return_value = ([], [])
        deps.llm_provider.complete.side_effect = Exception("API error")
        deps.llm_provider.provider_name = "test"
        deps.llm_provider.model_name = "test-model"

        update = _make_update()
        context = _make_context(deps)
        await message_handler(update, context)

        # Should reply with error message
        update.message.reply_text.assert_called_once()
        assert "error" in update.message.reply_text.call_args[0][0].lower()
        # Should log the failed call
        deps.llm_call_repo.save.assert_called_once()


class TestSearchHandler:
    @pytest.mark.asyncio
    async def test_empty_query(self):
        """search_handler asks for a query when none provided."""
        deps = _make_deps()
        update = _make_update()
        context = _make_context(deps)
        context.args = []
        await search_handler(update, context)
        assert "Usage" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_with_results(self):
        """search_handler returns both FTS and semantic results."""
        deps = _make_deps()
        deps.message_repo.search_fts.return_value = [
            Message(id=1, role="user", content="test result")
        ]
        deps.embedding_service.embed.return_value = [0.1, 0.2]
        deps.vector_store.query.return_value = [
            {"id": "msg-1", "text": "semantic result", "distance": 0.2}
        ]

        update = _make_update()
        context = _make_context(deps)
        context.args = ["test", "query"]
        await search_handler(update, context)

        reply = update.message.reply_text.call_args[0][0]
        assert "Text search" in reply
        assert "Semantic search" in reply

    @pytest.mark.asyncio
    async def test_no_results(self):
        """search_handler shows 'no results' when nothing matches."""
        deps = _make_deps()
        deps.message_repo.search_fts.return_value = []
        deps.embedding_service.embed.return_value = [0.1]
        deps.vector_store.query.return_value = []

        update = _make_update()
        context = _make_context(deps)
        context.args = ["nothing"]
        await search_handler(update, context)

        assert "No results" in update.message.reply_text.call_args[0][0]
