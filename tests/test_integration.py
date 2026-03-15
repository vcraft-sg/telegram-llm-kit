"""Integration test covering the full message flow with real SQLite + ChromaDB."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import chromadb
import pytest

from telegram_llm_kit.bot.handlers import HandlerDependencies, message_handler, search_handler
from telegram_llm_kit.llm.base import LLMResponse
from telegram_llm_kit.rag.embeddings import EmbeddingService
from telegram_llm_kit.rag.retriever import Retriever
from telegram_llm_kit.rag.store import VectorStore
from telegram_llm_kit.storage.llm_call_repo import LLMCallRepository
from telegram_llm_kit.storage.message_repo import MessageRepository


@pytest.fixture
def integration_deps(tmp_db):
    """Full integration dependencies: real SQLite + ChromaDB, mock LLM + embeddings."""
    message_repo = MessageRepository(tmp_db)
    llm_call_repo = LLMCallRepository(tmp_db)

    # Real ChromaDB (ephemeral)
    chroma_client = chromadb.Client()
    collection_name = f"test-{uuid.uuid4().hex[:8]}"
    vector_store = VectorStore(chroma_client, collection_name=collection_name)

    # Mock embedding service (returns deterministic vectors)
    embedding_service = MagicMock(spec=EmbeddingService)
    call_count = 0

    def fake_embed(text):
        nonlocal call_count
        call_count += 1
        # Generate a unique but deterministic embedding per call
        return [float(call_count), float(len(text) % 10), 0.5]

    embedding_service.embed.side_effect = fake_embed

    retriever = Retriever(
        message_repo=message_repo,
        vector_store=vector_store,
        embedding_service=embedding_service,
        recency_count=20,
        semantic_count=10,
    )

    # Mock LLM provider
    llm_provider = AsyncMock()
    llm_provider.provider_name = "test"
    llm_provider.model_name = "test-model"
    llm_provider.complete.return_value = LLMResponse(
        content="I'm a test response!",
        input_tokens=20,
        output_tokens=8,
        model="test-model",
        raw_request={"messages": []},
        raw_response={"choices": []},
        latency_ms=50,
    )

    deps = HandlerDependencies(
        message_repo=message_repo,
        llm_call_repo=llm_call_repo,
        llm_provider=llm_provider,
        retriever=retriever,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )
    return deps


def _make_update(text="hello", chat_id=123, message_id=1):
    update = MagicMock()
    update.message.text = text
    update.message.chat_id = chat_id
    update.message.message_id = message_id
    update.message.reply_text = AsyncMock()
    return update


def _make_context(deps):
    context = MagicMock()
    context.bot_data = {"deps": deps}
    return context


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_message_flow(self, integration_deps):
        """End-to-end: send message → save → embed → retrieve → LLM → reply → log."""
        deps = integration_deps
        update = _make_update(text="What is Python?", message_id=42)
        context = _make_context(deps)

        await message_handler(update, context)

        # Verify user message was saved to SQLite
        messages = deps.message_repo.get_recent(limit=10)
        assert len(messages) == 2  # user + assistant
        assert messages[0].role == "user"
        assert messages[0].content == "What is Python?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "I'm a test response!"

        # Verify both messages have chroma_ids
        assert messages[0].chroma_id is not None
        assert messages[1].chroma_id is not None

        # Verify LLM call was logged
        llm_call = deps.llm_call_repo.get_by_message_id(messages[1].id)
        assert llm_call is not None
        assert llm_call.input_tokens == 20
        assert llm_call.output_tokens == 8

        # Verify response was sent
        update.message.reply_text.assert_called_once_with("I'm a test response!")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, integration_deps):
        """Multiple messages build up context for subsequent calls."""
        deps = integration_deps

        # First message
        update1 = _make_update(text="My name is Alice", message_id=1)
        context = _make_context(deps)
        await message_handler(update1, context)

        # Second message — LLM should receive context from first exchange
        update2 = _make_update(text="What is my name?", message_id=2)
        await message_handler(update2, context)

        # Verify LLM was called twice
        assert deps.llm_provider.complete.call_count == 2

        # Second call should have context messages from first exchange
        second_call_messages = deps.llm_provider.complete.call_args_list[1][1]["messages"]
        # Should have system + previous messages + current
        assert len(second_call_messages) > 2
        # The conversation context should include "My name is Alice"
        all_content = " ".join(m["content"] for m in second_call_messages)
        assert "My name is Alice" in all_content

    @pytest.mark.asyncio
    async def test_search_after_messages(self, integration_deps):
        """Search finds messages that were previously sent."""
        deps = integration_deps

        # Send some messages
        for i, text in enumerate(["Python is great", "I love coding", "The weather is nice"]):
            update = _make_update(text=text, message_id=i + 1)
            context = _make_context(deps)
            await message_handler(update, context)

        # Search via FTS
        search_update = _make_update(text="/search Python")
        search_context = _make_context(deps)
        search_context.args = ["Python"]
        await search_handler(search_update, search_context)

        reply = search_update.message.reply_text.call_args[0][0]
        assert "Python is great" in reply

    @pytest.mark.asyncio
    async def test_fts_triggers_work(self, integration_deps):
        """FTS5 triggers properly index messages for search."""
        deps = integration_deps

        update = _make_update(text="ChromaDB is a vector database", message_id=1)
        context = _make_context(deps)
        await message_handler(update, context)

        # Direct FTS search should find the message
        results = deps.message_repo.search_fts("vector database")
        assert len(results) >= 1
        assert any("vector database" in r.content for r in results)
