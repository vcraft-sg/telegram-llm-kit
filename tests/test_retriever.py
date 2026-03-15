from unittest.mock import MagicMock

import pytest

from telegram_llm_kit.rag.retriever import Retriever
from telegram_llm_kit.storage.models import Message


class TestRetriever:
    @pytest.fixture
    def deps(self):
        """Create mock dependencies for the retriever."""
        message_repo = MagicMock()
        vector_store = MagicMock()
        embedding_service = MagicMock()
        embedding_service.embed.return_value = [0.1, 0.2, 0.3]
        return message_repo, vector_store, embedding_service

    def test_retrieve_combines_sources(self, deps):
        """retrieve() returns both recent and semantic messages."""
        message_repo, vector_store, embedding_service = deps
        recent_msgs = [Message(id=5, role="user", content="recent")]
        message_repo.get_recent.return_value = recent_msgs
        vector_store.query.return_value = [
            {"id": "msg-1", "text": "old relevant", "distance": 0.1}
        ]
        semantic_msgs = [Message(id=1, role="user", content="old relevant")]
        message_repo.get_by_ids.return_value = semantic_msgs

        retriever = Retriever(
            message_repo, vector_store, embedding_service,
            recency_count=20, semantic_count=10,
        )
        recent, semantic = retriever.retrieve("test query")

        assert recent == recent_msgs
        assert semantic == semantic_msgs
        embedding_service.embed.assert_called_once_with("test query")
        vector_store.query.assert_called_once()

    def test_retrieve_empty_semantic(self, deps):
        """retrieve() handles no semantic results gracefully."""
        message_repo, vector_store, embedding_service = deps
        message_repo.get_recent.return_value = []
        vector_store.query.return_value = []

        retriever = Retriever(message_repo, vector_store, embedding_service)
        recent, semantic = retriever.retrieve("query")

        assert recent == []
        assert semantic == []

    def test_retrieve_passes_counts(self, deps):
        """retrieve() uses configured recency and semantic counts."""
        message_repo, vector_store, embedding_service = deps
        message_repo.get_recent.return_value = []
        vector_store.query.return_value = []

        retriever = Retriever(
            message_repo, vector_store, embedding_service,
            recency_count=5, semantic_count=3,
        )
        retriever.retrieve("query")

        message_repo.get_recent.assert_called_once_with(limit=5)
        vector_store.query.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3], n_results=3
        )

    def test_retrieve_parses_chroma_ids(self, deps):
        """retrieve() extracts SQLite IDs from ChromaDB doc IDs."""
        message_repo, vector_store, embedding_service = deps
        message_repo.get_recent.return_value = []
        vector_store.query.return_value = [
            {"id": "msg-42", "text": "a", "distance": 0.1},
            {"id": "msg-7", "text": "b", "distance": 0.2},
        ]
        message_repo.get_by_ids.return_value = []

        retriever = Retriever(message_repo, vector_store, embedding_service)
        retriever.retrieve("query")

        message_repo.get_by_ids.assert_called_once_with([42, 7])
