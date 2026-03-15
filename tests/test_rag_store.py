import uuid

import chromadb
import pytest

from telegram_llm_kit.rag.store import VectorStore


class TestVectorStore:
    @pytest.fixture
    def store(self):
        """Create a VectorStore backed by an ephemeral ChromaDB client with a unique collection."""
        client = chromadb.Client()
        name = f"test-{uuid.uuid4().hex[:8]}"
        return VectorStore(client, collection_name=name)

    def test_add_and_query(self, store):
        """Adding a document makes it queryable by similarity."""
        store.add(doc_id="msg-1", text="hello world", embedding=[1.0, 0.0, 0.0])
        store.add(doc_id="msg-2", text="goodbye world", embedding=[0.0, 1.0, 0.0])

        results = store.query(embedding=[1.0, 0.1, 0.0], n_results=1)
        assert len(results) == 1
        assert results[0]["id"] == "msg-1"

    def test_query_returns_metadata(self, store):
        """Query results include id, text, and distance."""
        store.add(doc_id="msg-1", text="test doc", embedding=[1.0, 0.0])
        results = store.query(embedding=[1.0, 0.0], n_results=1)
        assert "id" in results[0]
        assert "text" in results[0]
        assert "distance" in results[0]
        assert results[0]["text"] == "test doc"

    def test_query_n_results(self, store):
        """Query respects n_results limit."""
        for i in range(5):
            store.add(doc_id=f"msg-{i}", text=f"doc {i}", embedding=[float(i), 0.0])
        results = store.query(embedding=[3.0, 0.0], n_results=2)
        assert len(results) == 2

    def test_query_empty_collection(self, store):
        """Querying an empty collection returns empty results."""
        results = store.query(embedding=[1.0, 0.0], n_results=5)
        assert results == []

    def test_add_batch(self, store):
        """add_batch inserts multiple documents at once."""
        store.add_batch(
            doc_ids=["a", "b"],
            texts=["first", "second"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        results = store.query(embedding=[1.0, 0.0], n_results=2)
        assert len(results) == 2
