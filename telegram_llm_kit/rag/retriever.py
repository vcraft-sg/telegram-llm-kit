from telegram_llm_kit.rag.embeddings import EmbeddingService
from telegram_llm_kit.rag.store import VectorStore
from telegram_llm_kit.storage.message_repo import MessageRepository
from telegram_llm_kit.storage.models import Message


class Retriever:
    """Combines recency-based and semantic retrieval for context building."""

    def __init__(
        self,
        message_repo: MessageRepository,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        recency_count: int = 20,
        semantic_count: int = 10,
    ):
        self._message_repo = message_repo
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._recency_count = recency_count
        self._semantic_count = semantic_count

    def retrieve(self, query: str) -> tuple[list[Message], list[Message]]:
        """Retrieve context messages for a query.

        Returns (recent_messages, semantic_messages) as separate lists.
        The caller (context builder) handles deduplication and ordering.
        """
        recent = self._message_repo.get_recent(limit=self._recency_count)

        # Embed the query and find semantically similar messages
        query_embedding = self._embedding_service.embed(query)
        similar_results = self._vector_store.query(
            embedding=query_embedding, n_results=self._semantic_count
        )

        # Convert ChromaDB results to Message objects via their IDs
        semantic = []
        if similar_results:
            # ChromaDB doc IDs are "msg-{sqlite_id}"
            sqlite_ids = []
            for result in similar_results:
                try:
                    sqlite_id = int(result["id"].split("-")[1])
                    sqlite_ids.append(sqlite_id)
                except (IndexError, ValueError):
                    continue
            semantic = self._message_repo.get_by_ids(sqlite_ids)

        return recent, semantic
