import chromadb


class VectorStore:
    """Manages a ChromaDB collection with pre-computed embeddings."""

    def __init__(self, client: chromadb.ClientAPI, collection_name: str = "messages"):
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, doc_id: str, text: str, embedding: list[float]) -> None:
        """Add a single document with its pre-computed embedding."""
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
        )

    def add_batch(
        self,
        doc_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add multiple documents with their pre-computed embeddings."""
        self._collection.add(
            ids=doc_ids,
            documents=texts,
            embeddings=embeddings,
        )

    def query(self, embedding: list[float], n_results: int = 10) -> list[dict]:
        """Query for similar documents. Returns list of {id, text, distance}."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._collection.count()),
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return output
