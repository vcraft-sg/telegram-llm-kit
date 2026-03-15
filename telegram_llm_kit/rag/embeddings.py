from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wraps SentenceTransformer with lazy loading and a simple API."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, returning a list of floats."""
        model = self._load_model()
        embeddings = model.encode([text])
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, returning a list of embedding lists."""
        model = self._load_model()
        embeddings = model.encode(texts)
        return [e.tolist() for e in embeddings]
