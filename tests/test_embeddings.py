from unittest.mock import MagicMock, patch

import numpy as np

from telegram_llm_kit.rag.embeddings import EmbeddingService


class TestEmbeddingService:
    def test_lazy_load(self):
        """Model is not loaded until first embed call."""
        service = EmbeddingService("test-model")
        assert service._model is None

    @patch("telegram_llm_kit.rag.embeddings.SentenceTransformer")
    def test_embed_single(self, mock_st_class):
        """embed() returns a list of floats for a single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        service = EmbeddingService("test-model")
        result = service.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(["hello"])

    @patch("telegram_llm_kit.rag.embeddings.SentenceTransformer")
    def test_embed_batch(self, mock_st_class):
        """embed_batch() returns a list of embedding lists."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st_class.return_value = mock_model

        service = EmbeddingService("test-model")
        result = service.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @patch("telegram_llm_kit.rag.embeddings.SentenceTransformer")
    def test_model_loaded_once(self, mock_st_class):
        """Model is loaded only once, even with multiple calls."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_st_class.return_value = mock_model

        service = EmbeddingService("test-model")
        service.embed("a")
        service.embed("b")

        mock_st_class.assert_called_once_with("test-model")
