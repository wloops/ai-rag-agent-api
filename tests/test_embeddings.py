import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.utils.embeddings import embed_text, embed_texts


class EmbeddingsTestCase(unittest.TestCase):
    def test_embed_text_returns_single_vector(self):
        mock_response = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
        )
        mock_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **kwargs: mock_response)
        )

        with patch("app.utils.embeddings._create_embedding_client", return_value=mock_client):
            with patch("app.utils.embeddings.settings.embedding_dimensions", 3):
                result = embed_text("hello")

        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_embed_texts_returns_multiple_vectors(self):
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )
        mock_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **kwargs: mock_response)
        )

        with patch("app.utils.embeddings._create_embedding_client", return_value=mock_client):
            with patch("app.utils.embeddings.settings.embedding_dimensions", 3):
                result = embed_texts(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def test_embed_texts_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            embed_texts(["", "world"])

    def test_embed_texts_validates_dimensions(self):
        mock_response = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        mock_client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **kwargs: mock_response)
        )

        with patch("app.utils.embeddings._create_embedding_client", return_value=mock_client):
            with patch("app.utils.embeddings.settings.embedding_dimensions", 3):
                with self.assertRaises(RuntimeError):
                    embed_texts(["hello"])
