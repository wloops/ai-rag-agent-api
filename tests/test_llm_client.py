import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.utils.llm_client import generate_answer


class LlmClientTestCase(unittest.TestCase):
    def test_generate_answer_returns_text(self):
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这是答案 [S1]")
                )
            ]
        )
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: mock_response)
            )
        )

        with patch("app.utils.llm_client._create_llm_client", return_value=mock_client):
            with patch("app.utils.llm_client.settings.llm_api_key", "test-key"):
                result = generate_answer("system", "user")

        self.assertEqual(result, "这是答案 [S1]")

    def test_generate_answer_raises_when_key_missing(self):
        with patch("app.utils.llm_client.settings.llm_api_key", ""):
            with self.assertRaises(RuntimeError):
                generate_answer("system", "user")

    def test_generate_answer_raises_when_content_empty(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
        )
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: mock_response)
            )
        )

        with patch("app.utils.llm_client._create_llm_client", return_value=mock_client):
            with patch("app.utils.llm_client.settings.llm_api_key", "test-key"):
                with self.assertRaises(RuntimeError):
                    generate_answer("system", "user")
