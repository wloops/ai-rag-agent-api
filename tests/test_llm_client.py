import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.utils.llm_client import generate_answer, rewrite_question, stream_answer


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

    def test_stream_answer_yields_delta_content(self):
        mock_stream = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="杩欐槸"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="绛旀"))]
            ),
        ]
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: mock_stream)
            )
        )

        with patch("app.utils.llm_client._create_llm_client", return_value=mock_client):
            with patch("app.utils.llm_client.settings.llm_api_key", "test-key"):
                result = list(stream_answer("system", "user"))

        self.assertEqual(result, ["杩欐槸", "绛旀"])

    def test_stream_answer_raises_when_no_delta_content(self):
        mock_stream = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            )
        ]
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: mock_stream)
            )
        )

        with patch("app.utils.llm_client._create_llm_client", return_value=mock_client):
            with patch("app.utils.llm_client.settings.llm_api_key", "test-key"):
                with self.assertRaises(RuntimeError):
                    list(stream_answer("system", "user"))

    def test_rewrite_question_returns_standalone_question(self):
        history = [
            SimpleNamespace(role="user", content="公司请假制度是什么？"),
            SimpleNamespace(role="assistant", content="需要提前提交审批。"),
        ]

        with patch(
            "app.utils.llm_client.generate_answer",
            return_value="员工请假制度需要提前多久提交审批？",
        ) as mocked_generate:
            result = rewrite_question(history, "那要提前多久？")

        self.assertEqual(result, "员工请假制度需要提前多久提交审批？")
        mocked_generate.assert_called_once()

    def test_rewrite_question_falls_back_when_llm_fails(self):
        history = [SimpleNamespace(role="user", content="介绍一下报销流程")]

        with patch(
            "app.utils.llm_client.generate_answer",
            side_effect=RuntimeError("llm unavailable"),
        ):
            result = rewrite_question(history, "要多久到账？")

        self.assertEqual(result, "要多久到账？")
