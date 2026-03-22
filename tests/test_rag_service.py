import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.schemas.retrieval import RetrievalSearchItem
from app.services.rag_service import NO_ANSWER_MESSAGE, ask_knowledge_base


class RagServiceTestCase(unittest.TestCase):
    def test_returns_fallback_when_no_retrieval_results(self):
        with patch("app.services.rag_service.search_chunks", return_value=[]):
            with patch("app.services.rag_service.generate_answer") as mocked_generate_answer:
                response = ask_knowledge_base(
                    db=None,
                    current_user_id=1,
                    knowledge_base_id=1,
                    question="问题",
                    top_k=3,
                    debug=False,
                )

        mocked_generate_answer.assert_not_called()
        self.assertEqual(response.answer, NO_ANSWER_MESSAGE)
        self.assertEqual(response.citations, [])
        self.assertIsNone(response.retrieved_chunks)

    def test_returns_fallback_when_top_score_too_low(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                content="低相关内容",
                score=0.2,
            )
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch("app.services.rag_service.generate_answer") as mocked_generate_answer:
                response = ask_knowledge_base(
                    db=None,
                    current_user_id=1,
                    knowledge_base_id=1,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        mocked_generate_answer.assert_not_called()
        self.assertEqual(response.answer, NO_ANSWER_MESSAGE)
        self.assertEqual(len(response.retrieved_chunks or []), 1)

    def test_returns_answer_and_citations_when_model_cites_sources(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                content="第二段内容",
                score=0.8,
            ),
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="结论如下 [S1] 进一步说明 [S2]",
            ):
                response = ask_knowledge_base(
                    db=None,
                    current_user_id=1,
                    knowledge_base_id=1,
                    question="问题",
                    top_k=3,
                    debug=False,
                )

        self.assertEqual(response.answer, "结论如下 [S1] 进一步说明 [S2]")
        self.assertEqual(len(response.citations), 2)
        self.assertEqual(response.citations[0].chunk_index, 0)
        self.assertIsNone(response.retrieved_chunks)

    def test_falls_back_to_retrieved_chunks_when_model_has_no_source_ids(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                content="第二段内容",
                score=0.8,
            ),
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="这是没有引用标记的答案",
            ):
                response = ask_knowledge_base(
                    db=None,
                    current_user_id=1,
                    knowledge_base_id=1,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        self.assertEqual(len(response.citations), 2)
        self.assertEqual(len(response.retrieved_chunks or []), 2)

    def test_blank_question_returns_400(self):
        with self.assertRaises(HTTPException) as context:
            ask_knowledge_base(
                db=None,
                current_user_id=1,
                knowledge_base_id=1,
                question="   ",
                top_k=3,
                debug=False,
            )

        self.assertEqual(context.exception.status_code, 400)
