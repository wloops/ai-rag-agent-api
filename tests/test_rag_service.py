import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.user import User
from app.schemas.retrieval import RetrievalSearchItem
from app.services.rag_service import NO_ANSWER_MESSAGE, ask_knowledge_base


class RagServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.db = TestingSessionLocal()

        user = User(
            email="tester@example.com",
            password_hash="hashed-password",
            nickname="tester",
        )
        other_user = User(
            email="other@example.com",
            password_hash="hashed-password",
            nickname="other",
        )
        self.db.add_all([user, other_user])
        self.db.commit()
        self.db.refresh(user)
        self.db.refresh(other_user)

        knowledge_base = KnowledgeBase(user_id=user.id, name="demo", description="demo kb")
        other_kb = KnowledgeBase(user_id=other_user.id, name="other", description="other kb")
        second_kb = KnowledgeBase(user_id=user.id, name="second", description="second kb")
        self.db.add_all([knowledge_base, other_kb, second_kb])
        self.db.commit()
        self.db.refresh(knowledge_base)
        self.db.refresh(other_kb)
        self.db.refresh(second_kb)

        self.user_id = user.id
        self.knowledge_base_id = knowledge_base.id
        self.other_knowledge_base_id = other_kb.id
        self.second_knowledge_base_id = second_kb.id

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_returns_reject_debug_when_no_retrieval_results(self):
        with patch("app.services.rag_service.search_chunks", return_value=[]):
            with patch("app.services.rag_service.generate_answer") as mocked_generate_answer:
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        mocked_generate_answer.assert_not_called()
        self.assertEqual(response.answer, NO_ANSWER_MESSAGE)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.debug.decision, "reject")
        self.assertIsNone(response.debug.top1_score)
        self.assertEqual(response.debug.threshold, 0.35)
        self.assertEqual(response.debug.llm_ms, 0)
        self.assertIsNone(response.debug.final_context_preview)
        self.assertEqual(response.debug.retrieved_chunks, [])
        self.assertEqual(response.retrieved_chunks, [])
        self.assertIsNotNone(response.conversation_id)

        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == response.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].content, NO_ANSWER_MESSAGE)

    def test_returns_answer_and_debug_info_when_model_cites_sources(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                start_offset=6,
                end_offset=12,
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
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        self.assertEqual(response.answer, "结论如下 [S1] 进一步说明 [S2]")
        self.assertEqual(len(response.citations), 2)
        self.assertEqual(response.citations[0].chunk_id, 1)
        self.assertEqual(response.citations[0].chunk_index, 0)
        self.assertEqual(response.citations[0].start_offset, 0)
        self.assertEqual(response.citations[0].end_offset, 6)
        self.assertEqual(len(response.retrieved_chunks or []), 2)
        self.assertEqual(response.debug.decision, "answer")
        self.assertEqual(response.debug.top1_score, 0.9)
        self.assertEqual(response.debug.threshold, 0.35)
        self.assertIsNotNone(response.debug.final_context_preview)
        self.assertEqual(len(response.debug.retrieved_chunks), 2)
        self.assertTrue(response.debug.retrieved_chunks[0].whether_cited)
        self.assertTrue(response.debug.retrieved_chunks[1].whether_cited)
        self.assertGreaterEqual(response.debug.llm_ms, 0)
        self.assertGreaterEqual(response.debug.retrieval_ms, 0)
        self.assertGreaterEqual(response.debug.total_ms, 0)

        conversation = self.db.query(Conversation).filter(Conversation.id == response.conversation_id).first()
        self.assertEqual(conversation.title, "问题")

        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == response.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(len(messages[1].citations_json or []), 2)
        self.assertEqual(messages[1].citations_json[0]["chunk_id"], 1)
        self.assertEqual(messages[1].citations_json[0]["start_offset"], 0)
        self.assertEqual(messages[1].citations_json[0]["end_offset"], 6)

    def test_fallback_citations_mark_only_top_three_chunks_as_cited(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                start_offset=6,
                end_offset=12,
                content="第二段内容",
                score=0.8,
            ),
            RetrievalSearchItem(
                chunk_id=3,
                document_id=1,
                filename="demo.txt",
                chunk_index=2,
                start_offset=12,
                end_offset=18,
                content="第三段内容",
                score=0.7,
            ),
            RetrievalSearchItem(
                chunk_id=4,
                document_id=1,
                filename="demo.txt",
                chunk_index=3,
                start_offset=18,
                end_offset=24,
                content="第四段内容",
                score=0.6,
            ),
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="这是没有引用标记的答案",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=4,
                    debug=True,
                )

        cited_flags = [item.whether_cited for item in response.debug.retrieved_chunks]
        self.assertEqual(cited_flags, [True, True, True, False])

    def test_debug_is_none_when_request_does_not_enable_it(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            )
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=False,
                )

        self.assertIsNone(response.debug)
        self.assertIsNone(response.retrieved_chunks)

    def test_blank_question_returns_400(self):
        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="   ",
                top_k=3,
                debug=False,
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_reuses_existing_conversation_and_saves_messages(self):
        conversation = Conversation(
            user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="已有会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            )
        ]

        with patch("app.services.rag_service.search_chunks", return_value=retrieved):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="继续问",
                    top_k=3,
                    debug=False,
                    conversation_id=conversation.id,
                )

        self.assertEqual(response.conversation_id, conversation.id)
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    def test_rejects_conversation_from_another_knowledge_base(self):
        conversation = Conversation(
            user_id=self.user_id,
            knowledge_base_id=self.second_knowledge_base_id,
            title="另一个知识库会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="问题",
                conversation_id=conversation.id,
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_rejects_unowned_conversation(self):
        conversation = Conversation(
            user_id=999,
            knowledge_base_id=self.knowledge_base_id,
            title="别人的会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="问题",
                conversation_id=conversation.id,
            )

        self.assertEqual(context.exception.status_code, 404)
