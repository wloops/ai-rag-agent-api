import unittest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.services.conversation_service import (
    build_conversation_title_from_question,
    create_conversation,
    list_conversation_messages,
    list_conversations,
    resolve_conversation_for_question,
    save_message,
)


class ConversationServiceTestCase(unittest.TestCase):
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
        self.db.add_all([knowledge_base, other_kb])
        self.db.commit()
        self.db.refresh(knowledge_base)
        self.db.refresh(other_kb)

        self.user_id = user.id
        self.knowledge_base_id = knowledge_base.id

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_create_conversation_uses_default_title_when_missing(self):
        conversation = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
        )

        self.assertEqual(conversation.title, "新会话")

    def test_build_conversation_title_from_question_truncates_to_30_chars(self):
        title = build_conversation_title_from_question("a" * 40)
        self.assertEqual(len(title), 30)

    def test_resolve_conversation_for_question_creates_conversation_when_missing(self):
        conversation = resolve_conversation_for_question(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            question="这是第一条问题标题",
            conversation_id=None,
        )

        self.assertEqual(conversation.title, "这是第一条问题标题")

    def test_save_message_updates_conversation(self):
        conversation = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="测试会话",
        )

        message = save_message(
            self.db,
            conversation=conversation,
            role="assistant",
            content="回答内容",
            citations_json=[{"document_id": 1}],
        )

        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.citations_json, [{"document_id": 1}])

    def test_list_conversations_orders_by_updated_at_desc(self):
        first = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="第一会话",
        )
        second = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="第二会话",
        )
        first.updated_at = datetime.utcnow() + timedelta(seconds=1)
        self.db.commit()
        self.db.refresh(first)

        conversations = list_conversations(self.db, self.user_id)
        self.assertEqual(conversations[0].id, first.id)
        self.assertEqual(conversations[1].id, second.id)

    def test_list_conversation_messages_returns_owned_messages(self):
        conversation = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="测试会话",
        )
        save_message(self.db, conversation, "user", "hello")
        save_message(self.db, conversation, "assistant", "world")

        messages = list_conversation_messages(self.db, conversation.id, self.user_id)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    def test_deleted_knowledge_base_hides_conversation_list(self):
        create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="隐藏会话",
        )

        knowledge_base = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == self.knowledge_base_id).first()
        knowledge_base.deleted_at = datetime.utcnow()
        self.db.commit()

        conversations = list_conversations(self.db, self.user_id)
        self.assertEqual(conversations, [])

    def test_deleted_knowledge_base_blocks_message_access(self):
        conversation = create_conversation(
            self.db,
            current_user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="不可访问会话",
        )
        save_message(self.db, conversation, "user", "hello")

        knowledge_base = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == self.knowledge_base_id).first()
        knowledge_base.deleted_at = datetime.utcnow()
        self.db.commit()

        with self.assertRaises(Exception) as context:
            list_conversation_messages(self.db, conversation.id, self.user_id)

        self.assertEqual(context.exception.status_code, 404)
