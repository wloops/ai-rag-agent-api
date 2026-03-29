import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.retrieval import RetrievalSearchItem
from app.services.agent_service import (
    EMPTY_KNOWLEDGE_BASE_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    run_agent_task,
)


class AgentServiceTestCase(unittest.TestCase):
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
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        kb = KnowledgeBase(user_id=user.id, name="demo", description="demo kb")
        self.db.add(kb)
        self.db.commit()
        self.db.refresh(kb)

        document = Document(
            knowledge_base_id=kb.id,
            filename="interview.md",
            file_type="md",
            storage_path="uploads/interview.md",
            status="success",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            content="聚焦网络聚焦外贸 AI 营销和小渔夫 SaaS。",
            metadata_json={"start_char": 0, "end_char": 24, "chunk_size": 100, "overlap": 0},
            embedding=[1.0, 0.0, 0.0],
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)

        self.user_id = user.id
        self.knowledge_base_id = kb.id
        self.document_id = document.id
        self.chunk_id = chunk.id

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_run_agent_returns_empty_message_when_no_context_is_found(self):
        with patch("app.services.agent_service.search_chunks", return_value=[]):
            with patch("app.services.agent_service.generate_answer") as mocked_generate:
                response = run_agent_task(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    task_type="interview_material",
                    query="面试要点",
                    top_k=5,
                )

        mocked_generate.assert_not_called()
        self.assertEqual(response.answer, EMPTY_KNOWLEDGE_BASE_MESSAGE)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.workflow_trace[-1].status, "skipped")

    def test_run_agent_rejects_low_relevance_results(self):
        with patch(
            "app.services.agent_service.search_chunks",
            return_value=[
                RetrievalSearchItem(
                    chunk_id=self.chunk_id,
                    document_id=self.document_id,
                    filename="interview.md",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=24,
                    content="聚焦网络聚焦外贸 AI 营销和小渔夫 SaaS。",
                    score=0.2,
                )
            ],
        ):
            with patch("app.services.agent_service.generate_answer") as mocked_generate:
                response = run_agent_task(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    task_type="knowledge_base_summary",
                    query=None,
                    top_k=5,
                )

        mocked_generate.assert_not_called()
        self.assertEqual(response.answer, LOW_RELEVANCE_MESSAGE)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.workflow_trace[-1].status, "skipped")

    def test_run_latest_documents_digest_returns_summary(self):
        with patch(
            "app.services.agent_service.generate_answer",
            return_value="最近文档集中在公司背景、产品矩阵和岗位映射。",
        ) as mocked_generate:
            response = run_agent_task(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                task_type="latest_documents_digest",
                query=None,
                top_k=5,
            )

        mocked_generate.assert_called_once()
        self.assertEqual(response.task_type, "latest_documents_digest")
        self.assertIn("最近文档集中在公司背景", response.answer)
        self.assertEqual(response.workflow_trace[-1].status, "completed")
        self.assertEqual(response.citations[0].chunk_id, self.chunk_id)
