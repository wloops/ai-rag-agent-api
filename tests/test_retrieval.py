import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.services.retrieval import search_chunks


class RetrievalServiceTestCase(unittest.TestCase):
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
        other_kb = KnowledgeBase(
            user_id=other_user.id,
            name="other",
            description="other kb",
        )
        self.db.add_all([knowledge_base, other_kb])
        self.db.commit()
        self.db.refresh(knowledge_base)
        self.db.refresh(other_kb)

        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename="demo.txt",
            file_type="txt",
            storage_path="uploads/1/1/demo.txt",
            status="success",
        )
        other_document = Document(
            knowledge_base_id=other_kb.id,
            filename="other.txt",
            file_type="txt",
            storage_path="uploads/2/2/other.txt",
            status="success",
        )
        self.db.add_all([document, other_document])
        self.db.commit()
        self.db.refresh(document)
        self.db.refresh(other_document)

        self.db.add_all(
            [
                Chunk(
                    document_id=document.id,
                    chunk_index=0,
                    content="alpha chunk",
                    metadata_json={"start_char": 0, "end_char": 5, "chunk_size": 5, "overlap": 0},
                    embedding=[1.0, 0.0, 0.0],
                ),
                Chunk(
                    document_id=document.id,
                    chunk_index=1,
                    content="beta chunk",
                    metadata_json={"start_char": 5, "end_char": 10, "chunk_size": 5, "overlap": 0},
                    embedding=[0.0, 1.0, 0.0],
                ),
                Chunk(
                    document_id=other_document.id,
                    chunk_index=0,
                    content="other chunk",
                    metadata_json={"start_char": 0, "end_char": 5, "chunk_size": 5, "overlap": 0},
                    embedding=[1.0, 0.0, 0.0],
                ),
                Chunk(
                    document_id=document.id,
                    chunk_index=2,
                    content="no embedding chunk",
                    metadata_json={"start_char": 10, "end_char": 15, "chunk_size": 5, "overlap": 0},
                    embedding=None,
                ),
            ]
        )
        self.db.commit()

        self.user_id = user.id
        self.knowledge_base_id = knowledge_base.id
        self.other_knowledge_base_id = other_kb.id

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_search_chunks_returns_ranked_results_for_owned_knowledge_base(self):
        with patch("app.services.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            results = search_chunks(
                self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                query="alpha",
                top_k=3,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, 1)
        self.assertEqual(results[0].content, "alpha chunk")
        self.assertEqual(results[0].filename, "demo.txt")
        self.assertEqual(results[0].start_offset, 0)
        self.assertEqual(results[0].end_offset, 5)
        self.assertGreater(results[0].score, results[1].score)

    def test_search_chunks_only_returns_requested_top_k(self):
        with patch("app.services.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            results = search_chunks(
                self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                query="alpha",
                top_k=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, 1)
        self.assertEqual(results[0].chunk_index, 0)
        self.assertEqual(results[0].start_offset, 0)
        self.assertEqual(results[0].end_offset, 5)

    def test_search_chunks_rejects_unowned_knowledge_base(self):
        with patch("app.services.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            with self.assertRaises(Exception) as context:
                search_chunks(
                    self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.other_knowledge_base_id,
                    query="alpha",
                    top_k=3,
                )

        self.assertIn("Knowledge base not found", str(context.exception))
