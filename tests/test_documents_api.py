import unittest
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.documents import router as documents_router
from app.core.database import Base, get_db
from app.core.security import get_current_user
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User


class DocumentsApiTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.testing_session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )
        Base.metadata.create_all(bind=self.engine)
        self.db = self.testing_session_local()

        self.user = User(
            email="tester@example.com",
            password_hash="hashed-password",
            nickname="tester",
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

        self.active_kb = KnowledgeBase(user_id=self.user.id, name="active", description="active kb")
        self.deleted_kb = KnowledgeBase(
            user_id=self.user.id,
            name="deleted",
            description="deleted kb",
            deleted_at=datetime.utcnow(),
        )
        self.db.add_all([self.active_kb, self.deleted_kb])
        self.db.commit()
        self.db.refresh(self.active_kb)
        self.db.refresh(self.deleted_kb)

        self.active_document = Document(
            knowledge_base_id=self.active_kb.id,
            filename="active.txt",
            file_type="txt",
            storage_path="uploads/active.txt",
            status="success",
        )
        self.deleted_document = Document(
            knowledge_base_id=self.deleted_kb.id,
            filename="deleted.txt",
            file_type="txt",
            storage_path="uploads/deleted.txt",
            status="success",
        )
        self.db.add_all([self.active_document, self.deleted_document])
        self.db.commit()
        self.db.refresh(self.active_document)
        self.db.refresh(self.deleted_document)

        self.db.add_all(
            [
                Chunk(
                    document_id=self.active_document.id,
                    chunk_index=0,
                    content="active chunk",
                    metadata_json={"start_char": 0, "end_char": 12, "chunk_size": 100, "overlap": 0},
                    embedding=[1.0, 0.0, 0.0],
                ),
                Chunk(
                    document_id=self.deleted_document.id,
                    chunk_index=0,
                    content="deleted chunk",
                    metadata_json={"start_char": 0, "end_char": 13, "chunk_size": 100, "overlap": 0},
                    embedding=[1.0, 0.0, 0.0],
                ),
            ]
        )
        self.db.commit()

        self.app = FastAPI()
        self.app.include_router(documents_router)
        self.app.dependency_overrides[get_db] = lambda: self.db
        self.app.dependency_overrides[get_current_user] = lambda: User(
            id=self.user.id,
            email=self.user.email,
            password_hash=self.user.password_hash,
            nickname=self.user.nickname,
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_list_documents_excludes_deleted_knowledge_base_documents(self):
        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["id"], self.active_document.id)

    def test_list_documents_for_deleted_knowledge_base_returns_404(self):
        response = self.client.get(f"/api/documents?knowledge_base_id={self.deleted_kb.id}")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Knowledge base not found")

    def test_get_document_for_deleted_knowledge_base_returns_404(self):
        response = self.client.get(f"/api/documents/{self.deleted_document.id}")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Document not found")

    def test_list_chunks_for_deleted_knowledge_base_returns_404(self):
        response = self.client.get(f"/api/documents/{self.deleted_document.id}/chunks")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Document not found")
