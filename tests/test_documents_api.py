import unittest
from datetime import datetime
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch

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
        self.temp_dir = tempfile.TemporaryDirectory()

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

        active_storage_path = Path(self.temp_dir.name) / "active.txt"
        active_storage_path.write_text("0123456789active chunk tail", encoding="utf-8")
        deleted_storage_path = Path(self.temp_dir.name) / "deleted.txt"
        deleted_storage_path.write_text("deleted chunk tail", encoding="utf-8")

        self.active_document = Document(
            knowledge_base_id=self.active_kb.id,
            filename="active.txt",
            file_type="txt",
            storage_path=str(active_storage_path),
            status="success",
        )
        self.deleted_document = Document(
            knowledge_base_id=self.deleted_kb.id,
            filename="deleted.txt",
            file_type="txt",
            storage_path=str(deleted_storage_path),
            status="success",
        )
        self.db.add_all([self.active_document, self.deleted_document])
        self.db.commit()
        self.db.refresh(self.active_document)
        self.db.refresh(self.deleted_document)

        self.active_chunk = Chunk(
            document_id=self.active_document.id,
            chunk_index=0,
            content="active chunk",
            metadata_json={"start_char": 10, "end_char": 22, "chunk_size": 100, "overlap": 0},
            embedding=[1.0, 0.0, 0.0],
        )
        self.deleted_chunk = Chunk(
            document_id=self.deleted_document.id,
            chunk_index=0,
            content="deleted chunk",
            metadata_json={"start_char": 0, "end_char": 13, "chunk_size": 100, "overlap": 0},
            embedding=[1.0, 0.0, 0.0],
        )
        self.db.add_all([self.active_chunk, self.deleted_chunk])
        self.db.commit()
        self.db.refresh(self.active_chunk)
        self.db.refresh(self.deleted_chunk)

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
        self.temp_dir.cleanup()

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

    def test_get_chunk_preview_returns_context_and_offsets(self):
        response = self.client.get(
            f"/api/documents/{self.active_document.id}/chunks/{self.active_chunk.id}/preview"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_id"], self.active_document.id)
        self.assertEqual(payload["chunk_id"], self.active_chunk.id)
        self.assertEqual(payload["filename"], "active.txt")
        self.assertEqual(payload["chunk_index"], 0)
        self.assertEqual(payload["start_offset"], 10)
        self.assertEqual(payload["end_offset"], 22)
        self.assertEqual(payload["preview_text"], "0123456789active chunk tail")
        self.assertEqual(payload["highlight_start_offset"], 10)
        self.assertEqual(payload["highlight_end_offset"], 22)

    def test_get_chunk_preview_for_deleted_knowledge_base_returns_404(self):
        response = self.client.get(
            f"/api/documents/{self.deleted_document.id}/chunks/{self.deleted_chunk.id}/preview"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Document not found")

    def test_get_chunk_preview_with_mismatched_chunk_returns_404(self):
        response = self.client.get(
            f"/api/documents/{self.active_document.id}/chunks/{self.deleted_chunk.id}/preview"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Chunk not found")

    def test_upload_document_returns_processing_and_enqueues_task(self):
        with patch(
            "app.api.documents.save_upload_file",
            new=AsyncMock(return_value=None),
        ) as mocked_save_upload_file:
            with patch("app.api.documents.enqueue_document_ingestion") as mocked_enqueue:
                response = self.client.post(
                    "/api/documents/upload",
                    data={"knowledge_base_id": str(self.active_kb.id)},
                    files={"file": ("queued.txt", b"hello world", "text/plain")},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["filename"], "queued.txt")
        self.assertEqual(payload["status"], "processing")
        mocked_save_upload_file.assert_awaited_once()
        mocked_enqueue.assert_called_once()

        created_document = (
            self.db.query(Document)
            .filter(Document.filename == "queued.txt")
            .order_by(Document.id.desc())
            .first()
        )
        self.assertIsNotNone(created_document)
        self.assertEqual(created_document.status, "processing")
