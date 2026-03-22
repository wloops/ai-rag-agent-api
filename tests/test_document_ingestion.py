import asyncio
import io
import tempfile
import unittest
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.services.document_ingestion import ingest_document_file, persist_document_chunks


class DocumentIngestionTestCase(unittest.TestCase):
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
            filename="demo.txt",
            file_type="txt",
            storage_path="uploads/1/1/demo.txt",
            status="pending",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        self.document = document

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_persist_document_chunks_success(self):
        persist_document_chunks(self.db, self.document, "abcdefghij", chunk_size=4, overlap=1)

        chunks = (
            self.db.query(Chunk)
            .filter(Chunk.document_id == self.document.id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        self.db.refresh(self.document)

        self.assertEqual(self.document.status, "success")
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1, 2])
        self.assertEqual(chunks[0].content, "abcd")
        self.assertEqual(
            chunks[0].metadata_json,
            {"start_char": 0, "end_char": 4, "chunk_size": 4, "overlap": 1},
        )

    def test_persist_document_chunks_marks_document_failed_when_split_fails(self):
        persist_document_chunks(self.db, self.document, "abcdefghij", chunk_size=4, overlap=4)

        chunks = self.db.query(Chunk).filter(Chunk.document_id == self.document.id).all()
        self.db.refresh(self.document)

        self.assertEqual(self.document.status, "failed")
        self.assertIn("overlap", self.document.error_message)
        self.assertEqual(chunks, [])

    def test_ingest_document_file_cleans_dirty_text_before_chunking(self):
        raw_text = "line1\u0001\tvalue\r\n\r\n\r\nline2\u200b  end"

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file = UploadFile(
                filename="dirty.txt",
                file=io.BytesIO(raw_text.encode("utf-8")),
            )

            asyncio.run(
                ingest_document_file(
                    self.db,
                    self.document,
                    upload_file,
                    storage_path=Path(temp_dir) / "dirty.txt",
                    file_type="txt",
                    chunk_size=100,
                    overlap=10,
                )
            )

        chunks = (
            self.db.query(Chunk)
            .filter(Chunk.document_id == self.document.id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        self.db.refresh(self.document)

        self.assertEqual(self.document.status, "success")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "line1 value\n\nline2 end")
        self.assertNotIn("\u0001", chunks[0].content)
        self.assertNotIn("\t", chunks[0].content)
        self.assertNotIn("\u200b", chunks[0].content)

    def test_ingest_document_file_marks_document_success_when_cleaned_text_is_empty(self):
        raw_text = "\u0001\u0002\t \r\n\u200b"

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file = UploadFile(
                filename="empty.txt",
                file=io.BytesIO(raw_text.encode("utf-8")),
            )

            asyncio.run(
                ingest_document_file(
                    self.db,
                    self.document,
                    upload_file,
                    storage_path=Path(temp_dir) / "empty.txt",
                    file_type="txt",
                    chunk_size=100,
                    overlap=10,
                )
            )

        chunks = self.db.query(Chunk).filter(Chunk.document_id == self.document.id).all()
        self.db.refresh(self.document)

        self.assertEqual(self.document.status, "success")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
