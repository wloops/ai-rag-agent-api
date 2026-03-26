import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.tasks.document_tasks import process_document_task


class DocumentTaskTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.testing_session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        Base.metadata.create_all(bind=self.engine)
        self.db = self.testing_session_local()

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

        self.document = Document(
            knowledge_base_id=kb.id,
            filename="demo.txt",
            file_type="txt",
            storage_path="uploads/demo.txt",
            status="pending",
        )
        self.db.add(self.document)
        self.db.commit()
        self.db.refresh(self.document)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_process_document_task_marks_document_success(self):
        with patch("app.tasks.document_tasks.SessionLocal", return_value=self.db):
            with patch("app.tasks.document_tasks.ingest_document_from_storage") as mocked_ingest:
                def _mark_success(db, document):
                    document.status = "success"
                    document.error_message = None
                    db.commit()
                    db.refresh(document)
                    return document

                mocked_ingest.side_effect = _mark_success
                result = process_document_task.run(self.document.id)

        verification_session = self.testing_session_local()
        refreshed_document = verification_session.get(Document, self.document.id)
        verification_session.close()

        self.assertEqual(result["status"], "success")
        self.assertEqual(refreshed_document.status, "success")
