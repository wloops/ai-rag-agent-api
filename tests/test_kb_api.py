import unittest
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.kb import router as kb_router
from app.core.database import Base, get_db
from app.core.security import get_current_user
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User


class KnowledgeBaseApiTestCase(unittest.TestCase):
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
        self.other_user = User(
            email="other@example.com",
            password_hash="other-password",
            nickname="other",
        )
        self.db.add_all([self.user, self.other_user])
        self.db.commit()
        self.db.refresh(self.user)

        self.app = FastAPI()
        self.app.include_router(kb_router)
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

    def _create_kb(
        self,
        *,
        user_id: int | None = None,
        name: str = "产品资料",
        description: str | None = "研发文档",
        created_at: datetime | None = None,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            user_id=user_id or self.user.id,
            name=name,
            description=description,
            created_at=created_at or datetime.utcnow(),
        )
        self.db.add(kb)
        self.db.commit()
        self.db.refresh(kb)
        return kb

    def test_create_kb_returns_aggregated_fields(self):
        response = self.client.post(
            "/api/kb",
            json={"name": "产品资料", "description": "研发文档"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "产品资料")
        self.assertEqual(payload["document_count"], 0)
        self.assertEqual(payload["updated_at"], payload["created_at"])

    def test_list_kbs_returns_document_count_and_latest_document_timestamp(self):
        first_kb = self._create_kb(
            name="已上传文档",
            description="有文档",
            created_at=datetime(2026, 3, 20, 10, 0, 0),
        )
        self._create_kb(
            name="空知识库",
            description="暂无文档",
            created_at=datetime(2026, 3, 21, 9, 0, 0),
        )

        self.db.add_all(
            [
                Document(
                    knowledge_base_id=first_kb.id,
                    filename="a.txt",
                    file_type="txt",
                    storage_path="uploads/a.txt",
                    status="success",
                    created_at=datetime(2026, 3, 21, 8, 0, 0),
                ),
                Document(
                    knowledge_base_id=first_kb.id,
                    filename="b.txt",
                    file_type="txt",
                    storage_path="uploads/b.txt",
                    status="success",
                    created_at=datetime(2026, 3, 22, 18, 30, 0),
                ),
            ]
        )
        self.db.commit()

        response = self.client.get("/api/kb")

        self.assertEqual(response.status_code, 200)
        payload = {item["name"]: item for item in response.json()}
        self.assertEqual(payload["已上传文档"]["document_count"], 2)
        self.assertEqual(payload["已上传文档"]["updated_at"], "2026-03-22T18:30:00")
        self.assertEqual(payload["空知识库"]["document_count"], 0)
        self.assertEqual(payload["空知识库"]["updated_at"], "2026-03-21T09:00:00")

    def test_get_kb_returns_single_item(self):
        kb = self._create_kb(name="演示库", description="演示描述")

        response = self.client.get(f"/api/kb/{kb.id}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], kb.id)
        self.assertEqual(payload["name"], "演示库")

    def test_update_kb_updates_name_and_description(self):
        kb = self._create_kb(name="旧名称", description="旧描述")

        response = self.client.patch(
            f"/api/kb/{kb.id}",
            json={"name": "新名称", "description": "新描述"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "新名称")
        self.assertEqual(payload["description"], "新描述")

    def test_update_kb_rejects_duplicate_active_name(self):
        target = self._create_kb(name="目标库")
        self._create_kb(name="冲突库")

        response = self.client.patch(
            f"/api/kb/{target.id}",
            json={"name": "冲突库", "description": "说明"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Knowledge base name already exists")

    def test_soft_deleted_kb_is_hidden_from_list(self):
        active = self._create_kb(name="保留库")
        deleted = self._create_kb(name="已删除库")

        delete_response = self.client.delete(f"/api/kb/{deleted.id}")
        self.assertEqual(delete_response.status_code, 204)

        response = self.client.get("/api/kb")

        self.assertEqual(response.status_code, 200)
        names = [item["name"] for item in response.json()]
        self.assertIn(active.name, names)
        self.assertNotIn(deleted.name, names)

    def test_can_recreate_same_name_after_soft_delete(self):
        kb = self._create_kb(name="可复用名称")

        delete_response = self.client.delete(f"/api/kb/{kb.id}")
        self.assertEqual(delete_response.status_code, 204)

        create_response = self.client.post(
            "/api/kb",
            json={"name": "可复用名称", "description": "新的描述"},
        )

        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(create_response.json()["name"], "可复用名称")


if __name__ == "__main__":
    unittest.main()
