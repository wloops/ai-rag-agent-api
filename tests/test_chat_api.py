import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router as chat_router
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User


class ChatApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat_router)
        self.app.dependency_overrides[get_db] = lambda: None
        self.app.dependency_overrides[get_current_user] = lambda: User(
            id=1,
            email="tester@example.com",
            password_hash="hashed",
            nickname="tester",
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_chat_ask_returns_response(self):
        mocked_response = {
            "answer": "答案 [S1]",
            "citations": [
                {
                    "document_id": 1,
                    "filename": "demo.txt",
                    "chunk_index": 0,
                    "snippet": "片段",
                }
            ],
            "retrieved_chunks": None,
        }

        with patch("app.api.chat.ask_knowledge_base", return_value=mocked_response):
            response = self.client.post(
                "/api/chat/ask",
                json={
                    "knowledge_base_id": 1,
                    "question": "什么是 demo？",
                    "top_k": 3,
                    "debug": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "答案 [S1]")

    def test_chat_ask_requires_non_blank_question(self):
        response = self.client.post(
            "/api/chat/ask",
            json={
                "knowledge_base_id": 1,
                "question": "   ",
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 400)
