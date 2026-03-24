import unittest
from datetime import datetime
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
            "conversation_id": 10,
            "answer": "答案 [S1]",
            "citations": [
                {
                    "document_id": 1,
                    "filename": "demo.txt",
                    "chunk_id": 101,
                    "chunk_index": 0,
                    "start_offset": 10,
                    "end_offset": 20,
                    "snippet": "片段",
                }
            ],
            "retrieved_chunks": None,
            "debug": {
                "question": "什么是 demo？",
                "knowledge_base_id": 1,
                "top_k": 3,
                "top1_score": 0.91,
                "threshold": 0.35,
                "decision": "answer",
                "retrieval_ms": 12,
                "llm_ms": 45,
                "total_ms": 61,
                "embedding_ms": 6,
                "final_context_preview": "S1\ncontent: demo",
                "retrieved_chunks": [
                    {
                        "chunk_id": 101,
                        "document_id": 1,
                        "filename": "demo.txt",
                        "chunk_index": 0,
                        "snippet": "demo snippet",
                        "score": 0.91,
                        "start_offset": 10,
                        "end_offset": 20,
                        "whether_cited": True,
                    }
                ],
            },
        }

        with patch("app.api.chat.ask_knowledge_base", return_value=mocked_response):
            response = self.client.post(
                "/api/chat/ask",
                json={
                    "knowledge_base_id": 1,
                    "question": "什么是 demo？",
                    "top_k": 3,
                    "debug": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "答案 [S1]")
        self.assertEqual(payload["conversation_id"], 10)
        self.assertEqual(payload["debug"]["decision"], "answer")
        self.assertTrue(payload["debug"]["retrieved_chunks"][0]["whether_cited"])

    def test_create_conversation_returns_response(self):
        mocked_response = {
            "id": 1,
            "knowledge_base_id": 2,
            "title": "测试会话",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        with patch("app.api.chat.create_conversation", return_value=mocked_response):
            response = self.client.post(
                "/api/chat/conversations",
                json={"knowledge_base_id": 2, "title": "测试会话"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "测试会话")

    def test_list_conversations_returns_response(self):
        mocked_response = [
            {
                "id": 1,
                "knowledge_base_id": 2,
                "title": "测试会话",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        ]

        with patch("app.api.chat.list_conversations", return_value=mocked_response):
            response = self.client.get("/api/chat/conversations")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)

    def test_get_messages_returns_response(self):
        mocked_response = [
            {
                "id": 1,
                "conversation_id": 10,
                "role": "user",
                "content": "你好",
                "citations_json": None,
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "id": 2,
                "conversation_id": 10,
                "role": "assistant",
                "content": "你好，我是助手",
                "citations_json": [
                    {
                        "document_id": 1,
                        "filename": "demo.txt",
                        "chunk_id": 101,
                        "chunk_index": 0,
                        "start_offset": 10,
                        "end_offset": 20,
                        "snippet": "片段",
                    }
                ],
                "created_at": datetime.utcnow().isoformat(),
            },
        ]

        with patch("app.api.chat.list_conversation_messages", return_value=mocked_response):
            response = self.client.get("/api/chat/conversations/10/messages")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 2)
