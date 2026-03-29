import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.agent import router as agent_router
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User


class AgentApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(agent_router)
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

    def test_run_agent_returns_structured_response(self):
        mocked_response = {
            "knowledge_base_id": 1,
            "task_type": "interview_material",
            "answer": "这家公司看重 Agent 和 RAG，因为它要把 AI 营销工具产品化。",
            "citations": [
                {
                    "chunk_id": 101,
                    "document_id": 10,
                    "filename": "聚焦网络-面试包.md",
                    "chunk_index": 0,
                    "start_offset": 0,
                    "end_offset": 120,
                    "snippet": "公司聚焦外贸 AI 营销 SaaS。",
                }
            ],
            "workflow_trace": [
                {
                    "step": "validate_knowledge_base",
                    "status": "completed",
                    "detail": "已确认知识库归属。",
                },
                {
                    "step": "generate_answer",
                    "status": "completed",
                    "detail": "已生成最终答案。",
                },
            ],
        }

        with patch("app.api.agent.run_agent_task", return_value=mocked_response):
            response = self.client.post(
                "/api/agent/run",
                json={
                    "knowledge_base_id": 1,
                    "task_type": "interview_material",
                    "query": "这家公司为什么重视 Agent？",
                    "top_k": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_type"], "interview_material")
        self.assertEqual(payload["workflow_trace"][0]["step"], "validate_knowledge_base")
        self.assertEqual(payload["citations"][0]["filename"], "聚焦网络-面试包.md")

    def test_run_agent_returns_400_for_invalid_payload(self):
        with patch(
            "app.api.agent.run_agent_task",
            side_effect=ValueError("Question is required for knowledge_base_qa"),
        ):
            response = self.client.post(
                "/api/agent/run",
                json={
                    "knowledge_base_id": 1,
                    "task_type": "knowledge_base_qa",
                    "query": "   ",
                    "top_k": 5,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Question is required for knowledge_base_qa")
