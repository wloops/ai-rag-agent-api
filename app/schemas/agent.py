from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatCitationItem


AgentTaskType = Literal[
    "knowledge_base_qa",
    "knowledge_base_summary",
    "latest_documents_digest",
    "interview_material",
]


class AgentRunRequest(BaseModel):
    knowledge_base_id: int
    task_type: AgentTaskType
    query: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class AgentWorkflowTraceItem(BaseModel):
    step: str
    status: Literal["completed", "skipped"]
    detail: str


class AgentRunResponse(BaseModel):
    knowledge_base_id: int
    task_type: AgentTaskType
    answer: str
    citations: list[ChatCitationItem]
    workflow_trace: list[AgentWorkflowTraceItem]
