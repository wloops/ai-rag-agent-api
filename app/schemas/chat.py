from typing import Literal

from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    knowledge_base_id: int
    question: str
    top_k: int = Field(default=3, ge=1)
    debug: bool = False
    conversation_id: int | None = None


class ChatCitationItem(BaseModel):
    chunk_id: int | None = None
    document_id: int
    filename: str
    chunk_index: int
    start_offset: int | None = None
    end_offset: int | None = None
    snippet: str | None = None


class RetrievedChunkDebugItem(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class DebugRetrievedChunkItem(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    snippet: str
    score: float
    start_offset: int | None = None
    end_offset: int | None = None
    whether_cited: bool = False


class DebugInfo(BaseModel):
    question: str
    knowledge_base_id: int
    top_k: int
    top1_score: float | None = None
    threshold: float
    decision: Literal["answer", "reject"]
    retrieval_ms: int
    llm_ms: int
    total_ms: int
    embedding_ms: int | None = None
    final_context_preview: str | None = None
    retrieved_chunks: list[DebugRetrievedChunkItem]


class ChatAskResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[ChatCitationItem]
    retrieved_chunks: list[RetrievedChunkDebugItem] | None = None
    debug: DebugInfo | None = None
