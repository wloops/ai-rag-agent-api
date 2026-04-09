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
    guard_score: float | None = None
    source_channels: list[str] = Field(default_factory=list)
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fusion_rank: int | None = None
    rerank_rank: int | None = None


class DebugRetrievedChunkItem(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    snippet: str
    score: float
    guard_score: float | None = None
    source_channels: list[str] = Field(default_factory=list)
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fusion_rank: int | None = None
    rerank_rank: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    whether_cited: bool = False


class DebugGraphTraceItem(BaseModel):
    node: str
    status: Literal["completed", "skipped"]
    duration_ms: int
    detail: str
    used_history: bool | None = None
    rewritten_question: str | None = None
    retrieval_count: int | None = None
    dense_candidates_count: int | None = None
    bm25_candidates_count: int | None = None
    fusion_candidates_count: int | None = None
    rerank_applied: bool | None = None
    top1_score: float | None = None
    threshold: float | None = None
    decision: Literal["answer", "reject"] | None = None
    reject_reason: Literal["no_candidate", "low_confidence"] | None = None
    cited_count: int | None = None
    used_fallback_citations: bool | None = None


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
    rerank_enabled: bool | None = None
    reject_reason: Literal["no_candidate", "low_confidence"] | None = None
    final_context_preview: str | None = None
    retrieved_chunks: list[DebugRetrievedChunkItem]
    graph_trace: list[DebugGraphTraceItem] = Field(default_factory=list)


class ChatAskResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[ChatCitationItem]
    retrieved_chunks: list[RetrievedChunkDebugItem] | None = None
    debug: DebugInfo | None = None
