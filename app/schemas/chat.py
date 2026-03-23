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


class ChatAskResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[ChatCitationItem]
    retrieved_chunks: list[RetrievedChunkDebugItem] | None = None
