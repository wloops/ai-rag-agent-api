from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    knowledge_base_id: int
    question: str
    top_k: int = Field(default=3, ge=1)
    debug: bool = False


class ChatCitationItem(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    snippet: str | None = None


class RetrievedChunkDebugItem(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class ChatAskResponse(BaseModel):
    answer: str
    citations: list[ChatCitationItem]
    retrieved_chunks: list[RetrievedChunkDebugItem] | None = None
