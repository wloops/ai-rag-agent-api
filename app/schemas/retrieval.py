from pydantic import BaseModel, Field, field_validator


class RetrievalSearchRequest(BaseModel):
    knowledge_base_id: int
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


class RetrievalSearchItem(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
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
