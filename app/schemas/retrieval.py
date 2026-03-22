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
    content: str
    score: float
