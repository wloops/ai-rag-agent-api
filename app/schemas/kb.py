from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    # 新建知识库时，前端提交的数据会先经过这里校验。
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class KnowledgeBaseResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: str | None
    created_at: datetime
    document_count: int
    updated_at: datetime

    model_config = {"from_attributes": True}
