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

    # 允许把 ORM 对象直接返回给 FastAPI，再由 Pydantic 转成 JSON。
    model_config = {"from_attributes": True}
