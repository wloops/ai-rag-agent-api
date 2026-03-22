from datetime import datetime

from pydantic import BaseModel

from app.schemas.chat import ChatCitationItem


class ConversationCreateRequest(BaseModel):
    knowledge_base_id: int
    title: str | None = None


class ConversationResponse(BaseModel):
    id: int
    knowledge_base_id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    citations_json: list[ChatCitationItem] | None
    created_at: datetime

    model_config = {"from_attributes": True}
