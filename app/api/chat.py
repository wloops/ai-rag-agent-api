from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.chat import ChatAskRequest, ChatAskResponse
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationResponse,
    MessageResponse,
)
from app.services.conversation_service import (
    create_conversation,
    list_conversation_messages,
    list_conversations,
)
from app.services.rag_service import ask_knowledge_base


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationResponse)
def create_chat_conversation(
    data: ConversationCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 路由层只负责参数接收与鉴权，会话标题和归属校验都由 service 层统一处理。
    return create_conversation(
        db=db,
        current_user_id=current_user.id,
        knowledge_base_id=data.knowledge_base_id,
        title=data.title,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
def list_chat_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_conversations(db, current_user.id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
def get_chat_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_conversation_messages(db, conversation_id, current_user.id)


@router.post("/ask", response_model=ChatAskResponse)
def ask_chat(
    data: ChatAskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 路由层只负责接参与鉴权，检索、Prompt 构造、会话落库和模型生成都放在 service 中统一处理。
    return ask_knowledge_base(
        db=db,
        current_user_id=current_user.id,
        knowledge_base_id=data.knowledge_base_id,
        question=data.question,
        top_k=data.top_k,
        debug=data.debug,
        conversation_id=data.conversation_id,
    )
