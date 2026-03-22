from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.chat import ChatAskRequest, ChatAskResponse
from app.services.rag_service import ask_knowledge_base


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatAskResponse)
def ask_chat(
    data: ChatAskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 路由层只负责接参与鉴权，检索、Prompt 构造和模型生成都放到 service 中统一处理。
    return ask_knowledge_base(
        db=db,
        current_user_id=current_user.id,
        knowledge_base_id=data.knowledge_base_id,
        question=data.question,
        top_k=data.top_k,
        debug=data.debug,
    )
