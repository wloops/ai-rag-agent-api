import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

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
from app.services.rag_service import ask_knowledge_base, stream_knowledge_base_events


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationResponse)
def create_chat_conversation(
    data: ConversationCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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
    return ask_knowledge_base(
        db=db,
        current_user_id=current_user.id,
        knowledge_base_id=data.knowledge_base_id,
        question=data.question,
        top_k=data.top_k,
        debug=data.debug,
        conversation_id=data.conversation_id,
    )


@router.post("/ask/stream")
def ask_chat_stream(
    data: ChatAskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    def event_stream():
        try:
            for event_name, payload in stream_knowledge_base_events(
                db=db,
                current_user_id=current_user.id,
                knowledge_base_id=data.knowledge_base_id,
                question=data.question,
                top_k=data.top_k,
                debug=data.debug,
                conversation_id=data.conversation_id,
            ):
                yield _format_sse_event(event_name, payload)
        except HTTPException as exc:
            yield _format_sse_event("error", {"detail": exc.detail})
        except Exception as exc:
            yield _format_sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # 显式禁止中间层压缩/改写 SSE，降低线上代理把分片攒到最后才下发的概率。
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
