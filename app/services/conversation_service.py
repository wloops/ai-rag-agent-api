from datetime import datetime

from fastapi import HTTPException

from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.services.knowledge_base_service import get_active_owned_knowledge_base

DEFAULT_CONVERSATION_TITLE = "新会话"
MAX_AUTO_TITLE_LENGTH = 30
ALLOWED_MESSAGE_ROLES = {"user", "assistant"}


def get_owned_knowledge_base(db, knowledge_base_id: int, current_user_id: int) -> KnowledgeBase:
    return get_active_owned_knowledge_base(db, knowledge_base_id, current_user_id)


def get_owned_conversation(db, conversation_id: int, current_user_id: int) -> Conversation:
    conversation = (
        db.query(Conversation)
        .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user_id,
            KnowledgeBase.deleted_at.is_(None),
        )
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


def create_conversation(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    title: str | None = None,
) -> Conversation:
    get_owned_knowledge_base(db, knowledge_base_id, current_user_id)
    normalized_title = _normalize_conversation_title(title) or DEFAULT_CONVERSATION_TITLE

    conversation = Conversation(
        user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        title=normalized_title,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db, current_user_id: int) -> list[Conversation]:
    return (
        db.query(Conversation)
        .join(KnowledgeBase, Conversation.knowledge_base_id == KnowledgeBase.id)
        .filter(
            Conversation.user_id == current_user_id,
            KnowledgeBase.deleted_at.is_(None),
        )
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .all()
    )


def list_conversation_messages(db, conversation_id: int, current_user_id: int) -> list[Message]:
    conversation = get_owned_conversation(db, conversation_id, current_user_id)
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )


def resolve_conversation_for_question(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    conversation_id: int | None = None,
) -> Conversation:
    if conversation_id is not None:
        conversation = get_owned_conversation(db, conversation_id, current_user_id)
        if conversation.knowledge_base_id != knowledge_base_id:
            raise HTTPException(
                status_code=400,
                detail="Conversation does not belong to the specified knowledge base",
            )
        return conversation

    auto_title = build_conversation_title_from_question(question)
    return create_conversation(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        title=auto_title,
    )


def save_message(
    db,
    conversation: Conversation,
    role: str,
    content: str,
    citations_json: list[dict] | None = None,
) -> Message:
    if role not in ALLOWED_MESSAGE_ROLES:
        raise ValueError(f"Unsupported message role: {role}")

    conversation.updated_at = datetime.utcnow()
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        citations_json=citations_json,
    )
    db.add(message)
    db.commit()
    db.refresh(conversation)
    db.refresh(message)
    return message


def build_conversation_title_from_question(question: str) -> str:
    normalized_question = question.strip()
    if not normalized_question:
        return DEFAULT_CONVERSATION_TITLE

    return normalized_question[:MAX_AUTO_TITLE_LENGTH]


def _normalize_conversation_title(title: str | None) -> str | None:
    if title is None:
        return None

    normalized_title = title.strip()
    if not normalized_title:
        return None

    return normalized_title[:255]
