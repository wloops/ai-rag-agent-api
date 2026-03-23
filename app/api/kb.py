from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.kb import KnowledgeBaseCreate, KnowledgeBaseResponse


# 这个 router 处理知识库的新增和查询。
router = APIRouter(prefix="/api/kb", tags=["knowledge_base"])


def _serialize_knowledge_base(
    kb: KnowledgeBase,
    document_count: int = 0,
    latest_document_created_at=None,
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        user_id=kb.user_id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at,
        document_count=document_count,
        updated_at=latest_document_created_at or kb.created_at,
    )


def _list_knowledge_base_rows(db: Session, current_user_id: int):
    return (
        db.query(
            KnowledgeBase,
            func.count(Document.id).label("document_count"),
            func.max(Document.created_at).label("latest_document_created_at"),
        )
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(KnowledgeBase.user_id == current_user_id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
        .all()
    )


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # current_user 来自鉴权依赖，所以这里只会为已登录用户创建知识库。
    existing_kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.user_id == current_user.id,
            KnowledgeBase.name == data.name,
        )
        .first()
    )
    if existing_kb:
        raise HTTPException(status_code=400, detail="Knowledge base name already exists")

    kb = KnowledgeBase(
        user_id=current_user.id,
        name=data.name,
        description=data.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _serialize_knowledge_base(kb=kb)


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 只查当前登录用户自己的知识库数据。
    rows = _list_knowledge_base_rows(db, current_user.id)
    return [
        _serialize_knowledge_base(
            kb=kb,
            document_count=document_count,
            latest_document_created_at=latest_document_created_at,
        )
        for kb, document_count, latest_document_created_at in rows
    ]
