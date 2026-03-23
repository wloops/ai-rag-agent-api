from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.kb import KnowledgeBaseCreate, KnowledgeBaseResponse, KnowledgeBaseUpdate
from app.services.knowledge_base_service import (
    find_active_knowledge_base_name_conflict,
    get_active_owned_knowledge_base,
)


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


def _normalize_name(name: str) -> str:
    normalized_name = name.strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Knowledge base name is required")

    return normalized_name


def _normalize_description(description: str | None) -> str | None:
    if description is None:
        return None

    normalized_description = description.strip()
    return normalized_description or None


def _knowledge_base_rows_query(db: Session, current_user_id: int):
    return (
        db.query(
            KnowledgeBase,
            func.count(Document.id).label("document_count"),
            func.max(Document.created_at).label("latest_document_created_at"),
        )
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            KnowledgeBase.user_id == current_user_id,
            KnowledgeBase.deleted_at.is_(None),
        )
        .group_by(KnowledgeBase.id)
    )


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    normalized_name = _normalize_name(data.name)
    if find_active_knowledge_base_name_conflict(db, current_user.id, normalized_name):
        raise HTTPException(status_code=400, detail="Knowledge base name already exists")

    kb = KnowledgeBase(
        user_id=current_user.id,
        name=normalized_name,
        description=_normalize_description(data.description),
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
    rows = (
        _knowledge_base_rows_query(db, current_user.id)
        .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
        .all()
    )
    return [
        _serialize_knowledge_base(
            kb=kb,
            document_count=document_count,
            latest_document_created_at=latest_document_created_at,
        )
        for kb, document_count, latest_document_created_at in rows
    ]


@router.get("/{id}", response_model=KnowledgeBaseResponse)
def get_kb(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        _knowledge_base_rows_query(db, current_user.id)
        .filter(KnowledgeBase.id == id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb, document_count, latest_document_created_at = row
    return _serialize_knowledge_base(
        kb=kb,
        document_count=document_count,
        latest_document_created_at=latest_document_created_at,
    )


@router.patch("/{id}", response_model=KnowledgeBaseResponse)
def update_kb(
    id: int,
    data: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    kb = get_active_owned_knowledge_base(db, id, current_user.id)
    normalized_name = _normalize_name(data.name)

    if find_active_knowledge_base_name_conflict(
        db,
        current_user.id,
        normalized_name,
        exclude_id=kb.id,
    ):
        raise HTTPException(status_code=400, detail="Knowledge base name already exists")

    kb.name = normalized_name
    kb.description = _normalize_description(data.description)
    db.commit()
    db.refresh(kb)

    latest_document_created_at = (
        db.query(func.max(Document.created_at))
        .filter(Document.knowledge_base_id == kb.id)
        .scalar()
    )
    document_count = (
        db.query(func.count(Document.id)).filter(Document.knowledge_base_id == kb.id).scalar() or 0
    )
    return _serialize_knowledge_base(
        kb=kb,
        document_count=document_count,
        latest_document_created_at=latest_document_created_at,
    )


@router.delete("/{id}", status_code=204)
def delete_kb(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    kb = get_active_owned_knowledge_base(db, id, current_user.id)
    kb.deleted_at = datetime.utcnow()
    db.commit()
    return Response(status_code=204)
