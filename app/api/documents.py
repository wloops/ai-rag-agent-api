from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.chunk import ChunkListResponse
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_ingestion import create_pending_document, ingest_document_file
from app.services.knowledge_base_service import get_active_owned_knowledge_base


router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_ROOT = Path("uploads")
ALLOWED_FILE_TYPES = {"txt", "md", "pdf"}


def _get_file_type(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Only txt, md and pdf are supported")

    return suffix


def _build_storage_path(user_id: int, knowledge_base_id: int, filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    unique_filename = f"{uuid4().hex}{suffix}"
    return UPLOAD_ROOT / str(user_id) / str(knowledge_base_id) / unique_filename


def _get_owned_document(db: Session, document_id: int, current_user_id: int) -> Document:
    document = (
        db.query(Document)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            Document.id == document_id,
            KnowledgeBase.user_id == current_user_id,
            KnowledgeBase.deleted_at.is_(None),
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_active_owned_knowledge_base(db, knowledge_base_id, current_user.id)
    file_type = _get_file_type(file.filename)
    storage_path = _build_storage_path(current_user.id, knowledge_base_id, file.filename)
    document = create_pending_document(
        db=db,
        knowledge_base_id=knowledge_base_id,
        filename=file.filename,
        file_type=file_type,
        storage_path=str(storage_path),
    )
    return await ingest_document_file(db, document, file, storage_path, file_type)


@router.get("", response_model=list[DocumentListResponse])
def list_documents(
    knowledge_base_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Document).join(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user.id,
        KnowledgeBase.deleted_at.is_(None),
    )

    if knowledge_base_id is not None:
        get_active_owned_knowledge_base(db, knowledge_base_id, current_user.id)
        query = query.filter(Document.knowledge_base_id == knowledge_base_id)

    return query.order_by(Document.created_at.desc()).all()


@router.get("/{id}", response_model=DocumentDetailResponse)
def get_document(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_owned_document(db, id, current_user.id)


@router.get("/{id}/chunks", response_model=list[ChunkListResponse])
def list_document_chunks(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = _get_owned_document(db, id, current_user.id)

    return (
        db.query(Chunk)
        .filter(Chunk.document_id == document.id)
        .order_by(Chunk.chunk_index.asc())
        .all()
    )
