from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.utils.file_parser import parse_file


router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_ROOT = Path("uploads")
ALLOWED_FILE_TYPES = {"txt", "md", "pdf"}


def _get_owned_knowledge_base(
    db: Session, knowledge_base_id: int, current_user_id: int
) -> KnowledgeBase:
    # 上传和查询文档前都要先确认知识库归当前用户所有，避免越权访问。
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user_id,
        )
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return knowledge_base


def _get_file_type(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Only txt, md and pdf are supported")

    return suffix


def _build_storage_path(user_id: int, knowledge_base_id: int, filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    # 实际落盘文件名增加随机值，避免同名文件覆盖历史数据。
    unique_filename = f"{uuid4().hex}{suffix}"
    return UPLOAD_ROOT / str(user_id) / str(knowledge_base_id) / unique_filename


async def _save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as output_file:
            while chunk := await upload_file.read(1024 * 1024):
                output_file.write(chunk)
    except OSError as exc:
        raise RuntimeError(f"Failed to save uploaded file: {exc}") from exc
    finally:
        await upload_file.close()


def _mark_document_failed(db: Session, document: Document, exc: Exception) -> Document:
    # 解析或落盘失败时保留失败记录，便于后续排查问题。
    document.status = "failed"
    document.error_message = str(exc)
    db.commit()
    db.refresh(document)
    return document


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_knowledge_base(db, knowledge_base_id, current_user.id)
    file_type = _get_file_type(file.filename)
    storage_path = _build_storage_path(current_user.id, knowledge_base_id, file.filename)

    # 先创建 pending 记录，这样无论后续成功还是失败，数据库里都有可追踪状态。
    document = Document(
        knowledge_base_id=knowledge_base_id,
        filename=file.filename,
        file_type=file_type,
        storage_path=str(storage_path),
        status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        await _save_upload_file(file, storage_path)
        parse_file(storage_path, file_type)
    except Exception as exc:
        return _mark_document_failed(db, document, exc)

    document.status = "success"
    document.error_message = None
    db.commit()
    db.refresh(document)
    return document


@router.get("", response_model=list[DocumentListResponse])
def list_documents(
    knowledge_base_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Document).join(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user.id
    )

    # 提供知识库维度过滤，方便前端在单个知识库详情页直接拉取文档列表。
    if knowledge_base_id is not None:
        _get_owned_knowledge_base(db, knowledge_base_id, current_user.id)
        query = query.filter(Document.knowledge_base_id == knowledge_base_id)

    return query.order_by(Document.created_at.desc()).all()


@router.get("/{id}", response_model=DocumentDetailResponse)
def get_document(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 文档详情查询也要通过知识库归属做权限收敛，避免直接枚举文档 id 越权。
    document = (
        db.query(Document)
        .join(KnowledgeBase)
        .filter(Document.id == id, KnowledgeBase.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document
