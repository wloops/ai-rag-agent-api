from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.chunk_term import ChunkTerm
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.services.bm25_index import rebuild_knowledge_base_bm25_stats
from app.schemas.chunk import ChunkPreviewResponse
from app.utils.file_parser import parse_file
from app.utils.text_cleaner import clean_text

PREVIEW_CONTEXT_WINDOW = 200


def get_owned_document(db: Session, document_id: int, current_user_id: int) -> Document:
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


def get_owned_document_chunk(
    db: Session,
    document_id: int,
    chunk_id: int,
    current_user_id: int,
) -> tuple[Document, Chunk]:
    document = get_owned_document(db, document_id, current_user_id)
    chunk = (
        db.query(Chunk)
        .filter(
            Chunk.id == chunk_id,
            Chunk.document_id == document.id,
        )
        .first()
    )
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return document, chunk


def build_chunk_preview(
    db: Session,
    document_id: int,
    chunk_id: int,
    current_user_id: int,
) -> ChunkPreviewResponse:
    document, chunk = get_owned_document_chunk(db, document_id, chunk_id, current_user_id)
    raw_text = parse_file(Path(document.storage_path), document.file_type)
    cleaned_text = clean_text(raw_text)

    start_offset, end_offset = get_chunk_offsets(chunk)
    text_length = len(cleaned_text)
    safe_start = max(0, min(start_offset, text_length))
    safe_end = max(safe_start, min(end_offset, text_length))

    preview_start = max(0, safe_start - PREVIEW_CONTEXT_WINDOW)
    preview_end = min(text_length, safe_end + PREVIEW_CONTEXT_WINDOW)

    return ChunkPreviewResponse(
        document_id=document.id,
        chunk_id=chunk.id,
        filename=document.filename,
        chunk_index=chunk.chunk_index,
        start_offset=safe_start,
        end_offset=safe_end,
        preview_text=cleaned_text[preview_start:preview_end],
        highlight_start_offset=safe_start - preview_start,
        highlight_end_offset=safe_end - preview_start,
    )


def delete_owned_document(db: Session, document_id: int, current_user_id: int) -> None:
    document = get_owned_document(db, document_id, current_user_id)
    if document.status in {"pending", "processing"}:
        # 进行中的文档仍可能被 worker 消费，先阻止删除以避免并发竞态。
        raise HTTPException(status_code=400, detail="Processing documents cannot be deleted")

    chunk_ids = [
        chunk_id
        for chunk_id, in db.query(Chunk.id).filter(Chunk.document_id == document.id).all()
    ]
    if chunk_ids:
        db.query(ChunkTerm).filter(ChunkTerm.chunk_id.in_(chunk_ids)).delete(
            synchronize_session=False
        )

    db.query(Chunk).filter(Chunk.document_id == document.id).delete(
        synchronize_session=False
    )
    db.delete(document)
    rebuild_knowledge_base_bm25_stats(db, document.knowledge_base_id)
    db.commit()

    # 存储文件丢失不应回滚主流程，数据库删除成功即可认为操作完成。
    try:
        Path(document.storage_path).unlink(missing_ok=True)
    except OSError:
        pass


def get_chunk_offsets(chunk: Chunk) -> tuple[int, int]:
    metadata = chunk.metadata_json or {}
    start_offset = int(metadata.get("start_char", 0))
    end_offset = int(metadata.get("end_char", start_offset + len(chunk.content)))
    if end_offset < start_offset:
        end_offset = start_offset

    return start_offset, end_offset
