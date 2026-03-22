from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.utils.embeddings import embed_texts
from app.utils.file_parser import parse_file
from app.utils.text_cleaner import clean_text
from app.utils.text_splitter import iter_chunk_ranges

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100


def create_pending_document(
    db: Session,
    knowledge_base_id: int,
    filename: str,
    file_type: str,
    storage_path: str,
) -> Document:
    # 先创建 pending 记录，这样无论后续成功还是失败，数据库里都有可追踪状态。
    document = Document(
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        file_type=file_type,
        storage_path=storage_path,
        status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


async def ingest_document_file(
    db: Session,
    document: Document,
    upload_file: UploadFile,
    storage_path: Path,
    file_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    try:
        await _save_upload_file(upload_file, storage_path)
        raw_text = parse_file(storage_path, file_type)
        cleaned_text = clean_text(raw_text)
        return persist_document_chunks(
            db,
            document,
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:
        db.rollback()
        return _mark_document_failed(db, document, exc)


def persist_document_chunks(
    db: Session,
    document: Document,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    try:
        chunk_ranges = iter_chunk_ranges(text, chunk_size=chunk_size, overlap=overlap)
        chunk_contents = [text[start:end] for start, end in chunk_ranges]

        if chunk_contents:
            # chunk 的 embedding 在入库前统一生成，避免查询时临时计算带来额外延迟和成本。
            embeddings = embed_texts(chunk_contents)
        else:
            embeddings = []

        chunks = _build_chunks(
            document.id,
            text,
            chunk_ranges,
            embeddings,
            chunk_size,
            overlap,
        )

        # chunk 入库和文档成功状态放在同一次提交里，避免出现 success 但 chunk 不完整。
        db.add_all(chunks)
        document.status = "success"
        document.error_message = None
        db.commit()
        db.refresh(document)
        return document
    except Exception as exc:
        db.rollback()
        return _mark_document_failed(db, document, exc)


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


def _build_chunks(
    document_id: int,
    text: str,
    chunk_ranges: list[tuple[int, int]],
    embeddings: list[list[float]],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for chunk_index, (start, end) in enumerate(chunk_ranges):
        content = text[start:end]
        embedding = embeddings[chunk_index] if chunk_index < len(embeddings) else None
        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                metadata_json={
                    "start_char": start,
                    "end_char": end,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
                embedding=embedding,
            )
        )

    return chunks


def _mark_document_failed(db: Session, document: Document, exc: Exception) -> Document:
    # 任一关键步骤失败都要把文档明确标记为 failed，方便前端和排障流程感知异常。
    document.status = "failed"
    document.error_message = str(exc)
    db.commit()
    db.refresh(document)
    return document
