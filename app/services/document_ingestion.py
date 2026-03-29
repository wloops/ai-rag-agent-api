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


class RetryableDocumentProcessingError(RuntimeError):
    pass


class NonRetryableDocumentProcessingError(RuntimeError):
    pass


def create_pending_document(
    db: Session,
    knowledge_base_id: int,
    filename: str,
    file_type: str,
    storage_path: str,
) -> Document:
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


def mark_document_processing(db: Session, document: Document) -> Document:
    document.status = "processing"
    document.error_message = None
    db.commit()
    db.refresh(document)
    return document


def mark_document_failed(db: Session, document: Document, exc: Exception) -> Document:
    # 文档入库失败往往发生在 flush/commit 阶段；先回滚旧事务，避免状态写回再次触发 PendingRollbackError。
    db.rollback()
    document.status = "failed"
    document.error_message = str(exc)
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
        await save_upload_file(upload_file, storage_path)
        document.file_type = file_type
        document.storage_path = str(storage_path)
        mark_document_processing(db, document)
        ingest_document_from_storage(
            db,
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return document
    except Exception as exc:
        db.rollback()
        return mark_document_failed(db, document, exc)


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as output_file:
            while chunk := await upload_file.read(1024 * 1024):
                output_file.write(chunk)
    except OSError as exc:
        raise NonRetryableDocumentProcessingError(
            f"Failed to save uploaded file: {exc}"
        ) from exc
    finally:
        await upload_file.close()


def ingest_document_from_storage(
    db: Session,
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    try:
        raw_text = parse_file(Path(document.storage_path), document.file_type)
    except Exception as exc:
        raise NonRetryableDocumentProcessingError(str(exc)) from exc

    cleaned_text = clean_text(raw_text)
    return persist_document_chunks_or_raise(
        db,
        document,
        cleaned_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def persist_document_chunks(
    db: Session,
    document: Document,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    try:
        return persist_document_chunks_or_raise(
            db,
            document,
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:
        db.rollback()
        return mark_document_failed(db, document, exc)


def persist_document_chunks_or_raise(
    db: Session,
    document: Document,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    chunk_ranges = _build_chunk_ranges(text, chunk_size=chunk_size, overlap=overlap)
    chunk_contents = [text[start:end] for start, end in chunk_ranges]
    embeddings = _build_embeddings(chunk_contents)

    db.query(Chunk).filter(Chunk.document_id == document.id).delete()
    db.add_all(
        _build_chunks(
            document.id,
            text,
            chunk_ranges,
            embeddings,
            chunk_size,
            overlap,
        )
    )
    document.status = "success"
    document.error_message = None
    db.commit()
    db.refresh(document)
    return document


def _build_chunk_ranges(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    try:
        return iter_chunk_ranges(text, chunk_size=chunk_size, overlap=overlap)
    except Exception as exc:
        raise NonRetryableDocumentProcessingError(str(exc)) from exc


def _build_embeddings(chunk_contents: list[str]) -> list[list[float]]:
    if not chunk_contents:
        return []

    try:
        return embed_texts(chunk_contents)
    except Exception as exc:
        raise RetryableDocumentProcessingError(str(exc)) from exc


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
