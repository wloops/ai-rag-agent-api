from __future__ import annotations

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.document import Document
from app.services.document_ingestion import (
    RetryableDocumentProcessingError,
    ingest_document_from_storage,
    mark_document_failed,
    mark_document_processing,
)


def _retry_countdown(retry_count: int) -> int:
    return settings.celery_document_retry_base_seconds * (2 ** retry_count)


@celery_app.task(
    bind=True,
    name="documents.process_document",
    acks_late=True,
    max_retries=settings.celery_document_max_retries,
)
def process_document_task(self, document_id: int) -> dict[str, int | str]:
    db = SessionLocal()
    document = None

    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            return {"document_id": document_id, "status": "missing"}

        mark_document_processing(db, document)
        ingest_document_from_storage(db, document)
        return {"document_id": document_id, "status": document.status}
    except RetryableDocumentProcessingError as exc:
        if document is None:
            raise

        if self.request.retries >= self.max_retries:
            mark_document_failed(db, document, exc)
            raise

        raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
    except Exception as exc:
        if document is not None:
            mark_document_failed(db, document, exc)
        raise
    finally:
        db.close()


def enqueue_document_ingestion(document_id: int) -> None:
    process_document_task.delay(document_id=document_id)
