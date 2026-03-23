import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.kb import router as kb_router
from app.api.retrieval import router as retrieval_router
from app.core.database import Base, engine
from app.models.chunk import Chunk  # noqa: F401
from app.models.conversation import Conversation  # noqa: F401
from app.models.document import Document  # noqa: F401
from app.models.knowledge_base import KnowledgeBase  # noqa: F401
from app.models.message import Message  # noqa: F401
from app.models.user import User  # noqa: F401


app = FastAPI(title="AI Knowledge Base API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_postgresql_schema_compatibility() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                ALTER TABLE knowledge_bases
                ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL
                """
            )
        )
        connection.execute(
            text(
                """
                ALTER TABLE knowledge_bases
                DROP CONSTRAINT IF EXISTS uq_knowledge_bases_user_id_name
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ix_knowledge_bases_user_id_name_active
                ON knowledge_bases (user_id, name)
                WHERE deleted_at IS NULL
                """
            )
        )


@app.on_event("startup")
def on_startup():
    try:
        if engine.dialect.name == "postgresql":
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
        if engine.dialect.name == "postgresql":
            _run_postgresql_schema_compatibility()
    except SQLAlchemyError:
        logger.exception("Database initialization failed during startup")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
