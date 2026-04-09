from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ChunkTerm(Base):
    __tablename__ = "chunk_terms"
    __table_args__ = (
        Index("ix_chunk_terms_knowledge_base_id_term", "knowledge_base_id", "term"),
        Index("ix_chunk_terms_chunk_id_term", "chunk_id", "term", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id"),
        nullable=False,
        index=True,
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id"),
        nullable=False,
        index=True,
    )
    term: Mapped[str] = mapped_column(String(128), nullable=False)
    term_freq: Mapped[int] = mapped_column(Integer, nullable=False)
