from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeBaseTermStat(Base):
    __tablename__ = "knowledge_base_term_stats"
    __table_args__ = (
        Index(
            "ix_knowledge_base_term_stats_knowledge_base_id_term",
            "knowledge_base_id",
            "term",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id"),
        nullable=False,
        index=True,
    )
    term: Mapped[str] = mapped_column(String(128), nullable=False)
    doc_freq: Mapped[int] = mapped_column(Integer, nullable=False)
