from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import re

import jieba
from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.chunk_term import ChunkTerm
from app.models.document import Document
from app.models.knowledge_base_term_stat import KnowledgeBaseTermStat

TOKEN_MIN_LENGTH = 1
LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def tokenize_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    # 中文场景下 BM25 需要稳定的词项切分；这里统一收口，
    # 让文档入库和查询阶段使用同一套分词规则，避免检索口径漂移。
    normalized_text = text.strip().lower()
    tokens: list[str] = []
    for token in jieba.lcut(normalized_text, cut_all=False):
        normalized_token = token.strip().lower()
        if not normalized_token:
            continue
        if LATIN_TOKEN_PATTERN.fullmatch(normalized_token):
            tokens.append(normalized_token)
            continue
        if len(normalized_token) >= TOKEN_MIN_LENGTH:
            tokens.append(normalized_token)
    return tokens


def build_chunk_term_frequencies(content: str) -> Counter[str]:
    return Counter(tokenize_text(content))


def rebuild_document_bm25_index(
    db: Session,
    document: Document,
    chunks: Iterable[Chunk] | None = None,
) -> None:
    if chunks is None:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index.asc(), Chunk.id.asc())
            .all()
        )

    chunk_list = list(chunks)
    chunk_ids = [chunk.id for chunk in chunk_list if chunk.id is not None]
    if chunk_ids:
        db.query(ChunkTerm).filter(ChunkTerm.chunk_id.in_(chunk_ids)).delete(
            synchronize_session=False
        )

    term_rows: list[ChunkTerm] = []
    for chunk in chunk_list:
        term_frequencies = build_chunk_term_frequencies(chunk.content)
        chunk.token_count = sum(term_frequencies.values())
        for term, term_freq in term_frequencies.items():
            term_rows.append(
                ChunkTerm(
                    chunk_id=chunk.id,
                    knowledge_base_id=document.knowledge_base_id,
                    term=term,
                    term_freq=term_freq,
                )
            )

    if term_rows:
        db.add_all(term_rows)

    rebuild_knowledge_base_bm25_stats(db, document.knowledge_base_id)


def rebuild_knowledge_base_bm25_stats(db: Session, knowledge_base_id: int) -> None:
    db.query(KnowledgeBaseTermStat).filter(
        KnowledgeBaseTermStat.knowledge_base_id == knowledge_base_id
    ).delete(synchronize_session=False)

    rows = (
        db.query(
            ChunkTerm.term.label("term"),
            func.count(distinct(ChunkTerm.chunk_id)).label("doc_freq"),
        )
        .join(Chunk, Chunk.id == ChunkTerm.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .filter(
            ChunkTerm.knowledge_base_id == knowledge_base_id,
            Document.status == "success",
        )
        .group_by(ChunkTerm.term)
        .all()
    )

    if rows:
        db.add_all(
            [
                KnowledgeBaseTermStat(
                    knowledge_base_id=knowledge_base_id,
                    term=row.term,
                    doc_freq=int(row.doc_freq),
                )
                for row in rows
            ]
        )


def rebuild_all_bm25_indexes(
    db: Session,
    knowledge_base_id: int | None = None,
) -> int:
    query = db.query(Document).filter(Document.status == "success")
    if knowledge_base_id is not None:
        query = query.filter(Document.knowledge_base_id == knowledge_base_id)

    documents = query.order_by(Document.knowledge_base_id.asc(), Document.id.asc()).all()
    rebuilt = 0
    rebuilt_knowledge_base_ids: set[int] = set()
    for document in documents:
        rebuild_document_bm25_index(db, document)
        rebuilt += 1
        rebuilt_knowledge_base_ids.add(document.knowledge_base_id)

    for current_knowledge_base_id in rebuilt_knowledge_base_ids:
        rebuild_knowledge_base_bm25_stats(db, current_knowledge_base_id)

    db.commit()
    return rebuilt
