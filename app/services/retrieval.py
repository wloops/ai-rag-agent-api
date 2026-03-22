import math

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.schemas.retrieval import RetrievalSearchItem
from app.utils.embeddings import embed_text


def search_chunks(db, current_user_id: int, knowledge_base_id: int, query: str, top_k: int = 3):
    knowledge_base = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == current_user_id,
        )
        .first()
    )
    if not knowledge_base:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 查询时只为用户问题生成一次 embedding，避免把计算成本放到每个 chunk 上。
    query_embedding = embed_text(query)

    if db.bind is not None and db.bind.dialect.name == "postgresql":
        return _search_chunks_with_pgvector(db, knowledge_base_id, query_embedding, top_k)

    return _search_chunks_in_python(db, knowledge_base_id, query_embedding, top_k)


def _search_chunks_with_pgvector(db, knowledge_base_id: int, query_embedding: list[float], top_k: int):
    distance_expr = Chunk.embedding.cosine_distance(query_embedding)
    rows = (
        db.query(
            Chunk.id.label("chunk_id"),
            Document.id.label("document_id"),
            Document.filename.label("filename"),
            Chunk.chunk_index.label("chunk_index"),
            Chunk.content.label("content"),
            (1 - distance_expr).label("score"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            Chunk.embedding.is_not(None),
        )
        .order_by(distance_expr.asc())
        .limit(top_k)
        .all()
    )

    return [
        RetrievalSearchItem(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            content=row.content,
            score=float(row.score),
        )
        for row in rows
    ]


def _search_chunks_in_python(db, knowledge_base_id: int, query_embedding: list[float], top_k: int):
    rows = (
        db.query(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            Chunk.embedding.is_not(None),
        )
        .all()
    )

    scored_results: list[RetrievalSearchItem] = []
    for chunk, document in rows:
        if not chunk.embedding:
            continue

        score = _cosine_similarity(query_embedding, chunk.embedding)
        scored_results.append(
            RetrievalSearchItem(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=score,
            )
        )

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return scored_results[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0

    dot_product = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return -1.0

    return dot_product / (left_norm * right_norm)
