from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from time import perf_counter

from sqlalchemy import func

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.chunk_term import ChunkTerm
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_term_stat import KnowledgeBaseTermStat
from app.schemas.retrieval import RetrievalSearchItem
from app.services.document_service import get_chunk_offsets
from app.services.knowledge_base_service import get_active_owned_knowledge_base
from app.services.bm25_index import tokenize_text
from app.utils.embeddings import embed_text
from app.utils.rerank_client import can_rerank, rerank_documents

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
FILENAME_MATCH_BONUS = 0.35


@dataclass
class RetrievalTrace:
    embedding_ms: int | None = None
    dense_candidates_count: int = 0
    bm25_candidates_count: int = 0
    fusion_candidates_count: int = 0
    rerank_applied: bool = False


@dataclass
class RetrievalCandidate:
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    content: str
    source_channels: set[str] = field(default_factory=set)
    score: float = 0.0
    guard_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fusion_rank: int | None = None
    rerank_rank: int | None = None

    def to_schema(self) -> RetrievalSearchItem:
        return RetrievalSearchItem(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            filename=self.filename,
            chunk_index=self.chunk_index,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            content=self.content,
            score=self.score,
            guard_score=self.guard_score,
            source_channels=sorted(self.source_channels),
            dense_score=self.dense_score,
            bm25_score=self.bm25_score,
            fusion_score=self.fusion_score,
            rerank_score=self.rerank_score,
            dense_rank=self.dense_rank,
            bm25_rank=self.bm25_rank,
            fusion_rank=self.fusion_rank,
            rerank_rank=self.rerank_rank,
        )


@dataclass
class RetrievalPipelineResult:
    dense_candidates: list[RetrievalSearchItem]
    bm25_candidates: list[RetrievalSearchItem]
    fused_candidates: list[RetrievalSearchItem]
    final_candidates: list[RetrievalSearchItem]
    trace: RetrievalTrace


def search_chunks(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    query: str,
    top_k: int = 3,
    trace: RetrievalTrace | None = None,
) -> list[RetrievalSearchItem]:
    pipeline_result = run_retrieval_pipeline(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        query=query,
        top_k=top_k,
        trace=trace,
    )
    return pipeline_result.final_candidates


def run_retrieval_pipeline(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    query: str,
    top_k: int = 3,
    trace: RetrievalTrace | None = None,
) -> RetrievalPipelineResult:
    get_active_owned_knowledge_base(db, knowledge_base_id, current_user_id)

    query_embedding = _embed_query(query, trace)
    dense_candidates = retrieve_dense_candidates(
        db=db,
        knowledge_base_id=knowledge_base_id,
        query_embedding=query_embedding,
        candidate_k=settings.retrieval_dense_candidate_k,
    )
    bm25_candidates = retrieve_bm25_candidates(
        db=db,
        knowledge_base_id=knowledge_base_id,
        query=query,
        candidate_k=settings.retrieval_bm25_candidate_k,
    )
    fused_candidates = fuse_candidates(
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        candidate_k=settings.retrieval_fusion_candidate_k,
    )
    final_candidates, rerank_applied = rerank_candidates(
        query=query,
        candidates=fused_candidates,
        top_k=top_k,
    )

    current_trace = trace or RetrievalTrace()
    current_trace.dense_candidates_count = len(dense_candidates)
    current_trace.bm25_candidates_count = len(bm25_candidates)
    current_trace.fusion_candidates_count = len(fused_candidates)
    current_trace.rerank_applied = rerank_applied

    return RetrievalPipelineResult(
        dense_candidates=[candidate.to_schema() for candidate in dense_candidates],
        bm25_candidates=[candidate.to_schema() for candidate in bm25_candidates],
        fused_candidates=[candidate.to_schema() for candidate in fused_candidates],
        final_candidates=[candidate.to_schema() for candidate in final_candidates],
        trace=current_trace,
    )


def retrieve_dense_candidates(
    db,
    knowledge_base_id: int,
    query_embedding: list[float],
    candidate_k: int,
) -> list[RetrievalCandidate]:
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        candidates = _search_dense_candidates_with_pgvector(
            db,
            knowledge_base_id=knowledge_base_id,
            query_embedding=query_embedding,
            candidate_k=candidate_k,
        )
    else:
        candidates = _search_dense_candidates_in_python(
            db,
            knowledge_base_id=knowledge_base_id,
            query_embedding=query_embedding,
            candidate_k=candidate_k,
        )

    for rank, candidate in enumerate(candidates, start=1):
        candidate.dense_rank = rank
        candidate.score = candidate.dense_score or 0.0
        candidate.guard_score = candidate.dense_score
        candidate.source_channels.add("dense")
    return candidates


def retrieve_bm25_candidates(
    db,
    knowledge_base_id: int,
    query: str,
    candidate_k: int,
) -> list[RetrievalCandidate]:
    query_terms = tokenize_text(query)
    if not query_terms:
        return []

    total_chunks, average_doc_len = _get_bm25_collection_stats(db, knowledge_base_id)
    if total_chunks == 0:
        return []

    term_stats_rows = (
        db.query(KnowledgeBaseTermStat)
        .filter(
            KnowledgeBaseTermStat.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseTermStat.term.in_(query_terms),
        )
        .all()
    )
    doc_freq_map = {row.term: row.doc_freq for row in term_stats_rows}
    if not doc_freq_map:
        return []

    rows = (
        db.query(Chunk, Document, ChunkTerm.term, ChunkTerm.term_freq)
        .join(Document, Chunk.document_id == Document.id)
        .join(ChunkTerm, ChunkTerm.chunk_id == Chunk.id)
        .filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.status == "success",
            ChunkTerm.term.in_(query_terms),
        )
        .all()
    )

    grouped_terms: dict[int, dict[str, int]] = defaultdict(dict)
    chunk_map: dict[int, tuple[Chunk, Document]] = {}
    for chunk, document, term, term_freq in rows:
        chunk_map[chunk.id] = (chunk, document)
        grouped_terms[chunk.id][term] = int(term_freq)

    scored_candidates: list[RetrievalCandidate] = []
    query_term_set = set(query_terms)
    for chunk_id, term_frequencies in grouped_terms.items():
        chunk, document = chunk_map[chunk_id]
        bm25_score = _calculate_bm25_score(
            term_frequencies=term_frequencies,
            doc_freq_map=doc_freq_map,
            total_chunks=total_chunks,
            document_length=max(chunk.token_count, 1),
            average_doc_len=average_doc_len,
        )
        filename_terms = set(tokenize_text(document.filename))
        filename_overlap = len(query_term_set & filename_terms)
        if filename_overlap > 0:
            bm25_score += filename_overlap * FILENAME_MATCH_BONUS

        start_offset, end_offset = get_chunk_offsets(chunk)
        scored_candidates.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                chunk_index=chunk.chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                content=chunk.content,
                bm25_score=bm25_score,
                guard_score=_normalize_sparse_score(bm25_score),
                score=bm25_score,
                source_channels={"bm25"},
            )
        )

    scored_candidates.sort(
        key=lambda item: (item.bm25_score or 0.0, item.chunk_id),
        reverse=True,
    )
    limited_candidates = scored_candidates[:candidate_k]
    for rank, candidate in enumerate(limited_candidates, start=1):
        candidate.bm25_rank = rank
    return limited_candidates


def fuse_candidates(
    dense_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    candidate_k: int,
) -> list[RetrievalCandidate]:
    by_chunk_id: dict[int, RetrievalCandidate] = {}
    for candidate in [*dense_candidates, *bm25_candidates]:
        existing = by_chunk_id.get(candidate.chunk_id)
        if existing is None:
            by_chunk_id[candidate.chunk_id] = RetrievalCandidate(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                chunk_index=candidate.chunk_index,
                start_offset=candidate.start_offset,
                end_offset=candidate.end_offset,
                content=candidate.content,
                source_channels=set(candidate.source_channels),
                dense_score=candidate.dense_score,
                bm25_score=candidate.bm25_score,
                dense_rank=candidate.dense_rank,
                bm25_rank=candidate.bm25_rank,
                score=candidate.score,
                guard_score=candidate.guard_score,
            )
            continue

        existing.source_channels.update(candidate.source_channels)
        existing.dense_score = existing.dense_score or candidate.dense_score
        existing.bm25_score = existing.bm25_score or candidate.bm25_score
        existing.dense_rank = existing.dense_rank or candidate.dense_rank
        existing.bm25_rank = existing.bm25_rank or candidate.bm25_rank
        existing.guard_score = max(
            value
            for value in [
                existing.guard_score,
                candidate.guard_score,
            ]
            if value is not None
        )

    fused_candidates = list(by_chunk_id.values())
    for candidate in fused_candidates:
        fusion_score = 0.0
        if candidate.dense_rank is not None:
            fusion_score += 1.0 / (RRF_K + candidate.dense_rank)
        if candidate.bm25_rank is not None:
            fusion_score += 1.0 / (RRF_K + candidate.bm25_rank)
        candidate.fusion_score = fusion_score
        candidate.score = fusion_score

    fused_candidates.sort(
        key=lambda item: (
            item.fusion_score or 0.0,
            item.dense_score or 0.0,
            item.bm25_score or 0.0,
            item.chunk_id,
        ),
        reverse=True,
    )
    limited_candidates = fused_candidates[:candidate_k]
    for rank, candidate in enumerate(limited_candidates, start=1):
        candidate.fusion_rank = rank
    return limited_candidates


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    top_k: int,
) -> tuple[list[RetrievalCandidate], bool]:
    if not candidates:
        return [], False

    final_candidates = [candidate for candidate in candidates]
    if not can_rerank():
        for candidate in final_candidates:
            candidate.score = _resolve_final_score(candidate)
            candidate.guard_score = _resolve_guard_score(candidate)
        final_candidates.sort(
            key=lambda item: (
                item.fusion_score or 0.0,
                item.dense_score or 0.0,
                item.bm25_score or 0.0,
                item.chunk_id,
            ),
            reverse=True,
        )
        return final_candidates[:top_k], False

    rerank_top_n = min(len(final_candidates), max(top_k, settings.rerank_top_n))
    try:
        rerank_results = rerank_documents(
            query=query,
            documents=[candidate.content for candidate in final_candidates],
            top_n=rerank_top_n,
        )
    except RuntimeError:
        for candidate in final_candidates:
            candidate.score = _resolve_final_score(candidate)
            candidate.guard_score = _resolve_guard_score(candidate)
        final_candidates.sort(
            key=lambda item: (
                item.fusion_score or 0.0,
                item.dense_score or 0.0,
                item.bm25_score or 0.0,
                item.chunk_id,
            ),
            reverse=True,
        )
        return final_candidates[:top_k], False

    reranked_by_index = {index: score for index, score in rerank_results}
    for index, candidate in enumerate(final_candidates):
        rerank_score = reranked_by_index.get(index)
        if rerank_score is not None:
            candidate.rerank_score = rerank_score
            candidate.source_channels.add("rerank")
            candidate.guard_score = rerank_score
            candidate.score = rerank_score
        else:
            candidate.guard_score = _resolve_guard_score(candidate)
            candidate.score = _resolve_final_score(candidate)

    final_candidates.sort(
        key=lambda item: (
            item.rerank_score if item.rerank_score is not None else -1.0,
            item.fusion_score or 0.0,
            item.dense_score or 0.0,
            item.bm25_score or 0.0,
            item.chunk_id,
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(final_candidates, start=1):
        if candidate.rerank_score is not None:
            candidate.rerank_rank = rank
    return final_candidates[:top_k], True


def _embed_query(query: str, trace: RetrievalTrace | None) -> list[float]:
    embedding_started_at = perf_counter()
    query_embedding = embed_text(query)
    if trace is not None:
        trace.embedding_ms = _elapsed_ms(embedding_started_at)
    return query_embedding


def _search_dense_candidates_with_pgvector(
    db,
    knowledge_base_id: int,
    query_embedding: list[float],
    candidate_k: int,
) -> list[RetrievalCandidate]:
    distance_expr = Chunk.embedding.cosine_distance(query_embedding)
    start_offset_expr = Chunk.metadata_json["start_char"].as_integer()
    end_offset_expr = Chunk.metadata_json["end_char"].as_integer()
    rows = (
        db.query(
            Chunk.id.label("chunk_id"),
            Document.id.label("document_id"),
            Document.filename.label("filename"),
            Chunk.chunk_index.label("chunk_index"),
            start_offset_expr.label("start_offset"),
            end_offset_expr.label("end_offset"),
            Chunk.content.label("content"),
            (1 - distance_expr).label("dense_score"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.deleted_at.is_(None),
            Document.status == "success",
            Chunk.embedding.is_not(None),
        )
        .order_by(distance_expr.asc())
        .limit(candidate_k)
        .all()
    )

    return [
        RetrievalCandidate(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            start_offset=int(row.start_offset or 0),
            end_offset=int(row.end_offset or row.start_offset or 0),
            content=row.content,
            dense_score=float(row.dense_score),
        )
        for row in rows
    ]


def _search_dense_candidates_in_python(
    db,
    knowledge_base_id: int,
    query_embedding: list[float],
    candidate_k: int,
) -> list[RetrievalCandidate]:
    rows = (
        db.query(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.deleted_at.is_(None),
            Document.status == "success",
            Chunk.embedding.is_not(None),
        )
        .all()
    )

    scored_results: list[RetrievalCandidate] = []
    for chunk, document in rows:
        if not chunk.embedding:
            continue
        score = _cosine_similarity(query_embedding, chunk.embedding)
        start_offset, end_offset = get_chunk_offsets(chunk)
        scored_results.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                chunk_index=chunk.chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                content=chunk.content,
                dense_score=score,
            )
        )

    scored_results.sort(
        key=lambda item: (item.dense_score or 0.0, item.chunk_id),
        reverse=True,
    )
    return scored_results[:candidate_k]


def _get_bm25_collection_stats(db, knowledge_base_id: int) -> tuple[int, float]:
    total_chunks = (
        db.query(func.count(Chunk.id))
        .join(Document, Chunk.document_id == Document.id)
        .filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.status == "success",
            Chunk.token_count > 0,
        )
        .scalar()
        or 0
    )
    average_doc_len = (
        db.query(func.avg(Chunk.token_count))
        .join(Document, Chunk.document_id == Document.id)
        .filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.status == "success",
            Chunk.token_count > 0,
        )
        .scalar()
    )
    return int(total_chunks), float(average_doc_len or 1.0)


def _calculate_bm25_score(
    term_frequencies: dict[str, int],
    doc_freq_map: dict[str, int],
    total_chunks: int,
    document_length: int,
    average_doc_len: float,
) -> float:
    score = 0.0
    safe_average_doc_len = max(average_doc_len, 1.0)
    for term, term_freq in term_frequencies.items():
        doc_freq = max(doc_freq_map.get(term, 0), 1)
        idf = math.log(1 + ((total_chunks - doc_freq + 0.5) / (doc_freq + 0.5)))
        denominator = term_freq + BM25_K1 * (
            1 - BM25_B + BM25_B * (document_length / safe_average_doc_len)
        )
        score += idf * (term_freq * (BM25_K1 + 1)) / denominator
    return score


def _normalize_sparse_score(score: float) -> float:
    if score <= 0:
        return 0.0
    return score / (score + 1.0)


def _resolve_guard_score(candidate: RetrievalCandidate) -> float | None:
    if candidate.rerank_score is not None:
        return candidate.rerank_score

    score_candidates = [
        candidate.dense_score,
        _normalize_sparse_score(candidate.bm25_score or 0.0)
        if candidate.bm25_score is not None
        else None,
    ]
    filtered_scores = [score for score in score_candidates if score is not None]
    if not filtered_scores:
        return None
    return max(filtered_scores)


def _resolve_final_score(candidate: RetrievalCandidate) -> float:
    if candidate.rerank_score is not None:
        return candidate.rerank_score
    if candidate.fusion_score is not None:
        return candidate.fusion_score
    if candidate.dense_score is not None:
        return candidate.dense_score
    if candidate.bm25_score is not None:
        return candidate.bm25_score
    return 0.0


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0

    dot_product = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return -1.0

    return dot_product / (left_norm * right_norm)


def _elapsed_ms(started_at: float) -> int:
    return int(round((perf_counter() - started_at) * 1000))
