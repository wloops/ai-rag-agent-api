from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.agent import AgentRunResponse, AgentTaskType, AgentWorkflowTraceItem
from app.schemas.chat import ChatCitationItem
from app.schemas.retrieval import RetrievalSearchItem
from app.services.document_service import get_chunk_offsets
from app.services.knowledge_base_service import get_active_owned_knowledge_base
from app.services.retrieval import search_chunks
from app.utils.llm_client import generate_answer

DEFAULT_AGENT_TOP_K = 5
AGENT_RELEVANCE_SCORE_THRESHOLD = settings.retrieval_relevance_threshold
EMPTY_KNOWLEDGE_BASE_MESSAGE = "当前知识库中还没有可用于执行该任务的内容。"
LOW_RELEVANCE_MESSAGE = "当前知识库中未找到足够相关内容，暂时无法稳定完成该任务。"


@dataclass
class AgentPreparedContext:
    task_type: AgentTaskType
    knowledge_base_id: int
    query: str | None
    top_k: int
    workflow_trace: list[AgentWorkflowTraceItem]


def run_agent_task(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    task_type: AgentTaskType,
    query: str | None = None,
    top_k: int = DEFAULT_AGENT_TOP_K,
) -> AgentRunResponse:
    knowledge_base = get_active_owned_knowledge_base(db, knowledge_base_id, current_user_id)
    context = AgentPreparedContext(
        task_type=task_type,
        knowledge_base_id=knowledge_base_id,
        query=_normalize_query(query),
        top_k=top_k,
        workflow_trace=[
            AgentWorkflowTraceItem(
                step="validate_knowledge_base",
                status="completed",
                detail=f"已确认知识库《{knowledge_base.name}》归属与可访问性。",
            )
        ],
    )

    if task_type == "latest_documents_digest":
        return _run_latest_documents_digest(db, knowledge_base_id, context)

    search_query = _resolve_search_query(task_type, context.query)
    retrieved_chunks = search_chunks(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        query=search_query,
        top_k=top_k,
    )
    context.workflow_trace.append(
        AgentWorkflowTraceItem(
            step="retrieve_context",
            status="completed",
            detail=f"围绕任务意图召回了 {len(retrieved_chunks)} 个候选片段。",
        )
    )

    if not retrieved_chunks:
        context.workflow_trace.append(
            AgentWorkflowTraceItem(
                step="generate_answer",
                status="skipped",
                detail="没有召回到有效片段，直接返回空知识库提示。",
            )
        )
        return AgentRunResponse(
            knowledge_base_id=knowledge_base_id,
            task_type=task_type,
            answer=EMPTY_KNOWLEDGE_BASE_MESSAGE,
            citations=[],
            workflow_trace=context.workflow_trace,
        )

    top1_score = _resolve_agent_top1_score(retrieved_chunks)
    if top1_score is None or top1_score < AGENT_RELEVANCE_SCORE_THRESHOLD:
        context.workflow_trace.append(
            AgentWorkflowTraceItem(
                step="relevance_guard",
                status="completed",
                detail=(
                    f"最高相关度 {top1_score:.3f} 低于阈值 {AGENT_RELEVANCE_SCORE_THRESHOLD:.2f}，"
                    "触发保守拒答。"
                    if top1_score is not None
                    else "候选片段缺少稳定相关度信号，触发保守拒答。"
                ),
            )
        )
        context.workflow_trace.append(
            AgentWorkflowTraceItem(
                step="generate_answer",
                status="skipped",
                detail="为避免输出低质量总结或材料，本次未调用大模型生成最终结果。",
            )
        )
        return AgentRunResponse(
            knowledge_base_id=knowledge_base_id,
            task_type=task_type,
            answer=LOW_RELEVANCE_MESSAGE,
            citations=_build_citations_from_retrieval(retrieved_chunks[:3]),
            workflow_trace=context.workflow_trace,
        )

    answer = generate_answer(
        system_prompt=_build_agent_system_prompt(task_type),
        user_prompt=_build_agent_user_prompt(task_type, search_query, context.query, retrieved_chunks),
    )
    context.workflow_trace.append(
        AgentWorkflowTraceItem(
            step="generate_answer",
            status="completed",
            detail=f"基于 {len(retrieved_chunks)} 个片段生成了任务结果。",
        )
    )
    return AgentRunResponse(
        knowledge_base_id=knowledge_base_id,
        task_type=task_type,
        answer=answer,
        citations=_build_citations_from_retrieval(retrieved_chunks[:3]),
        workflow_trace=context.workflow_trace,
    )


def _run_latest_documents_digest(db, knowledge_base_id: int, context: AgentPreparedContext) -> AgentRunResponse:
    rows = (
        db.query(Document)
        .filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.status == "success",
        )
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(max(context.top_k, 5))
        .all()
    )
    context.workflow_trace.append(
        AgentWorkflowTraceItem(
            step="list_latest_documents",
            status="completed",
            detail=f"获取到 {len(rows)} 份最近成功入库的文档。",
        )
    )

    if not rows:
        context.workflow_trace.append(
            AgentWorkflowTraceItem(
                step="generate_answer",
                status="skipped",
                detail="知识库内暂无成功建库的文档，无法生成最新文档汇总。",
            )
        )
        return AgentRunResponse(
            knowledge_base_id=context.knowledge_base_id,
            task_type=context.task_type,
            answer=EMPTY_KNOWLEDGE_BASE_MESSAGE,
            citations=[],
            workflow_trace=context.workflow_trace,
        )

    latest_chunks = _load_latest_document_chunks(db, [document.id for document in rows])
    citations = _build_citations_from_retrieval(latest_chunks[:3])
    answer = generate_answer(
        system_prompt=_build_agent_system_prompt(context.task_type),
        user_prompt=_build_latest_documents_prompt(rows, latest_chunks, context.query),
    )
    context.workflow_trace.append(
        AgentWorkflowTraceItem(
            step="generate_answer",
            status="completed",
            detail=f"基于最近 {len(rows)} 份文档生成了汇总内容。",
        )
    )
    return AgentRunResponse(
        knowledge_base_id=context.knowledge_base_id,
        task_type=context.task_type,
        answer=answer,
        citations=citations,
        workflow_trace=context.workflow_trace,
    )


def _normalize_query(query: str | None) -> str | None:
    if query is None:
        return None

    normalized_query = query.strip()
    return normalized_query or None


def _resolve_search_query(task_type: AgentTaskType, query: str | None) -> str:
    if task_type == "knowledge_base_qa":
        if not query:
            raise ValueError("Question is required for knowledge_base_qa")
        return query

    defaults = {
        "knowledge_base_summary": "请总结这个知识库的核心主题、业务价值、产品结构、技术亮点与适合面试时讲述的重点。",
        "interview_material": "请提炼这个知识库里最适合面试表达的公司背景、产品能力、岗位要求映射、技术关键词与风险提示。",
    }
    return query or defaults[task_type]


def _build_agent_system_prompt(task_type: AgentTaskType) -> str:
    prompts = {
        "knowledge_base_qa": "你是知识库 Agent。你只能基于给定上下文回答，回答尽量结构化，无法确认时明确说明不确定。",
        "knowledge_base_summary": "你是知识库总结 Agent。请提炼业务背景、产品能力、技术结构和可用于面试表达的亮点，避免空泛描述。",
        "latest_documents_digest": "你是最新文档汇总 Agent。请用简洁结构总结最近文档的主题、价值和可追问方向。",
        "interview_material": "你是面试材料 Agent。请围绕公司、产品、岗位和候选人项目映射，输出适合面试前复习的材料。",
    }
    return prompts[task_type]


def _build_agent_user_prompt(
    task_type: AgentTaskType,
    search_query: str,
    raw_query: str | None,
    retrieved_chunks: list[RetrievalSearchItem],
) -> str:
    context_blocks = []
    for index, item in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"filename: {item.filename}",
                    f"chunk_index: {item.chunk_index}",
                    f"score: {item.score:.3f}",
                    f"content: {item.content}",
                ]
            )
        )

    task_guidance = {
        "knowledge_base_qa": "请直接回答问题，并在关键结论后标注 [S1] 这类来源编号。",
        "knowledge_base_summary": "请按“知识库主题 / 业务价值 / 技术亮点 / 面试可讲点”输出。",
        "interview_material": "请按“公司速览 / 岗位映射 / 技术关键词 / 面试话术”输出。",
    }

    sections = []
    if raw_query:
        sections.append(f"用户输入：\n{raw_query}")
    sections.append(f"任务检索问题：\n{search_query}")
    sections.append(f"任务要求：\n{task_guidance[task_type]}")
    sections.append("上下文：\n\n" + "\n\n".join(context_blocks))
    return "\n\n".join(sections)


def _load_latest_document_chunks(db, document_ids: list[int]) -> list[RetrievalSearchItem]:
    if not document_ids:
        return []

    rows = (
        db.query(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.id.in_(document_ids))
        .order_by(Document.created_at.desc(), Chunk.chunk_index.asc(), Chunk.id.asc())
        .all()
    )

    first_chunk_by_document: dict[int, RetrievalSearchItem] = {}
    for chunk, document in rows:
        if document.id in first_chunk_by_document:
            continue

        start_offset, end_offset = get_chunk_offsets(chunk)
        first_chunk_by_document[document.id] = RetrievalSearchItem(
            chunk_id=chunk.id,
            document_id=document.id,
            filename=document.filename,
            chunk_index=chunk.chunk_index,
            start_offset=start_offset,
            end_offset=end_offset,
            content=chunk.content,
            score=1.0,
            guard_score=1.0,
            source_channels=["latest"],
        )

    ordered_items: list[RetrievalSearchItem] = []
    for document_id in document_ids:
        item = first_chunk_by_document.get(document_id)
        if item is not None:
            ordered_items.append(item)
    return ordered_items


def _build_latest_documents_prompt(
    documents: list[Document],
    latest_chunks: list[RetrievalSearchItem],
    query: str | None,
) -> str:
    document_lines = [
        f"- {document.filename} | status={document.status} | created_at={document.created_at.isoformat()}"
        for document in documents
    ]
    chunk_lines = [
        f"[S{index}] {item.filename} / chunk {item.chunk_index}: {item.content}"
        for index, item in enumerate(latest_chunks, start=1)
    ]

    sections = []
    if query:
        sections.append(f"补充要求：\n{query}")
    sections.append("最近文档列表：\n" + "\n".join(document_lines))
    if chunk_lines:
        sections.append("文档代表片段：\n" + "\n".join(chunk_lines))
    sections.append("请按“最近更新 / 每份文档大意 / 值得继续追问的问题”输出。")
    return "\n\n".join(sections)


def _build_citations_from_retrieval(
    retrieved_chunks: list[RetrievalSearchItem],
) -> list[ChatCitationItem]:
    citations: list[ChatCitationItem] = []
    for item in retrieved_chunks:
        citations.append(
            ChatCitationItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                chunk_index=item.chunk_index,
                start_offset=item.start_offset,
                end_offset=item.end_offset,
                snippet=_build_snippet(item.content),
            )
        )
    return citations


def _build_snippet(content: str, max_length: int = 140) -> str:
    normalized_content = " ".join(content.split())
    if len(normalized_content) <= max_length:
        return normalized_content
    return normalized_content[:max_length].rstrip() + "..."


def _resolve_agent_top1_score(retrieved_chunks: list[RetrievalSearchItem]) -> float | None:
    if not retrieved_chunks:
        return None
    top_candidate = retrieved_chunks[0]
    if top_candidate.guard_score is not None:
        return top_candidate.guard_score
    return top_candidate.score
