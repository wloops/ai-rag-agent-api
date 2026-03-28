import re
from collections.abc import Generator
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from fastapi import HTTPException

from app.models.conversation import Conversation
from app.schemas.chat import (
    ChatAskResponse,
    ChatCitationItem,
    DebugInfo,
    DebugRetrievedChunkItem,
    RetrievedChunkDebugItem,
)
from app.schemas.retrieval import RetrievalSearchItem
from app.services.conversation_service import (
    get_owned_knowledge_base,
    list_recent_conversation_messages,
    resolve_conversation_for_question,
    save_message,
)
from app.services.retrieval import RetrievalTrace, search_chunks
from app.utils.llm_client import generate_answer, rewrite_question, stream_answer

RELEVANCE_SCORE_THRESHOLD = 0.35
FINAL_CONTEXT_PREVIEW_MAX_LENGTH = 2000
RECENT_MESSAGE_WINDOW = 6
RECENT_TURN_SUMMARY_MESSAGES = 4
NO_ANSWER_MESSAGE = "当前知识库中未找到足够相关内容。"
SOURCE_ID_PATTERN = re.compile(r"\[(S\d+)\]")


@dataclass
class AskPreparedContext:
    conversation: Conversation
    normalized_question: str
    standalone_question: str
    recent_turn_summary: str | None
    retrieved_chunks: list[RetrievalSearchItem]
    top_k: int
    debug: bool
    retrieval_trace: RetrievalTrace | None
    retrieval_ms: int
    total_started_at: float
    top1_score: float | None
    decision: Literal["answer", "reject"]


def ask_knowledge_base(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int = 3,
    debug: bool = False,
    conversation_id: int | None = None,
) -> ChatAskResponse:
    context = _prepare_ask_context(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        question=question,
        top_k=top_k,
        debug=debug,
        conversation_id=conversation_id,
    )

    if context.decision == "reject":
        return _finalize_response(
            db=db,
            context=context,
            assistant_message=NO_ANSWER_MESSAGE,
            citations=[],
            final_context_preview=None,
            llm_ms=0,
            knowledge_base_id=knowledge_base_id,
        )

    source_mapping = _build_source_mapping(context.retrieved_chunks)
    context_blocks = _build_context_blocks(source_mapping)
    final_context_preview = _build_final_context_preview(
        standalone_question=context.standalone_question,
        recent_turn_summary=context.recent_turn_summary,
        context_blocks=context_blocks,
    )

    llm_started_at = perf_counter()
    assistant_message = generate_answer(
        system_prompt=_build_system_prompt(),
        user_prompt=_build_user_prompt(
            original_question=context.normalized_question,
            standalone_question=context.standalone_question,
            source_mapping=source_mapping,
            context_blocks=context_blocks,
            recent_turn_summary=context.recent_turn_summary,
        ),
    )
    llm_ms = _elapsed_ms(llm_started_at)
    citations = _resolve_citations(
        retrieved_chunks=context.retrieved_chunks,
        source_mapping=source_mapping,
        assistant_message=assistant_message,
    )
    return _finalize_response(
        db=db,
        context=context,
        assistant_message=assistant_message,
        citations=citations,
        final_context_preview=final_context_preview,
        llm_ms=llm_ms,
        knowledge_base_id=knowledge_base_id,
    )


def stream_knowledge_base_events(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int = 3,
    debug: bool = False,
    conversation_id: int | None = None,
) -> Generator[tuple[str, dict], None, None]:
    context = _prepare_ask_context(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        question=question,
        top_k=top_k,
        debug=debug,
        conversation_id=conversation_id,
    )
    yield "start", {"conversation_id": context.conversation.id}

    if context.decision == "reject":
        response = _finalize_response(
            db=db,
            context=context,
            assistant_message=NO_ANSWER_MESSAGE,
            citations=[],
            final_context_preview=None,
            llm_ms=0,
            knowledge_base_id=knowledge_base_id,
        )
        yield "final", response.model_dump(mode="json")
        return

    source_mapping = _build_source_mapping(context.retrieved_chunks)
    context_blocks = _build_context_blocks(source_mapping)
    final_context_preview = _build_final_context_preview(
        standalone_question=context.standalone_question,
        recent_turn_summary=context.recent_turn_summary,
        context_blocks=context_blocks,
    )

    llm_started_at = perf_counter()
    answer_chunks: list[str] = []
    for delta in stream_answer(
        system_prompt=_build_system_prompt(),
        user_prompt=_build_user_prompt(
            original_question=context.normalized_question,
            standalone_question=context.standalone_question,
            source_mapping=source_mapping,
            context_blocks=context_blocks,
            recent_turn_summary=context.recent_turn_summary,
        ),
    ):
        answer_chunks.append(delta)
        yield "delta", {"content": delta}

    assistant_message = "".join(answer_chunks).strip()
    if not assistant_message:
        raise RuntimeError("LLM returned empty content")

    llm_ms = _elapsed_ms(llm_started_at)
    citations = _resolve_citations(
        retrieved_chunks=context.retrieved_chunks,
        source_mapping=source_mapping,
        assistant_message=assistant_message,
    )
    response = _finalize_response(
        db=db,
        context=context,
        assistant_message=assistant_message,
        citations=citations,
        final_context_preview=final_context_preview,
        llm_ms=llm_ms,
        knowledge_base_id=knowledge_base_id,
    )
    yield "final", response.model_dump(mode="json")


def _prepare_ask_context(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int,
    debug: bool,
    conversation_id: int | None = None,
) -> AskPreparedContext:
    total_started_at = perf_counter()
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank")

    get_owned_knowledge_base(db, knowledge_base_id, current_user_id)
    conversation = resolve_conversation_for_question(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        question=normalized_question,
        conversation_id=conversation_id,
    )

    recent_messages = list_recent_conversation_messages(
        db=db,
        conversation_id=conversation.id,
        current_user_id=current_user_id,
        limit=RECENT_MESSAGE_WINDOW,
    )
    standalone_question = _rewrite_question_with_history(
        recent_messages,
        normalized_question,
    )
    recent_turn_summary = _build_recent_turn_summary(recent_messages)

    save_message(
        db=db,
        conversation=conversation,
        role="user",
        content=normalized_question,
    )

    retrieval_trace = RetrievalTrace() if debug else None
    retrieval_started_at = perf_counter()
    retrieved_chunks = search_chunks(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        query=standalone_question,
        top_k=top_k,
        trace=retrieval_trace,
    )
    retrieval_ms = _elapsed_ms(retrieval_started_at)
    top1_score = retrieved_chunks[0].score if retrieved_chunks else None
    decision: Literal["answer", "reject"] = (
        "answer"
        if top1_score is not None and top1_score >= RELEVANCE_SCORE_THRESHOLD
        else "reject"
    )

    return AskPreparedContext(
        conversation=conversation,
        normalized_question=normalized_question,
        standalone_question=standalone_question,
        recent_turn_summary=recent_turn_summary,
        retrieved_chunks=retrieved_chunks,
        top_k=top_k,
        debug=debug,
        retrieval_trace=retrieval_trace,
        retrieval_ms=retrieval_ms,
        total_started_at=total_started_at,
        top1_score=top1_score,
        decision=decision,
    )


def _build_system_prompt() -> str:
    return (
        "你是企业知识库问答助手。"
        "你只能依据提供的上下文回答，不能使用上下文之外的知识补全。"
        "如果上下文没有足够答案，必须明确回答“当前知识库中未找到足够相关内容”或“不知道”。"
        "回答尽量结构化，并在使用证据时引用 source id，例如 [S1]、[S2]。"
    )


def _build_user_prompt(
    original_question: str,
    standalone_question: str,
    source_mapping: dict[str, RetrievalSearchItem],
    context_blocks: list[str] | None = None,
    recent_turn_summary: str | None = None,
) -> str:
    if context_blocks is None:
        context_blocks = _build_context_blocks(source_mapping)

    sections = [
        f"用户原始问题：\n{original_question}",
        f"用于检索的独立问题：\n{standalone_question}",
    ]
    if recent_turn_summary:
        sections.append(f"最近对话摘要：\n{recent_turn_summary}")
    sections.append("请基于以下上下文回答：\n\n" + "\n\n".join(context_blocks))
    return "\n\n".join(sections)


def _build_context_blocks(source_mapping: dict[str, RetrievalSearchItem]) -> list[str]:
    context_blocks: list[str] = []
    for source_id, item in source_mapping.items():
        context_blocks.append(
            "\n".join(
                [
                    source_id,
                    f"document_id: {item.document_id}",
                    f"filename: {item.filename}",
                    f"chunk_index: {item.chunk_index}",
                    f"content: {item.content}",
                ]
            )
        )
    return context_blocks


def _build_source_mapping(
    retrieved_chunks: list[RetrievalSearchItem],
) -> dict[str, RetrievalSearchItem]:
    return {f"S{index + 1}": item for index, item in enumerate(retrieved_chunks)}


def _parse_cited_source_ids(answer: str) -> list[str]:
    matched = SOURCE_ID_PATTERN.findall(answer)
    unique_source_ids: list[str] = []
    for source_id in matched:
        if source_id not in unique_source_ids:
            unique_source_ids.append(source_id)
    return unique_source_ids


def _build_citations(
    source_mapping: dict[str, RetrievalSearchItem],
    source_ids: list[str],
) -> list[ChatCitationItem]:
    citations: list[ChatCitationItem] = []
    for source_id in source_ids:
        item = source_mapping.get(source_id)
        if item is None:
            continue

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


def _resolve_citations(
    retrieved_chunks: list[RetrievalSearchItem],
    source_mapping: dict[str, RetrievalSearchItem],
    assistant_message: str,
) -> list[ChatCitationItem]:
    cited_source_ids = _parse_cited_source_ids(assistant_message)
    citations = _build_citations(source_mapping, cited_source_ids)
    if citations:
        return citations
    return _build_fallback_citations(retrieved_chunks)


def _build_fallback_citations(
    retrieved_chunks: list[RetrievalSearchItem],
) -> list[ChatCitationItem]:
    return [
        ChatCitationItem(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            filename=item.filename,
            chunk_index=item.chunk_index,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            snippet=_build_snippet(item.content),
        )
        for item in retrieved_chunks[:3]
    ]


def _build_debug_items(
    retrieved_chunks: list[RetrievalSearchItem],
) -> list[RetrievedChunkDebugItem]:
    return [
        RetrievedChunkDebugItem(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            filename=item.filename,
            chunk_index=item.chunk_index,
            content=item.content,
            score=item.score,
        )
        for item in retrieved_chunks
    ]


def _build_detailed_debug_items(
    retrieved_chunks: list[RetrievalSearchItem],
    cited_chunk_ids: set[int],
) -> list[DebugRetrievedChunkItem]:
    return [
        DebugRetrievedChunkItem(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            filename=item.filename,
            chunk_index=item.chunk_index,
            snippet=_build_snippet(item.content, max_length=220),
            score=item.score,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            whether_cited=item.chunk_id in cited_chunk_ids,
        )
        for item in retrieved_chunks
    ]


def _finalize_response(
    db,
    context: AskPreparedContext,
    assistant_message: str,
    citations: list[ChatCitationItem],
    final_context_preview: str | None,
    llm_ms: int,
    knowledge_base_id: int,
) -> ChatAskResponse:
    save_message(
        db=db,
        conversation=context.conversation,
        role="assistant",
        content=assistant_message,
        citations_json=[citation.model_dump() for citation in citations] or None,
    )

    cited_chunk_ids = {
        citation.chunk_id for citation in citations if citation.chunk_id is not None
    }
    debug_info = None
    if context.debug:
        debug_info = DebugInfo(
            question=context.normalized_question,
            knowledge_base_id=knowledge_base_id,
            top_k=context.top_k,
            top1_score=context.top1_score,
            threshold=RELEVANCE_SCORE_THRESHOLD,
            decision=context.decision,
            retrieval_ms=context.retrieval_ms,
            llm_ms=llm_ms,
            total_ms=_elapsed_ms(context.total_started_at),
            embedding_ms=(
                context.retrieval_trace.embedding_ms
                if context.retrieval_trace is not None
                else None
            ),
            final_context_preview=final_context_preview,
            retrieved_chunks=_build_detailed_debug_items(
                context.retrieved_chunks,
                cited_chunk_ids,
            ),
        )

    return ChatAskResponse(
        conversation_id=context.conversation.id,
        answer=assistant_message,
        citations=citations,
        retrieved_chunks=_build_debug_items(context.retrieved_chunks) if context.debug else None,
        debug=debug_info,
    )


def _rewrite_question_with_history(recent_messages: list[object], question: str) -> str:
    if not recent_messages:
        return question
    return rewrite_question(recent_messages, question)


def _build_recent_turn_summary(recent_messages: list[object]) -> str | None:
    if not recent_messages:
        return None

    messages = recent_messages[-RECENT_TURN_SUMMARY_MESSAGES:]
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "role", "")
        if role not in {"user", "assistant"}:
            continue

        content = getattr(message, "content", "")
        normalized_content = _build_snippet(content, max_length=160)
        prefix = "用户" if role == "user" else "助手"
        lines.append(f"- {prefix}: {normalized_content}")

    return "\n".join(lines) or None


def _build_final_context_preview(
    standalone_question: str,
    recent_turn_summary: str | None,
    context_blocks: list[str],
    max_length: int = FINAL_CONTEXT_PREVIEW_MAX_LENGTH,
) -> str | None:
    if not context_blocks:
        return None

    sections = [f"standalone_question:\n{standalone_question}"]
    if recent_turn_summary:
        sections.append(f"recent_turn_summary:\n{recent_turn_summary}")
    sections.append("\n\n".join(context_blocks).strip())

    preview = "\n\n".join(sections).strip()
    if len(preview) <= max_length:
        return preview
    return preview[:max_length].rstrip() + "..."


def _build_snippet(content: str, max_length: int = 120) -> str:
    normalized_content = " ".join(content.split())
    if len(normalized_content) <= max_length:
        return normalized_content
    return normalized_content[:max_length] + "..."


def _elapsed_ms(started_at: float) -> int:
    return int(round((perf_counter() - started_at) * 1000))
