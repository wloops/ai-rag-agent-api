import operator
import re
from collections.abc import Generator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from fastapi import HTTPException
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from app.models.conversation import Conversation
from app.schemas.chat import (
    ChatAskResponse,
    ChatCitationItem,
    DebugGraphTraceItem,
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
    graph_trace: list[DebugGraphTraceItem]


class AskGraphState(TypedDict, total=False):
    db: Any
    current_user_id: int
    knowledge_base_id: int
    question: str
    top_k: int
    debug: bool
    conversation_id: int | None
    total_started_at: float
    normalized_question: str
    conversation: Conversation
    recent_messages: list[object]
    standalone_question: str
    recent_turn_summary: str | None
    retrieval_trace: RetrievalTrace | None
    retrieved_chunks: list[RetrievalSearchItem]
    retrieval_ms: int
    top1_score: float | None
    decision: Literal["answer", "reject"]
    assistant_message: str
    llm_ms: int
    source_mapping: dict[str, RetrievalSearchItem]
    final_context_preview: str | None
    citations: list[ChatCitationItem]
    used_fallback_citations: bool
    response: ChatAskResponse
    graph_trace: Annotated[list[DebugGraphTraceItem], operator.add]


def ask_knowledge_base(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int = 3,
    debug: bool = False,
    conversation_id: int | None = None,
) -> ChatAskResponse:
    result = CHAT_ASK_GRAPH.invoke(
        _build_initial_graph_state(
            db=db,
            current_user_id=current_user_id,
            knowledge_base_id=knowledge_base_id,
            question=question,
            top_k=top_k,
            debug=debug,
            conversation_id=conversation_id,
        )
    )
    response = result.get("response")
    if response is None:
        raise RuntimeError("Chat graph produced no response")
    return response


def stream_knowledge_base_events(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int = 3,
    debug: bool = False,
    conversation_id: int | None = None,
) -> Generator[tuple[str, dict], None, None]:
    state = _build_initial_graph_state(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        question=question,
        top_k=top_k,
        debug=debug,
        conversation_id=conversation_id,
    )
    state = _merge_graph_state(state, _graph_node_validate_request(state))
    state = _merge_graph_state(state, _graph_node_resolve_conversation(state))
    yield "start", {"conversation_id": state["conversation"].id}
    state = _merge_graph_state(state, _graph_node_rewrite_question(state))
    state = _merge_graph_state(state, _graph_node_retrieve_context(state))
    state = _merge_graph_state(state, _graph_node_relevance_guard(state))

    if state["decision"] == "answer":
        started_at = perf_counter()
        source_mapping = _build_source_mapping(state["retrieved_chunks"])
        context_blocks = _build_context_blocks(source_mapping)
        final_context_preview = _build_final_context_preview(
            standalone_question=state["standalone_question"],
            recent_turn_summary=state.get("recent_turn_summary"),
            context_blocks=context_blocks,
        )
        user_prompt = _build_user_prompt(
            original_question=state["normalized_question"],
            standalone_question=state["standalone_question"],
            source_mapping=source_mapping,
            context_blocks=context_blocks,
            recent_turn_summary=state.get("recent_turn_summary"),
        )

        llm_started_at = perf_counter()
        answer_chunks: list[str] = []
        for delta in stream_answer(
            system_prompt=_build_system_prompt(),
            user_prompt=user_prompt,
        ):
            answer_chunks.append(delta)
            yield "delta", {"content": delta}

        assistant_message = "".join(answer_chunks).strip()
        if not assistant_message:
            raise RuntimeError("LLM returned empty content")

        state = _merge_graph_state(
            state,
            _build_node_result(
                node_name="stream_answer",
                started_at=started_at,
                detail=f"Generated answer using {len(source_mapping)} retrieved chunks.",
                updates={
                    "assistant_message": assistant_message,
                    "llm_ms": _elapsed_ms(llm_started_at),
                    "source_mapping": source_mapping,
                    "final_context_preview": final_context_preview,
                },
            ),
        )

    state = _merge_graph_state(state, _graph_node_build_citations(state))
    state = _merge_graph_state(state, _graph_node_finalize_response(state))
    response = state.get("response")
    if response is None:
        raise RuntimeError("Chat stream flow produced no final response")
    yield "final", response.model_dump(mode="json")


def _build_initial_graph_state(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int,
    debug: bool,
    conversation_id: int | None,
) -> AskGraphState:
    return AskGraphState(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        question=question,
        top_k=top_k,
        debug=debug,
        conversation_id=conversation_id,
        total_started_at=perf_counter(),
        graph_trace=[],
    )


def _graph_node_validate_request(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    normalized_question = state["question"].strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank")

    get_owned_knowledge_base(
        state["db"],
        state["knowledge_base_id"],
        state["current_user_id"],
    )
    return _build_node_result(
        node_name="validate_request",
        started_at=started_at,
        detail=(
            f"Validated question length {len(normalized_question)} and confirmed "
            f"knowledge base #{state['knowledge_base_id']} access."
        ),
        updates={"normalized_question": normalized_question},
    )


def _graph_node_resolve_conversation(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    conversation = resolve_conversation_for_question(
        db=state["db"],
        current_user_id=state["current_user_id"],
        knowledge_base_id=state["knowledge_base_id"],
        question=state["normalized_question"],
        conversation_id=state.get("conversation_id"),
    )
    recent_messages = list_recent_conversation_messages(
        db=state["db"],
        conversation_id=conversation.id,
        current_user_id=state["current_user_id"],
        limit=RECENT_MESSAGE_WINDOW,
    )
    save_message(
        db=state["db"],
        conversation=conversation,
        role="user",
        content=state["normalized_question"],
    )
    _write_stream_event("start", {"conversation_id": conversation.id})
    return _build_node_result(
        node_name="resolve_conversation",
        started_at=started_at,
        detail=(
            f"Resolved conversation #{conversation.id} and loaded "
            f"{len(recent_messages)} recent messages."
        ),
        updates={
            "conversation": conversation,
            "recent_messages": recent_messages,
        },
    )


def _graph_node_rewrite_question(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    recent_messages = state.get("recent_messages", [])
    standalone_question = _rewrite_question_with_history(
        recent_messages,
        state["normalized_question"],
    )
    recent_turn_summary = _build_recent_turn_summary(recent_messages)
    used_history = bool(recent_messages)
    return _build_node_result(
        node_name="rewrite_question",
        started_at=started_at,
        detail=(
            "Rewrote the follow-up question using recent history."
            if used_history
            else "No recent history found; reused the normalized question."
        ),
        status="completed" if used_history else "skipped",
        updates={
            "standalone_question": standalone_question,
            "recent_turn_summary": recent_turn_summary,
        },
        trace_overrides={
            "used_history": used_history,
            "rewritten_question": standalone_question,
        },
    )


def _graph_node_retrieve_context(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    retrieval_trace = RetrievalTrace() if state["debug"] else None
    retrieval_started_at = perf_counter()
    retrieved_chunks = search_chunks(
        db=state["db"],
        current_user_id=state["current_user_id"],
        knowledge_base_id=state["knowledge_base_id"],
        query=state["standalone_question"],
        top_k=state["top_k"],
        trace=retrieval_trace,
    )
    retrieval_ms = _elapsed_ms(retrieval_started_at)
    top1_score = retrieved_chunks[0].score if retrieved_chunks else None
    return _build_node_result(
        node_name="retrieve_context",
        started_at=started_at,
        detail=(
            f"Retrieved {len(retrieved_chunks)} chunks with top1 score "
            f"{top1_score:.3f}."
            if top1_score is not None
            else "Retrieved 0 chunks."
        ),
        updates={
            "retrieval_trace": retrieval_trace,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_ms": retrieval_ms,
            "top1_score": top1_score,
        },
        trace_overrides={
            "retrieval_count": len(retrieved_chunks),
            "top1_score": top1_score,
        },
    )


def _graph_node_relevance_guard(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    top1_score = state.get("top1_score")
    decision: Literal["answer", "reject"] = (
        "answer"
        if top1_score is not None and top1_score >= RELEVANCE_SCORE_THRESHOLD
        else "reject"
    )
    updates: AskGraphState = {"decision": decision}
    if decision == "reject":
        updates.update(
            {
                "assistant_message": NO_ANSWER_MESSAGE,
                "llm_ms": 0,
                "source_mapping": {},
                "final_context_preview": None,
            }
        )
    detail = (
        f"Top1 score {top1_score:.3f} passed threshold {RELEVANCE_SCORE_THRESHOLD:.2f}."
        if decision == "answer"
        else (
            f"Top1 score {top1_score:.3f} did not pass threshold "
            f"{RELEVANCE_SCORE_THRESHOLD:.2f}."
            if top1_score is not None
            else "No retrieved chunks passed the relevance guard."
        )
    )
    return _build_node_result(
        node_name="relevance_guard",
        started_at=started_at,
        detail=detail,
        updates=updates,
        trace_overrides={
            "top1_score": top1_score,
            "threshold": RELEVANCE_SCORE_THRESHOLD,
            "decision": decision,
        },
    )


def _graph_node_generate_answer(state: AskGraphState) -> AskGraphState:
    return _generate_answer_update(
        state,
        node_name="generate_answer",
        streamer=None,
    )


def _graph_node_stream_answer(state: AskGraphState) -> AskGraphState:
    return _generate_answer_update(
        state,
        node_name="stream_answer",
        streamer=_stream_llm_answer,
    )


def _graph_node_build_citations(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    if state["decision"] == "reject":
        return _build_node_result(
            node_name="build_citations",
            started_at=started_at,
            detail="Skipped citation building because the request was rejected.",
            status="skipped",
            updates={
                "citations": [],
                "used_fallback_citations": False,
            },
            trace_overrides={
                "cited_count": 0,
                "used_fallback_citations": False,
            },
        )

    source_mapping = state.get("source_mapping") or _build_source_mapping(
        state["retrieved_chunks"]
    )
    citations = _resolve_citations(
        retrieved_chunks=state["retrieved_chunks"],
        source_mapping=source_mapping,
        assistant_message=state["assistant_message"],
    )
    parsed_source_ids = _parse_cited_source_ids(state["assistant_message"])
    used_fallback_citations = len(citations) > 0 and not parsed_source_ids
    return _build_node_result(
        node_name="build_citations",
        started_at=started_at,
        detail=f"Built {len(citations)} citations for the final answer.",
        updates={
            "source_mapping": source_mapping,
            "citations": citations,
            "used_fallback_citations": used_fallback_citations,
        },
        trace_overrides={
            "cited_count": len(citations),
            "used_fallback_citations": used_fallback_citations,
        },
    )


def _graph_node_finalize_response(state: AskGraphState) -> AskGraphState:
    started_at = perf_counter()
    save_message(
        db=state["db"],
        conversation=state["conversation"],
        role="assistant",
        content=state["assistant_message"],
        citations_json=[citation.model_dump() for citation in state.get("citations", [])] or None,
    )
    trace_item = _build_graph_trace_item(
        node_name="finalize_response",
        started_at=started_at,
        detail=(
            f"Persisted assistant message for conversation #{state['conversation'].id} "
            f"with {len(state.get('citations', []))} citations."
        ),
    )
    response = _build_chat_response(
        state=state,
        graph_trace=[*state.get("graph_trace", []), trace_item],
    )
    return {
        "response": response,
        "graph_trace": [trace_item],
    }


def _generate_answer_update(
    state: AskGraphState,
    node_name: str,
    streamer,
) -> AskGraphState:
    started_at = perf_counter()
    source_mapping = _build_source_mapping(state["retrieved_chunks"])
    context_blocks = _build_context_blocks(source_mapping)
    final_context_preview = _build_final_context_preview(
        standalone_question=state["standalone_question"],
        recent_turn_summary=state.get("recent_turn_summary"),
        context_blocks=context_blocks,
    )
    user_prompt = _build_user_prompt(
        original_question=state["normalized_question"],
        standalone_question=state["standalone_question"],
        source_mapping=source_mapping,
        context_blocks=context_blocks,
        recent_turn_summary=state.get("recent_turn_summary"),
    )

    llm_started_at = perf_counter()
    if streamer is None:
        assistant_message = generate_answer(
            system_prompt=_build_system_prompt(),
            user_prompt=user_prompt,
        )
    else:
        assistant_message = streamer(
            system_prompt=_build_system_prompt(),
            user_prompt=user_prompt,
        )
    llm_ms = _elapsed_ms(llm_started_at)
    return _build_node_result(
        node_name=node_name,
        started_at=started_at,
        detail=f"Generated answer using {len(source_mapping)} retrieved chunks.",
        updates={
            "assistant_message": assistant_message,
            "llm_ms": llm_ms,
            "source_mapping": source_mapping,
            "final_context_preview": final_context_preview,
        },
    )


def _stream_llm_answer(system_prompt: str, user_prompt: str) -> str:
    return _stream_llm_answer_with_handler(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        delta_handler=lambda delta: _write_stream_event("delta", {"content": delta}),
    )


def _stream_llm_answer_with_handler(
    system_prompt: str,
    user_prompt: str,
    delta_handler,
) -> str:
    answer_chunks: list[str] = []
    for delta in stream_answer(system_prompt=system_prompt, user_prompt=user_prompt):
        answer_chunks.append(delta)
        delta_handler(delta)

    assistant_message = "".join(answer_chunks).strip()
    if not assistant_message:
        raise RuntimeError("LLM returned empty content")
    return assistant_message


def _build_chat_response(
    state: AskGraphState,
    graph_trace: list[DebugGraphTraceItem],
) -> ChatAskResponse:
    context = _build_context_from_state(state, graph_trace)
    cited_chunk_ids = {
        citation.chunk_id
        for citation in state.get("citations", [])
        if citation.chunk_id is not None
    }

    debug_info = None
    if context.debug:
        debug_info = DebugInfo(
            question=context.normalized_question,
            knowledge_base_id=state["knowledge_base_id"],
            top_k=context.top_k,
            top1_score=context.top1_score,
            threshold=RELEVANCE_SCORE_THRESHOLD,
            decision=context.decision,
            retrieval_ms=context.retrieval_ms,
            llm_ms=state.get("llm_ms", 0),
            total_ms=_elapsed_ms(context.total_started_at),
            embedding_ms=(
                context.retrieval_trace.embedding_ms
                if context.retrieval_trace is not None
                else None
            ),
            final_context_preview=state.get("final_context_preview"),
            retrieved_chunks=_build_detailed_debug_items(
                context.retrieved_chunks,
                cited_chunk_ids,
            ),
            graph_trace=graph_trace,
        )

    return ChatAskResponse(
        conversation_id=context.conversation.id,
        answer=state["assistant_message"],
        citations=state.get("citations", []),
        retrieved_chunks=_build_debug_items(context.retrieved_chunks) if context.debug else None,
        debug=debug_info,
    )


def _build_context_from_state(
    state: AskGraphState,
    graph_trace: list[DebugGraphTraceItem],
) -> AskPreparedContext:
    return AskPreparedContext(
        conversation=state["conversation"],
        normalized_question=state["normalized_question"],
        standalone_question=state["standalone_question"],
        recent_turn_summary=state.get("recent_turn_summary"),
        retrieved_chunks=state["retrieved_chunks"],
        top_k=state["top_k"],
        debug=state["debug"],
        retrieval_trace=state.get("retrieval_trace"),
        retrieval_ms=state.get("retrieval_ms", 0),
        total_started_at=state["total_started_at"],
        top1_score=state.get("top1_score"),
        decision=state["decision"],
        graph_trace=graph_trace,
    )


def _build_node_result(
    node_name: str,
    started_at: float,
    detail: str,
    updates: dict[str, Any] | None = None,
    status: Literal["completed", "skipped"] = "completed",
    trace_overrides: dict[str, Any] | None = None,
) -> AskGraphState:
    result: AskGraphState = {}
    if updates:
        result.update(updates)
    result["graph_trace"] = [
        _build_graph_trace_item(
            node_name=node_name,
            started_at=started_at,
            detail=detail,
            status=status,
            **(trace_overrides or {}),
        )
    ]
    return result


def _build_graph_trace_item(
    node_name: str,
    started_at: float,
    detail: str,
    status: Literal["completed", "skipped"] = "completed",
    used_history: bool | None = None,
    rewritten_question: str | None = None,
    retrieval_count: int | None = None,
    top1_score: float | None = None,
    threshold: float | None = None,
    decision: Literal["answer", "reject"] | None = None,
    cited_count: int | None = None,
    used_fallback_citations: bool | None = None,
) -> DebugGraphTraceItem:
    return DebugGraphTraceItem(
        node=node_name,
        status=status,
        duration_ms=_elapsed_ms(started_at),
        detail=detail,
        used_history=used_history,
        rewritten_question=rewritten_question,
        retrieval_count=retrieval_count,
        top1_score=top1_score,
        threshold=threshold,
        decision=decision,
        cited_count=cited_count,
        used_fallback_citations=used_fallback_citations,
    )


def _write_stream_event(event_name: str, payload: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer({"event": event_name, "payload": payload})


def _merge_graph_state(
    state: AskGraphState,
    updates: AskGraphState,
) -> AskGraphState:
    merged = dict(state)
    for key, value in updates.items():
        if key == "graph_trace":
            merged["graph_trace"] = [*merged.get("graph_trace", []), *value]
            continue
        merged[key] = value
    return merged


def _route_after_relevance_guard(answer_node_name: str):
    def _route(state: AskGraphState) -> str:
        if state["decision"] == "answer":
            return answer_node_name
        return "build_citations"

    return _route


def _build_chat_graph(answer_node_name: str, answer_node):
    builder = StateGraph(AskGraphState)
    builder.add_node("validate_request", _graph_node_validate_request)
    builder.add_node("resolve_conversation", _graph_node_resolve_conversation)
    builder.add_node("rewrite_question", _graph_node_rewrite_question)
    builder.add_node("retrieve_context", _graph_node_retrieve_context)
    builder.add_node("relevance_guard", _graph_node_relevance_guard)
    builder.add_node(answer_node_name, answer_node)
    builder.add_node("build_citations", _graph_node_build_citations)
    builder.add_node("finalize_response", _graph_node_finalize_response)

    builder.add_edge(START, "validate_request")
    builder.add_edge("validate_request", "resolve_conversation")
    builder.add_edge("resolve_conversation", "rewrite_question")
    builder.add_edge("rewrite_question", "retrieve_context")
    builder.add_edge("retrieve_context", "relevance_guard")
    builder.add_conditional_edges(
        "relevance_guard",
        _route_after_relevance_guard(answer_node_name),
        {
            answer_node_name: answer_node_name,
            "build_citations": "build_citations",
        },
    )
    builder.add_edge(answer_node_name, "build_citations")
    builder.add_edge("build_citations", "finalize_response")
    builder.add_edge("finalize_response", END)
    return builder.compile()


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


CHAT_ASK_GRAPH = _build_chat_graph("generate_answer", _graph_node_generate_answer)
