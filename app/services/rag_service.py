import re

from fastapi import HTTPException

from app.schemas.chat import (
    ChatAskResponse,
    ChatCitationItem,
    RetrievedChunkDebugItem,
)
from app.schemas.retrieval import RetrievalSearchItem
from app.services.conversation_service import (
    get_owned_knowledge_base,
    resolve_conversation_for_question,
    save_message,
)
from app.services.retrieval import search_chunks
from app.utils.llm_client import generate_answer

RELEVANCE_SCORE_THRESHOLD = 0.35
NO_ANSWER_MESSAGE = "当前知识库中未找到足够相关内容。"
SOURCE_ID_PATTERN = re.compile(r"\[(S\d+)\]")


def ask_knowledge_base(
    db,
    current_user_id: int,
    knowledge_base_id: int,
    question: str,
    top_k: int = 3,
    debug: bool = False,
    conversation_id: int | None = None,
) -> ChatAskResponse:
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

    # 先保存用户问题，这样即使后续检索结果为空，也能恢复完整会话历史。
    save_message(
        db=db,
        conversation=conversation,
        role="user",
        content=normalized_question,
    )

    retrieved_chunks = search_chunks(
        db=db,
        current_user_id=current_user_id,
        knowledge_base_id=knowledge_base_id,
        query=normalized_question,
        top_k=top_k,
    )

    if not retrieved_chunks or retrieved_chunks[0].score < RELEVANCE_SCORE_THRESHOLD:
        assistant_message = NO_ANSWER_MESSAGE
        citations: list[ChatCitationItem] = []
    else:
        source_mapping = _build_source_mapping(retrieved_chunks)
        assistant_message = generate_answer(
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(normalized_question, source_mapping),
        )

        cited_source_ids = _parse_cited_source_ids(assistant_message)
        citations = _build_citations(source_mapping, cited_source_ids)
        if not citations:
            citations = _build_fallback_citations(retrieved_chunks)

    save_message(
        db=db,
        conversation=conversation,
        role="assistant",
        content=assistant_message,
        citations_json=[citation.model_dump() for citation in citations] or None,
    )

    return ChatAskResponse(
        conversation_id=conversation.id,
        answer=assistant_message,
        citations=citations,
        retrieved_chunks=_build_debug_items(retrieved_chunks) if debug else None,
    )


def _build_system_prompt() -> str:
    return (
        "你是知识库问答助手。"
        "你只能依据提供的上下文回答，不能使用上下文之外的知识补全。"
        "如果上下文没有足够答案，必须明确回答“当前知识库中未找到足够相关内容”或“不知道”。"
        "请尽量结构化回答。"
        "引用依据时必须使用给定的 source id，例如 [S1]、[S2]。"
    )


def _build_user_prompt(question: str, source_mapping: dict[str, RetrievalSearchItem]) -> str:
    context_blocks = []
    for source_id, item in source_mapping.items():
        context_blocks.append(
            "\n".join(
                [
                    f"{source_id}",
                    f"document_id: {item.document_id}",
                    f"filename: {item.filename}",
                    f"chunk_index: {item.chunk_index}",
                    f"content: {item.content}",
                ]
            )
        )

    # Prompt 里显式限制模型只能依据检索到的上下文回答，避免模型脱离知识库自由发挥。
    return (
        f"用户问题：{question}\n\n"
        "请基于以下上下文回答：\n\n"
        + "\n\n".join(context_blocks)
    )


def _build_source_mapping(retrieved_chunks: list[RetrievalSearchItem]) -> dict[str, RetrievalSearchItem]:
    return {f"S{index + 1}": item for index, item in enumerate(retrieved_chunks)}


def _parse_cited_source_ids(answer: str) -> list[str]:
    matched = SOURCE_ID_PATTERN.findall(answer)
    unique_source_ids: list[str] = []
    for source_id in matched:
        if source_id not in unique_source_ids:
            unique_source_ids.append(source_id)

    return unique_source_ids


def _build_citations(
    source_mapping: dict[str, RetrievalSearchItem], source_ids: list[str]
) -> list[ChatCitationItem]:
    citations: list[ChatCitationItem] = []
    for source_id in source_ids:
        item = source_mapping.get(source_id)
        if item is None:
            continue

        citations.append(
            ChatCitationItem(
                document_id=item.document_id,
                filename=item.filename,
                chunk_index=item.chunk_index,
                snippet=_build_snippet(item.content),
            )
        )

    return citations


def _build_fallback_citations(retrieved_chunks: list[RetrievalSearchItem]) -> list[ChatCitationItem]:
    return [
        ChatCitationItem(
            document_id=item.document_id,
            filename=item.filename,
            chunk_index=item.chunk_index,
            snippet=_build_snippet(item.content),
        )
        for item in retrieved_chunks[:3]
    ]


def _build_debug_items(retrieved_chunks: list[RetrievalSearchItem]) -> list[RetrievedChunkDebugItem]:
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


def _build_snippet(content: str, max_length: int = 120) -> str:
    normalized_content = " ".join(content.split())
    if len(normalized_content) <= max_length:
        return normalized_content

    return normalized_content[:max_length] + "..."
