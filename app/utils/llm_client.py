from collections.abc import Iterator, Sequence

from app.core.config import settings


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    if not settings.llm_api_key:
        raise RuntimeError("LLM API key is not configured")

    client = _create_llm_client()
    response = client.chat.completions.create(
        **_build_chat_completion_params(system_prompt, user_prompt)
    )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("LLM returned no choices")

    content = choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM returned empty content")

    return content.strip()


def stream_answer(system_prompt: str, user_prompt: str) -> Iterator[str]:
    if not settings.llm_api_key:
        raise RuntimeError("LLM API key is not configured")

    client = _create_llm_client()
    stream = client.chat.completions.create(
        **_build_chat_completion_params(system_prompt, user_prompt),
        stream=True,
    )

    yielded = False
    for chunk in stream:
        delta_text = _extract_delta_content(chunk)
        if not delta_text:
            continue

        yielded = True
        yield delta_text

    if not yielded:
        raise RuntimeError("LLM returned empty content")


def rewrite_question(messages: Sequence[object], question: str) -> str:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question cannot be blank")

    history_lines = _serialize_messages(messages)
    if not history_lines:
        return normalized_question

    try:
        rewritten = generate_answer(
            system_prompt=(
                "You rewrite follow-up questions into standalone questions. "
                "Keep the original language. "
                "Do not answer the question. "
                "Return only the rewritten standalone question."
            ),
            user_prompt=(
                "Conversation history:\n"
                f"{history_lines}\n\n"
                f"Follow-up question:\n{normalized_question}\n\n"
                "Rewrite the follow-up question as a standalone question."
            ),
        )
    except RuntimeError:
        return normalized_question

    return _normalize_rewritten_question(rewritten, normalized_question)


def _serialize_messages(messages: Sequence[object]) -> str:
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", "")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _normalize_rewritten_question(rewritten: str, fallback: str) -> str:
    normalized = rewritten.strip()
    if not normalized:
        return fallback

    if "\n" in normalized:
        normalized = normalized.splitlines()[0].strip()

    prefixes = (
        "standalone question:",
        "rewritten question:",
        "question:",
        "独立问题：",
        "独立问题:",
        "改写后问题：",
        "改写后问题:",
    )
    lower_normalized = normalized.lower()
    for prefix in prefixes:
        if lower_normalized.startswith(prefix.lower()):
            normalized = normalized[len(prefix) :].strip()
            break

    return normalized or fallback


def _build_chat_completion_params(system_prompt: str, user_prompt: str) -> dict:
    return {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }


def _extract_delta_content(chunk: object) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""

    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""

    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    segments: list[str] = []
    for item in content:
        if isinstance(item, str):
            segments.append(item)
            continue

        text = getattr(item, "text", None)
        if isinstance(text, str):
            segments.append(text)

    return "".join(segments)


def _create_llm_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI dependency is not installed") from exc

    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
