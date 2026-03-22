from app.core.config import settings


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    if not settings.llm_api_key:
        raise RuntimeError("LLM API key is not configured")

    # LLM 调用统一封装在这里，避免 service 和路由层直接依赖具体 SDK。
    # 这样后续切换模型、供应商或补充重试逻辑时，只需要改这一层。
    client = _create_llm_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("LLM returned no choices")

    content = choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM returned empty content")

    return content.strip()


def _create_llm_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI dependency is not installed") from exc

    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
