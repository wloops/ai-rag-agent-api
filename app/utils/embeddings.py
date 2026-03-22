from app.core.config import settings


def embed_text(text: str) -> list[float]:
    embeddings = embed_texts([text])
    return embeddings[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    for text in texts:
        if not text or not text.strip():
            raise ValueError("Embedding text cannot be empty")

    # embedding 请求统一封装在这里，避免业务层直接耦合具体 SDK 和接口细节。
    # 这样后续切换 provider、统一做重试或打点时，只需要改这一层。
    client = _create_embedding_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )

    vectors = [item.embedding for item in response.data]
    for vector in vectors:
        _validate_embedding_dimensions(vector)

    return vectors


def _create_embedding_client():
    if not settings.embedding_api_key:
        raise RuntimeError("Embedding API key is not configured")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI dependency is not installed") from exc

    return OpenAI(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def _validate_embedding_dimensions(vector: list[float]) -> None:
    if len(vector) != settings.embedding_dimensions:
        raise RuntimeError(
            f"Unexpected embedding dimensions: expected {settings.embedding_dimensions}, got {len(vector)}"
        )
