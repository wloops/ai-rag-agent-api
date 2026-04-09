from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from app.core.config import settings


def can_rerank() -> bool:
    return bool(
        settings.rerank_enabled
        and settings.rerank_api_key
        and settings.rerank_base_url
        and settings.rerank_model
    )


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    if not can_rerank():
        raise RuntimeError("Rerank is not configured")

    if not documents:
        return []

    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    endpoint = settings.rerank_base_url.rstrip("/") + "/rerank"
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.rerank_api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(http_request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"Rerank request failed: {exc}") from exc

    return _parse_rerank_response(json.loads(response_body))


def _parse_rerank_response(payload: dict[str, Any]) -> list[tuple[int, float]]:
    raw_results = payload.get("results") or payload.get("data") or []
    parsed_results: list[tuple[int, float]] = []
    for raw_result in raw_results:
        index = raw_result.get("index")
        score = raw_result.get("relevance_score", raw_result.get("score"))
        if not isinstance(index, int):
            continue
        if not isinstance(score, (int, float)):
            continue
        parsed_results.append((index, float(score)))
    return parsed_results
