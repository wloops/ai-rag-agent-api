FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=5

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY main.py ./main.py

RUN mkdir -p uploads

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
