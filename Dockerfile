FROM python:3.11-slim

ARG UV_HTTP_TIMEOUT=300
ARG UV_HTTP_RETRIES=10
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} \
    UV_HTTP_RETRIES=${UV_HTTP_RETRIES} \
    UV_INDEX_URL=${UV_INDEX_URL}

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
# 依赖层单独缓存，避免服务器重复跨网下载 Python 包。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY main.py ./main.py

RUN mkdir -p uploads

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
