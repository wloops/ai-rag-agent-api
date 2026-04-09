#!/usr/bin/env bash

set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BACKEND_DIR}/.." && pwd)"

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

MIRRORS=()
if [[ -n "${UV_INDEX_URL:-}" ]]; then
  MIRRORS+=("${UV_INDEX_URL}")
fi
MIRRORS+=(
  "https://mirrors.aliyun.com/pypi/simple"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.ustc.edu.cn/pypi/web/simple"
)

declare -A SEEN=()
UNIQUE_MIRRORS=()
for mirror in "${MIRRORS[@]}"; do
  if [[ -n "${mirror}" && -z "${SEEN["$mirror"]+x}" ]]; then
    UNIQUE_MIRRORS+=("${mirror}")
    SEEN["$mirror"]=1
  fi
done

echo "[1/4] 更新后端代码"
git -C "${BACKEND_DIR}" pull --ff-only

build_success=0
for mirror in "${UNIQUE_MIRRORS[@]}"; do
  echo "[2/4] 尝试构建 backend / worker"
  echo "  UV_INDEX_URL=${mirror}"
  echo "  UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}"
  echo "  UV_HTTP_RETRIES=${UV_HTTP_RETRIES}"

  if (
    cd "${ROOT_DIR}" && \
    docker compose build \
      --build-arg UV_INDEX_URL="${mirror}" \
      --build-arg UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT}" \
      --build-arg UV_HTTP_RETRIES="${UV_HTTP_RETRIES}" \
      backend worker
  ); then
    build_success=1
    export UV_INDEX_URL="${mirror}"
    break
  fi

  echo "  当前镜像源构建失败，继续尝试下一个镜像源。"
done

if [[ "${build_success}" -ne 1 ]]; then
  echo "[ERROR] 所有镜像源都构建失败，请改用本地构建镜像或镜像仓库发布。"
  exit 1
fi

echo "[3/4] 重建并启动 backend / worker"
(
  cd "${ROOT_DIR}" && \
  docker compose up -d --force-recreate backend worker
)

echo "[4/4] 当前服务状态"
(
  cd "${ROOT_DIR}" && \
  docker compose ps backend worker
)

echo
echo "如需观察日志，可执行："
echo "cd ${ROOT_DIR} && docker compose logs -f backend worker"
