# Backend

后端基于 FastAPI，使用 `uv` 管理依赖，默认通过 PostgreSQL + pgvector 存储数据。

## 环境变量

先复制模板：

```bash
cp .env.example .env
```

PowerShell 可用：

```powershell
Copy-Item .env.example .env
```

`.env` 同时给两类场景提供配置：

- 应用本地运行时使用 `DATABASE_URL`
- 当前目录下的 `docker-compose.yml` 使用 `POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`

`DATABASE_URL` 的本地默认示例指向宿主机 `localhost:5432`。如果通过当前目录的 `docker compose` 启动后端容器，Compose 会自动把它覆盖为容器内的 `db:5432`。

## 仅启动数据库

在当前目录执行：

```bash
docker compose up -d db
```

数据库就绪后，后端启动时会自动：

- 创建表
- 执行 `CREATE EXTENSION IF NOT EXISTS vector`
- 补齐当前项目需要的 PostgreSQL 兼容索引/字段

## 本地运行

安装依赖：

```bash
uv sync
```

启动服务：

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

## Docker 部署

### 仅构建后端镜像

构建镜像：

```bash
docker build -t ai-rag-agent-backend .
```

### 单独运行后端容器

运行容器：

```bash
docker run --rm ^
  -p 8000:8000 ^
  --env-file .env ^
  -e DATABASE_URL=postgresql://ai_rag_agent:change_me_123@host.docker.internal:5432/ai_rag_agent ^
  -v ai-rag-agent-backend-uploads:/app/uploads ^
  ai-rag-agent-backend
```

### 使用 Compose 启动数据库 + 后端

```bash
docker compose up --build
```

访问地址：

- API: `http://localhost:8000`
- Health: `http://localhost:8000/health`

说明：

- 如果你只想要数据库，执行 `docker compose up -d db`
- 如果你希望数据库和后端一起启动，执行 `docker compose up --build`
- `host.docker.internal` 适用于 Docker Desktop 环境；如果你在 Linux 上运行独立容器，请改成宿主机实际可访问地址

## 与本地运行的差异

- 本地运行时，`backend/.env` 中的 `DATABASE_URL` 应指向 `localhost:5432`
- Compose 启动时，数据库连接会自动切换到 `db:5432`
- Compose 会额外挂载 `/app/uploads` 卷，避免容器重建后上传文件丢失
- 前端不在这里编排，前端请单独参考 [frontend/README.md](/G:/@restflux.com/workspace/Open/ai-rag-agent/frontend/README.md)
