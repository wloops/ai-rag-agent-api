# Backend 运行手册

后端基于 `FastAPI + SQLAlchemy + PostgreSQL + pgvector + Redis + Celery`，负责认证、知识库管理、文档上传、异步建库、检索问答和多轮会话。

## 目录说明

- `app/api`: HTTP 接口层
- `app/services`: 业务逻辑
- `app/tasks`: Celery 异步任务
- `app/utils`: LLM、Embedding、文件解析、切片等底层封装
- `tests`: 后端单元测试

## 环境变量

先复制模板：

```powershell
Copy-Item .env.example .env
```

如果从项目根目录使用统一 Docker Compose 部署，只有在需要自定义端口或数据库容器参数时，才需要额外复制：

```powershell
Copy-Item ../.env.example ../.env
```

核心变量：

- `DATABASE_URL`: 本地直连 PostgreSQL 时使用
- `REDIS_URL`: Redis 连接地址
- `CELERY_BROKER_URL`: Celery broker，默认可与 Redis 共用
- `CELERY_RESULT_BACKEND`: Celery 结果后端，默认可与 Redis 共用
- `EMBEDDING_*`: Embedding 模型配置
- `LLM_*`: 大模型配置

如果 `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` 为空，代码会回退到 `REDIS_URL`。

## 本地开发

安装依赖：

```bash
uv sync
```

启动 API：

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动 Celery Worker：

```bash
uv run celery -A app.core.celery_app.celery_app worker --loglevel=info
```

健康检查：

```bash
curl http://localhost:8000/health
```

## Docker Compose

后端目录下的 `docker-compose.yml` 适合只启动后端相关服务：

当前本地编排为：

- `db`: PostgreSQL + pgvector
- `redis`: Redis
- `backend`: FastAPI API 服务
- `worker`: Celery Worker

启动全部服务：

```bash
docker compose up --build
```

如果要连前端一起启动，请回到项目根目录执行：

```bash
cd ..
docker compose up -d --build
```

只启动基础依赖：

```bash
docker compose up -d db redis
```

接口地址：

- API: `http://localhost:8000`
- Health: `http://localhost:8000/health`

## 当前后端能力

- JWT 注册、登录、当前用户查询
- 知识库创建、列表、详情、软删除
- 文档上传与状态查询
- 文档异步处理：保存文件、解析、清洗、切片、Embedding、Chunk 入库
- pgvector 相似度检索
- RAG 问答与引用返回
- 多轮会话：最近消息参与问题改写和回答 Prompt
- 会话历史与消息列表
- Chunk 原文预览

## 异步文档处理链路

当前上传接口不是同步建库，而是：

1. 接口校验文件类型与知识库归属
2. 保存原始文件到 `uploads/`
3. 创建 `documents` 记录
4. 将状态推进到 `processing`
5. 投递 Celery 任务
6. Worker 后台完成解析、切片、Embedding、Chunk 入库
7. 最终写回 `success` 或 `failed`

状态含义：

- `pending`: 记录已创建，尚未开始执行
- `processing`: 已入队或正在由 worker 处理
- `success`: 建库完成
- `failed`: 处理失败，可查看 `error_message`

## 多轮会话记忆

当前实现不是把 Redis 当聊天记忆，而是：

- 会话和消息持久化在 PostgreSQL
- 继续追问时读取最近 6 条消息
- 先把当前问题改写成独立问题，再做向量检索
- 把最近 2 轮摘要一起带入最终回答 Prompt

这样做的目的，是让多轮问答既能保留上下文，又不会让检索直接吃到模糊指代。

## 测试

运行后端测试：

```bash
uv run python -m unittest discover -s tests
```

当前测试覆盖：

- API 路由
- 文档上传与预览
- 异步任务状态流转
- 文本清洗与切片
- Embedding / LLM 封装
- Retrieval / RAG / 会话服务

## 常见问题

### 1. 上传后文档一直是 `processing`

- 先确认 Redis 是否启动
- 再确认 Celery worker 是否启动
- 检查 `LLM_*` / `EMBEDDING_*` 配置是否有效

### 2. `docker compose up --build` 后无法处理文档

- 确认 `worker` 服务已正常拉起
- 确认 `backend/.env` 中 API key 可在容器内使用
- 如果使用项目根目录编排，确认根目录 `.env` 的数据库配置没有和 `backend/.env` 的预期冲突

### 3. 为什么没有先接 LangChain

当前项目优先保证主链路、异步工程化和可解释性。现阶段直接封装 Embedding、Retrieval、LLM Client 更轻量，也更方便面试时讲清楚数据流和控制点。
