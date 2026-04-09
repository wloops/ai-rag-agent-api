# Backend 运行手册

后端基于 `FastAPI + SQLAlchemy + PostgreSQL + pgvector + Redis + Celery + LangGraph`，负责认证、知识库管理、文档上传、异步建库、检索问答、多轮会话和轻量 Agent 任务入口。

## 目录说明

- `app/api`: HTTP 接口层
- `app/services`: 业务逻辑
- `app/tasks`: Celery 异步任务
- `app/utils`: LLM、Embedding、Rerank、文件解析、切片等底层封装
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
- `RERANK_ENABLED`: 是否启用 Rerank
- `RERANK_API_KEY` / `RERANK_BASE_URL` / `RERANK_MODEL`: Rerank 服务配置

如果 `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` 为空，代码会回退到 `REDIS_URL`。

## 本地开发启动

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

如果要启动完整项目，优先使用项目根目录统一编排：

```bash
cd ..
docker compose up -d --build
```

当前根目录编排包含：

- `db`: PostgreSQL 16 + pgvector
- `redis`: Redis
- `backend`: FastAPI API 服务
- `worker`: Celery Worker
- `frontend`: Next.js 前端

如果只想启动基础依赖：

```bash
docker compose up -d db redis
```

接口地址：

- API：`http://localhost:8000`
- Health：`http://localhost:8000/health`

## 当前后端能力

- JWT 注册、登录、当前用户查询
- 知识库创建、列表、详情、软删除
- 文档上传与状态查询
- 文档异步处理：保存文件、解析、清洗、切片、Embedding、Chunk 入库
- Dense 语义召回、BM25 文本召回、RRF 融合、Rerank 精排
- LangGraph 编排的 RAG 问答与引用返回
- 多轮会话：最近消息参与问题改写和回答 Prompt
- SSE 流式问答
- Chunk 原文预览
- Agent 任务入口：知识库问答、知识库总结、最新文档汇总、面试材料生成
- 失败文档手动重试

## 检索链路说明

当前后端不是“单次向量检索 + 直接回答”的简单 RAG，而是：

```text
rewrite_question
  -> retrieve_dense_candidates
  -> retrieve_bm25_candidates
  -> fuse_candidates
  -> rerank_candidates
  -> relevance_guard
  -> generate_answer / stream_answer
```

说明：

- Dense 负责语义召回
- BM25 负责关键词召回
- Fusion 使用 RRF 融合双路候选
- Rerank 对最终候选重排
- relevance guard 负责保守拒答

Chat 与 Agent 共用同一 retrieval pipeline。

## 异步文档处理链路

当前上传接口不是同步建库，而是：

1. 接口校验文件类型与知识库归属
2. 保存原始文件到 `uploads/`
3. 创建 `documents` 记录
4. 将状态推进到 `processing`
5. 投递 Celery 任务
6. Worker 后台完成解析、切片、Embedding、Chunk 入库与 BM25 索引维护
7. 最终写回 `success` 或 `failed`

状态含义：

- `pending`: 记录已创建，尚未开始执行
- `processing`: 已入队或正在由 worker 处理
- `success`: 建库完成
- `failed`: 处理失败，可查看 `error_message`

## 当前项目定位

当前后端更适合表述为：

> `LangGraph 编排的 Hybrid / Workflow RAG`

它已经具备多阶段检索、重排和条件决策节点，但还没有形成循环式 Router / Reasoning 闭环，因此不建议直接宣传为“标准 Agentic RAG”。

## 校验命令

开发联调完成后，建议至少执行：

后端测试：

```bash
uv run python -m unittest discover -s tests
```

如果是整套联调，再补充：

- `curl http://localhost:8000/health`
- 根目录执行 `docker compose logs -f backend worker`

## 常见问题

### 1. 上传后文档一直是 `processing`

- 先确认 Redis 是否启动
- 再确认 Celery worker 是否启动
- 检查 `LLM_*` / `EMBEDDING_*` / `RERANK_*` 配置是否有效

### 2. `docker compose up -d --build` 后无法处理文档

- 确认 `worker` 服务已正常拉起
- 确认 `backend/.env` 中 API key 可在容器内使用
- 如果使用项目根目录编排，确认根目录 `.env` 的数据库配置没有和 `backend/.env` 的预期冲突

### 3. 为什么当前不算严格意义上的 Agentic RAG

当前流程虽然使用了 LangGraph 和条件分支，但主问答链路仍以一次性决策为主，没有 Planner、工具路由后的再判断、自我反思补证据或循环重试闭环。
