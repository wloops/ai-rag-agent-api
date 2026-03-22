import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.kb import router as kb_router
from app.api.retrieval import router as retrieval_router
from app.core.database import Base, engine

# 确保模型已导入，这样 create_all 才知道需要创建哪些表。
from app.models.chunk import Chunk  # noqa: F401
from app.models.document import Document  # noqa: F401
from app.models.knowledge_base import KnowledgeBase  # noqa: F401
from app.models.user import User  # noqa: F401


# FastAPI 应用入口。这里负责组装中间件、路由和启动逻辑。
app = FastAPI(title="AI Knowledge Base API")
logger = logging.getLogger(__name__)


# CORS 用来决定浏览器前端是否允许跨域访问这个后端。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # 项目启动时尝试创建表。
    # 对新手来说可以先这样快速跑通；正式项目通常会改成 Alembic 迁移。
    try:
        if engine.dialect.name == "postgresql":
            # pgvector 是后续向量检索的基础扩展，这里在建表前确保数据库具备该能力。
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError:
        logger.exception("Database initialization failed during startup")


@app.get("/health")
def health():
    # 最简单的健康检查接口，常用于确认服务进程是否活着。
    return {"status": "ok"}


# 把不同业务模块的路由挂到主应用上。
app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(documents_router)
app.include_router(retrieval_router)
