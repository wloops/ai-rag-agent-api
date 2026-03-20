from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


# Engine 可以理解成“数据库连接工厂”。
# 后续所有 Session 都会基于它创建。
engine = create_engine(settings.database_url, echo=True)

# SessionLocal 是每次请求使用的数据库会话工厂。
# autocommit=False: 需要手动 commit
# autoflush=False: 不在每次查询前自动把修改刷到数据库
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 所有 ORM 模型都会继承这个 Base。
class Base(DeclarativeBase):
    pass


def get_db():
    # FastAPI 的依赖注入函数。
    # 每次请求进来时创建一个 Session，请求结束后自动关闭。
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
