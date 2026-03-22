from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# SQLAlchemy ORM 模型：一行 users 表记录会映射成一个 User 对象。
class User(Base):
    __tablename__ = "users"

    # mapped_column 定义数据库字段；Mapped[...] 定义 Python 侧的类型。
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # relationship 不会直接在表里生成字段，它描述的是表之间的对象关系。
    knowledge_bases = relationship("KnowledgeBase", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
