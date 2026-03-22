from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.kb import KnowledgeBaseCreate, KnowledgeBaseResponse


# 这个 router 处理知识库的新增和查询。
router = APIRouter(prefix="/api/kb", tags=["knowledge_base"])


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # current_user 来自鉴权依赖，所以这里只会为已登录用户创建知识库。
    existing_kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.user_id == current_user.id,
            KnowledgeBase.name == data.name,
        )
        .first()
    )
    if existing_kb:
        raise HTTPException(status_code=400, detail="Knowledge base name already exists")

    kb = KnowledgeBase(
        user_id=current_user.id,
        name=data.name,
        description=data.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 只查当前登录用户自己的知识库数据。
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == current_user.id).all()
    return kbs
