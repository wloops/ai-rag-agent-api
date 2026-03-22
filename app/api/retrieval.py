from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.retrieval import RetrievalSearchItem, RetrievalSearchRequest
from app.services.retrieval import search_chunks


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search", response_model=list[RetrievalSearchItem])
def retrieval_search(
    data: RetrievalSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 路由层只负责接参数和鉴权，真正的 embedding 与检索逻辑都放到 service 里。
    return search_chunks(
        db=db,
        current_user_id=current_user.id,
        knowledge_base_id=data.knowledge_base_id,
        query=data.query,
        top_k=data.top_k,
    )
