from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.agent import AgentRunRequest, AgentRunResponse
from app.services.agent_service import run_agent_task


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
def run_agent(
    data: AgentRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return run_agent_task(
            db=db,
            current_user_id=current_user.id,
            knowledge_base_id=data.knowledge_base_id,
            task_type=data.task_type,
            query=data.query,
            top_k=data.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
