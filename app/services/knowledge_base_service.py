from fastapi import HTTPException

from app.models.knowledge_base import KnowledgeBase


def get_active_owned_knowledge_base_query(db, current_user_id: int):
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == current_user_id,
        KnowledgeBase.deleted_at.is_(None),
    )


def get_active_owned_knowledge_base(db, knowledge_base_id: int, current_user_id: int) -> KnowledgeBase:
    knowledge_base = (
        get_active_owned_knowledge_base_query(db, current_user_id)
        .filter(KnowledgeBase.id == knowledge_base_id)
        .first()
    )
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return knowledge_base


def find_active_knowledge_base_name_conflict(
    db,
    current_user_id: int,
    name: str,
    exclude_id: int | None = None,
) -> KnowledgeBase | None:
    query = get_active_owned_knowledge_base_query(db, current_user_id).filter(
        KnowledgeBase.name == name
    )
    if exclude_id is not None:
        query = query.filter(KnowledgeBase.id != exclude_id)

    return query.first()
