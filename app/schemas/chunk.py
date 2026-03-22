from datetime import datetime

from pydantic import BaseModel


class ChunkListResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    metadata_json: dict[str, int]
    created_at: datetime

    model_config = {"from_attributes": True}
