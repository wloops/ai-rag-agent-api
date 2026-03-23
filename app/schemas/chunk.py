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


class ChunkPreviewResponse(BaseModel):
    document_id: int
    chunk_id: int
    filename: str
    chunk_index: int
    start_offset: int
    end_offset: int
    preview_text: str
    highlight_start_offset: int
    highlight_end_offset: int
