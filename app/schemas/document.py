from datetime import datetime

from pydantic import BaseModel


class DocumentBaseResponse(BaseModel):
    id: int
    knowledge_base_id: int
    filename: str
    file_type: str
    storage_path: str
    status: str
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(DocumentBaseResponse):
    pass


class DocumentListResponse(DocumentBaseResponse):
    pass


class DocumentDetailResponse(DocumentBaseResponse):
    pass
