from typing import Literal

from pydantic import BaseModel, Field


DocumentType = Literal["pdf", "text", "json", "code"]


class DocumentIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1, max_length=512)
    source: str | None = None
    document_type: DocumentType = "text"
    tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    section_heading: str | None = None


class DocumentReprocessRequest(BaseModel):
    text: str | None = None
    section_heading: str | None = None


class DocumentUpdateRequest(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=512)
    source: str | None = None
    document_type: DocumentType | None = None
    tags: list[str] | None = None
    user_id: str | None = None


class DocumentRead(BaseModel):
    id: int
    filename: str
    source: str | None
    document_type: str
    tags: list[str] | None
    user_id: str | None
    embedding_model_name: str | None
    chunk_count: int | None
    created_at: str


class DocumentIngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    chunks_skipped: int
    tokens_processed: int
    elapsed_ms: int
    embedding_model: str


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentRead]
