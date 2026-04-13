from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Document(Base):
    """
    Uploaded or ingested document metadata for RAG.

    Chunks and vectors live in DocumentChunk (CASCADE delete).
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    # "pdf" | "text" | "json" | "code"
    document_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'text'"),
    )

    tags: Mapped[list | dict | None] = mapped_column(JSONB, nullable=True)

    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    embedding_model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
