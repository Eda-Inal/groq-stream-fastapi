from sqlalchemy import String, Text, DateTime, func, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(primary_key=True)

    # LLM input (for quick inspection / search)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Full OpenAI-compatible messages payload (source of truth)
    messages: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # LLM output (final response after streaming)
    response: Mapped[str] = mapped_column(Text, nullable=False)

    # Model metadata
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # LLM parameters
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    presence_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
