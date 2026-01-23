from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(primary_key=True)

    # LLM input
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # LLM output
    response: Mapped[str] = mapped_column(Text, nullable=False)

    # Model metadata
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Timestamps
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
