from sqlalchemy import String, Text, DateTime, func, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ChatEvaluation(Base):
    __tablename__ = "chat_evaluations"
    __table_args__ = (
        # One evaluation per chat_log per rubric version (supports re-evals later)
        UniqueConstraint("chat_log_id", "rubric_version", name="uq_chat_eval_log_rubric"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # Link back to the generated chat
    chat_log_id: Mapped[int] = mapped_column(
        ForeignKey("chat_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Metadata about the judge run
    judge_model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")

    # Scores (1–5)
    relevance: Mapped[int] = mapped_column(Integer, nullable=False)
    completeness: Mapped[int] = mapped_column(Integer, nullable=False)
    clarity: Mapped[int] = mapped_column(Integer, nullable=False)
    overall_score: Mapped[int] = mapped_column(Integer, nullable=False)

    # Short notes from the judge (1–2 sentences)
    notes: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Optional relationship (not required, but nice for debugging)
    chat_log = relationship("ChatLog", lazy="joined")
