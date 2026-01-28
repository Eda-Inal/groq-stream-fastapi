from sqlalchemy import String, Text, DateTime, func, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ChatPairwiseEvaluation(Base):
    __tablename__ = "chat_pairwise_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "chat_log_id",
            "rubric_version",
            "candidate_model_a",
            "candidate_model_b",
            name="uq_chat_pairwise_log_rubric_pair",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    chat_log_id: Mapped[int] = mapped_column(
        ForeignKey("chat_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rubric_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")

    # Candidates (the models being compared)
    candidate_model_a: Mapped[str] = mapped_column(String(100), nullable=False)
    candidate_model_b: Mapped[str] = mapped_column(String(100), nullable=False)

    # The actual generated answers for auditability
    answer_a: Mapped[str] = mapped_column(Text, nullable=False)
    answer_b: Mapped[str] = mapped_column(Text, nullable=False)

    # Judge config + result
    judge_model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    winner: Mapped[str] = mapped_column(String(20), nullable=False) # "A" / "B" / "Tie"
    notes: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chat_log = relationship("ChatLog", lazy="joined")
