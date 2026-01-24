"""add openai compatible llm fields to chat_logs

Revision ID: 66af22511718
Revises: b33ed1d7647f
Create Date: 2026-01-24 06:53:29.077200
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "66af22511718"
down_revision: Union[str, Sequence[str], None] = "b33ed1d7647f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # OpenAI-compatible LLM parameters
    op.add_column("chat_logs", sa.Column("temperature", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("max_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("top_p", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("frequency_penalty", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("presence_penalty", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("seed", sa.Integer(), nullable=True))

    # Source of truth: OpenAI-compatible messages payload
    op.add_column(
        "chat_logs",
        sa.Column(
            "messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_logs", "messages")
    op.drop_column("chat_logs", "seed")
    op.drop_column("chat_logs", "presence_penalty")
    op.drop_column("chat_logs", "frequency_penalty")
    op.drop_column("chat_logs", "top_p")
    op.drop_column("chat_logs", "max_tokens")
    op.drop_column("chat_logs", "temperature")
