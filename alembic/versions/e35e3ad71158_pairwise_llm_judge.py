"""pairwise_llm_judge

Revision ID: e35e3ad71158
Revises: c7f2949e3d38
Create Date: 2026-01-27 12:10:59.399436
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e35e3ad71158"
down_revision: Union[str, Sequence[str], None] = "c7f2949e3d38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade schema:
    - Drop old single-score evaluation table
    - Create new pairwise evaluation table
    """

    # --- Drop old single-score table ---
    op.drop_table("chat_evaluations")

    # --- Create new pairwise evaluation table ---
    op.create_table(
        "chat_pairwise_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True),

        sa.Column(
            "chat_log_id",
            sa.Integer(),
            sa.ForeignKey("chat_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column(
            "rubric_version",
            sa.String(length=50),
            nullable=False,
            server_default="v1",
        ),

        sa.Column(
            "candidate_model_a",
            sa.String(length=100),
            nullable=False,
        ),

        sa.Column(
            "candidate_model_b",
            sa.String(length=100),
            nullable=False,
        ),

        sa.Column(
            "answer_a",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "answer_b",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "judge_model_name",
            sa.String(length=100),
            nullable=False,
        ),

        sa.Column(
            "winner",
            sa.String(length=10),
            nullable=False,
        ),

        sa.Column(
            "notes",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),

        sa.UniqueConstraint(
            "chat_log_id",
            "rubric_version",
            "candidate_model_a",
            "candidate_model_b",
            name="uq_chat_pairwise_log_rubric_pair",
        ),
    )

    op.create_index(
        "ix_chat_pairwise_evaluations_chat_log_id",
        "chat_pairwise_evaluations",
        ["chat_log_id"],
    )


def downgrade() -> None:
    """
    Downgrade schema:
    - Drop pairwise evaluation table
    - Restore old single-score evaluation table
    """

    op.drop_index(
        "ix_chat_pairwise_evaluations_chat_log_id",
        table_name="chat_pairwise_evaluations",
    )

    op.drop_table("chat_pairwise_evaluations")

    # --- Restore old single-score table ---
    op.create_table(
        "chat_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True),

        sa.Column(
            "chat_log_id",
            sa.Integer(),
            sa.ForeignKey("chat_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column(
            "judge_model_name",
            sa.String(length=100),
            nullable=False,
        ),

        sa.Column(
            "rubric_version",
            sa.String(length=50),
            nullable=False,
            server_default="v1",
        ),

        sa.Column("correctness", sa.Integer(), nullable=False),
        sa.Column("relevance", sa.Integer(), nullable=False),
        sa.Column("completeness", sa.Integer(), nullable=False),
        sa.Column("clarity", sa.Integer(), nullable=False),
        sa.Column("overall_score", sa.Integer(), nullable=False),

        sa.Column(
            "notes",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),

        sa.UniqueConstraint(
            "chat_log_id",
            "rubric_version",
            name="uq_chat_eval_log_rubric",
        ),
    )

    op.create_index(
        "ix_chat_evaluations_chat_log_id",
        "chat_evaluations",
        ["chat_log_id"],
    )
