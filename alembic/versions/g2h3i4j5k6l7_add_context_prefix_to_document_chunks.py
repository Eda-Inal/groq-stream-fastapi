"""Add context_prefix column to document_chunks

Revision ID: g2h3i4j5k6l7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g2h3i4j5k6l7"
down_revision: Union[str, Sequence[str], None] = "52f4ef9962b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("context_prefix", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "context_prefix")
