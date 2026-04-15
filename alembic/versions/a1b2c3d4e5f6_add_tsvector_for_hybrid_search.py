"""Add tsvector column and GIN index to document_chunks for hybrid search

Adds a GENERATED ALWAYS AS tsvector column so PostgreSQL full-text search
can be combined with pgvector cosine similarity via Reciprocal Rank Fusion.

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-04-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Generated stored tsvector — updated automatically on text changes.
    op.execute(
        sa.text(
            "ALTER TABLE document_chunks "
            "ADD COLUMN text_search tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
        )
    )
    # GIN index for fast full-text lookups.
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_text_search "
            "ON document_chunks USING GIN(text_search)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_text_search"))
    op.execute(sa.text("ALTER TABLE document_chunks DROP COLUMN IF EXISTS text_search"))
