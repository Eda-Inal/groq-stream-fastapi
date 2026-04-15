"""Resize embedding vector column from 768 to 1024 for mxbai-embed-large

Revision ID: f1a2b3c4d5e6
Revises: d4e5f6a7b8c9
Create Date: 2026-04-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_DIM = 768
NEW_DIM = 1024


def upgrade() -> None:
    # Drop the HNSW index first (cannot ALTER a column with a dependent index)
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))

    # Delete all existing chunks — their 768-dim vectors are incompatible with 1024-dim
    op.execute(sa.text("DELETE FROM document_chunks"))
    op.execute(sa.text("UPDATE documents SET chunk_count = 0"))

    # Alter the column type to the new dimension
    op.execute(
        sa.text(
            f"ALTER TABLE document_chunks "
            f"ALTER COLUMN embedding TYPE vector({NEW_DIM})"
        )
    )

    # Recreate HNSW index for the new dimension
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))
    op.execute(sa.text("DELETE FROM document_chunks"))
    op.execute(sa.text("UPDATE documents SET chunk_count = 0"))
    op.execute(
        sa.text(
            f"ALTER TABLE document_chunks "
            f"ALTER COLUMN embedding TYPE vector({OLD_DIM})"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
    )
