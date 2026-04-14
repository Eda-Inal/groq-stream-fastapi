"""Replace IVFFlat vector index with HNSW on document_chunks.embedding

IVFFlat requires a minimum of (lists * 10) rows before the planner uses it;
with lists=100 that means 1000 chunks. Below that threshold PostgreSQL falls
back to a sequential scan while still paying the index-planning overhead.

HNSW has no minimum-row requirement, maintains good recall at any scale,
and performs better on approximate nearest-neighbour searches in practice.

Revision ID: d4e5f6a7b8c9
Revises: a9c4e2f18d0b
Create Date: 2026-04-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "a9c4e2f18d0b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_embedding_ivfflat"))
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_embedding_ivfflat ON document_chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
    )
