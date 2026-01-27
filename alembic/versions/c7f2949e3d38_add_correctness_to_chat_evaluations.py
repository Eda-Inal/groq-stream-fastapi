"""add correctness to chat_evaluations

Revision ID: c7f2949e3d38
Revises: 87305e11a64c
Create Date: 2026-01-27 06:07:31.800396
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7f2949e3d38'
down_revision: Union[str, Sequence[str], None] = '87305e11a64c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'chat_evaluations',
        sa.Column('correctness', sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chat_evaluations', 'correctness')
