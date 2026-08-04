"""add structured_data JSONB to policies and policy_versions

Structured policy payloads (e.g. profanity term lists by score level) are now
DB-managed on the ATTRIBUTE node instead of an external file pointer.

Revision ID: d1f4c7a9b2e5
Revises: c7e9b1a2f4d0
Create Date: 2026-08-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = 'd1f4c7a9b2e5'
down_revision = 'c7e9b1a2f4d0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('policies', sa.Column('structured_data', JSONB(), nullable=True))
    op.add_column('policy_versions', sa.Column('structured_data', JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column('policy_versions', 'structured_data')
    op.drop_column('policies', 'structured_data')
