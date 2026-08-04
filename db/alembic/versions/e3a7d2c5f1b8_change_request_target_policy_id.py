"""add target_policy_id to policy_change_requests

Records which existing policy node a change request targets, so a proposal can
edit an attribute definition or decision-rule node rather than only create one.

Revision ID: e3a7d2c5f1b8
Revises: d1f4c7a9b2e5
Create Date: 2026-08-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'e3a7d2c5f1b8'
down_revision = 'd1f4c7a9b2e5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('policy_change_requests', sa.Column('target_policy_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('policy_change_requests', 'target_policy_id')
