"""add category/node_type to policy_change_requests

Lets an approved change request materialise into a policy node (ATTRIBUTE /
EDGE_CASE) under its category — closes the bootstrap → policy-set loop.

Revision ID: b2f4a1c9d3e7
Revises: 776a5d3e1eb0
Create Date: 2026-08-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'b2f4a1c9d3e7'
down_revision = '776a5d3e1eb0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('policy_change_requests', sa.Column('category', sa.String(), nullable=True))
    op.add_column('policy_change_requests', sa.Column('node_type', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('policy_change_requests', 'node_type')
    op.drop_column('policy_change_requests', 'category')
