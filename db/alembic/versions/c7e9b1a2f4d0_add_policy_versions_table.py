"""add policy_versions history table

Per-version snapshot of policy nodes so a label's (policy_id, version) pin
resolves to the exact node text it used, even after later version bumps.

Revision ID: c7e9b1a2f4d0
Revises: b2f4a1c9d3e7
Create Date: 2026-08-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = 'c7e9b1a2f4d0'
down_revision = 'b2f4a1c9d3e7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('policy_versions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('policy_id', sa.String(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('category', sa.String(), nullable=False),
    sa.Column('parent_id', sa.String(), nullable=True),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('structured_ref', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('policy_id', 'version', name='uq_policy_versions_id_version')
    )
    op.create_index(op.f('ix_policy_versions_policy_id'), 'policy_versions', ['policy_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_policy_versions_policy_id'), table_name='policy_versions')
    op.drop_table('policy_versions')
