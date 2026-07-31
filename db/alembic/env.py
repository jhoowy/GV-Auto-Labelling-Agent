"""Alembic environment. Autogenerate targets db.models metadata.

The first migration must `CREATE EXTENSION IF NOT EXISTS vector;` before
creating pgvector columns, plus tsvector/GIN indexes for BM25.
"""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.base import Base
from db import models  # noqa: F401  (register tables on Base.metadata)

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    os.getenv("DATABASE_URL", "postgresql+psycopg://vlabel:vlabel@localhost:5432/vlabel"),
)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"),
                      target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.", poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
