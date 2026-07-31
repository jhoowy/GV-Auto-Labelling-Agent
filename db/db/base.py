"""Engine / session / declarative base for Postgres."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def get_engine():
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://vlabel:vlabel@localhost:5432/vlabel",
    )
    return create_engine(url, future=True)


SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, future=True)
