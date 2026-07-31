"""Persistence service layer over Postgres.

Thin repository functions used by ingestion, labelling, and the backend.
SQLAlchemy models + sessions live in the `db` package.
"""
from __future__ import annotations

from schemas import Label, Segment, Video


def upsert_video(video: Video) -> None:
    raise NotImplementedError


def get_video(video_id: str) -> Video | None:
    raise NotImplementedError


def upsert_segments(segments: list[Segment]) -> None:
    raise NotImplementedError


def get_segments(video_id: str) -> list[Segment]:
    """Ordered by idx."""
    raise NotImplementedError


def save_label(label: Label) -> None:
    """Persist a label. Append-only; revisions keep history."""
    raise NotImplementedError


def revise_ingestion(segment_id: str, patch: dict) -> None:
    """Apply an agent correction to ingestion output, preserving the original
    and writing a revision log."""
    raise NotImplementedError
