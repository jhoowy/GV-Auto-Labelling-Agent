"""SQLAlchemy models = the physical schema, managed via Alembic.

Embeddings use pgvector; large media are pointers. Full-text/BM25 is served by
tsvector indexes added in migrations. Embedding dims match config/profiles
(text=3072, image=768) — adjust per profile.
"""
from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Video(Base):
    __tablename__ = "videos"
    video_id: Mapped[str] = mapped_column(String, primary_key=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_blob: Mapped[str | None] = mapped_column(String, nullable=True)
    global_overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    text_embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)


class Segment(Base):
    __tablename__ = "segments"
    segment_id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    frame_ptrs: Mapped[list] = mapped_column(JSONB, default=list)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_attributes: Mapped[list] = mapped_column(JSONB, default=list)
    text_embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    image_embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")


class Label(Base):
    __tablename__ = "labels"
    label_id: Mapped[str] = mapped_column(String, primary_key=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.segment_id"), index=True)
    category: Mapped[str] = mapped_column(String, index=True)
    score: Mapped[int] = mapped_column(Integer)
    rationale: Mapped[str] = mapped_column(Text)
    cited_policy_ids: Mapped[list] = mapped_column(JSONB, default=list)
    evidence_attributes: Mapped[list] = mapped_column(JSONB, default=list)
    used_segment_ids: Mapped[list] = mapped_column(JSONB, default=list)
    tool_trace: Mapped[list] = mapped_column(JSONB, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_verified: Mapped[bool] = mapped_column(Boolean, default=False)


class Policy(Base):
    __tablename__ = "policies"
    policy_id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    structured_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")


class PolicySet(Base):
    __tablename__ = "policy_sets"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_versions: Mapped[dict] = mapped_column(JSONB, default=dict)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class PolicyChangeRequest(Base):
    __tablename__ = "policy_change_requests"
    req_id: Mapped[str] = mapped_column(String, primary_key=True)
    proposed_change: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    affected_segments: Mapped[list] = mapped_column(JSONB, default=list)
    similar_policies: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String, default="queued", index=True)
