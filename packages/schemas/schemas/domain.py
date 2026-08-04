"""Core domain models (Pydantic) — the shared contract between ingestion,
labelling, policy, backend and UI.

Large binaries (video, clips, keyframes, audio) are not stored here — only
blob-store pointers. Embeddings are plain lists at the contract layer;
persistence uses pgvector.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import (
    AttributeLayer,
    Category,
    ChangeRequestStatus,
    PolicyType,
    VideoStatus,
)


class Attribute(BaseModel):
    """A single observed/derived attribute on a segment."""

    key: str
    value: str | float | bool
    confidence: float | None = None
    evidence: str | None = None          # frame range / text span
    layer: AttributeLayer
    source: str                          # "ingestion" | agent tool name
    policy_version: int | None = None    # set for policy-layer attributes


class VideoMetadata(BaseModel):
    title: str | None = None
    description: str | None = None
    channel_id: str | None = None
    thumbnail_blob: str | None = None    # blob pointer, not the image


class Video(BaseModel):
    video_id: str                        # YouTube ID; acquisition via yt-dlp
    metadata: VideoMetadata = Field(default_factory=VideoMetadata)
    duration_s: float | None = None
    source_blob: str | None = None       # blob pointer to original media
    global_overview: str | None = None   # aggregated shot summaries
    status: VideoStatus = VideoStatus.PENDING
    text_embedding: list[float] | None = None


class Segment(BaseModel):
    """One shot. Final labels are emitted at this granularity."""

    segment_id: str
    video_id: str
    idx: int
    t_start: float
    t_end: float
    clip_blob: str | None = None                          # av clip pointer (video+audio)
    transcript: str | None = None                         # raw ASR
    summary: str | None = None
    base_attributes: list[Attribute] = Field(default_factory=list)
    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    status: str = "pending"


class Utterance(BaseModel):
    """One word-level ASR unit (ForcedAligner output) on the video timeline.
    Produced by ingestion; the agent merges those overlapping a refined segment
    into that segment's ASR attribute."""

    video_id: str
    idx: int
    t_start: float
    t_end: float
    text: str


class Label(BaseModel):
    """Per-segment moderation judgement with its full trace."""

    label_id: str
    segment_id: str
    category: Category
    score: int                                   # 0..5
    rationale: str
    cited_policy_ids: list[str] = Field(default_factory=list)   # (id,version) pins
    evidence_attributes: list[Attribute] = Field(default_factory=list)
    used_segment_ids: list[str] = Field(default_factory=list)   # neighbours/frames
    tool_trace: list[dict] = Field(default_factory=list)        # agent tool calls
    confidence: float | None = None
    human_verified: bool = False


class Policy(BaseModel):
    """A single policy node = one RAG chunk. Human-readable text + embedding."""

    policy_id: str
    type: PolicyType
    category: Category
    version: int = 1
    parent_id: str | None = None
    text: str
    embedding: list[float] | None = None
    structured_ref: str | None = None            # legacy pointer (deprecated)
    structured_data: dict | None = None          # DB-managed structured payload, e.g. term levels
    status: str = "active"


class PolicySet(BaseModel):
    """Snapshot tag over the whole policy tree."""

    version: int
    policy_versions: dict[str, int] = Field(default_factory=dict)  # policy_id -> version
    note: str | None = None


class PolicyChangeRequest(BaseModel):
    """Queued policy edit proposal for a human operator."""

    req_id: str
    proposed_change: str
    rationale: str
    category: str | None = None          # target category of the proposed node
    node_type: str | None = None         # "attribute" | "edge_case" to materialise
    affected_segments: list[str] = Field(default_factory=list)
    similar_policies: list[str] = Field(default_factory=list)
    status: ChangeRequestStatus = ChangeRequestStatus.QUEUED
