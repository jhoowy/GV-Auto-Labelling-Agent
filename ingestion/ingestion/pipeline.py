"""Ingestion pipeline — fixed MLLM batch, no agent.

    fetch(yt-dlp) -> segment(fixed chunks + 1fps frames) -> ASR(transcript)
    -> MLLM caption(vision+audio) -> embed(text|image) -> store

Prompts vary per profile; the pipeline shape is fixed. Video acquisition is
documented via yt-dlp only — record the Video ID, nothing else.
"""
from __future__ import annotations

from models import base_config, get_asr, get_mllm
from schemas import Segment, Video
from tools import embeddings, storage
from tools.blob import get_blob_store


def ingest_video(video_id: str) -> Video:
    """Run the full fixed pipeline for one YouTube video id."""
    cfg = base_config()["ingestion"]
    _ = (get_blob_store(), get_mllm(), get_asr(), cfg)
    raise NotImplementedError


def _segment_video(video: Video, segment_seconds: int, fps: int) -> list[Segment]:
    """Fixed-length chunking + frame extraction."""
    raise NotImplementedError


def build_global_overview(video_id: str) -> str:
    """Aggregate shot summaries + metadata into a video overview, injected into
    every labelling window as global context."""
    _ = embeddings, storage
    raise NotImplementedError
