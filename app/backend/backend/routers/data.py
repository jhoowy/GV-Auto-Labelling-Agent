"""Data browsing endpoints — back the Data Viewer."""
from __future__ import annotations

from fastapi import APIRouter

from tools import storage

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/videos")
def list_videos():
    raise NotImplementedError


@router.get("/videos/{video_id}")
def get_video(video_id: str):
    return storage.get_video(video_id)


@router.get("/segments/{segment_id}")
def get_segment(segment_id: str):
    raise NotImplementedError


@router.get("/labels")
def list_labels(segment_id: str | None = None):
    """Labels include their full trace (rationale, cited policies, tool_trace)."""
    raise NotImplementedError
