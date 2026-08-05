"""Data browsing endpoints — back the Data Viewer."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from tools import storage
from tools.blob import get_blob_store

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/videos")
def list_videos(
    search: str | None = None,
    page: int = 1,
    page_size: int = 24,
    dataset: str | None = None,
    status: str | None = None,
    category: str | None = None,
    score_min: int | None = None,
    score_max: int | None = None,
):
    """Paginated video gallery: title/duration/status/thumbnail + segment count.
    Case-insensitive title substring search via `search`; optional `dataset`
    filters on metadata_json->>'dataset'. `status` filters on the labelling
    lifecycle (ingested/labelled/unlabelled); `category` + `score_min`/`score_max`
    keep videos with a matching label in that score range."""
    return storage.list_videos_page(
        search=search, page=page, page_size=page_size, dataset=dataset,
        status=status, category=category,
        score_min=score_min, score_max=score_max,
    )


@router.get("/videos/{video_id}")
def get_video(video_id: str):
    video = storage.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"unknown video {video_id}")
    return video


@router.get("/videos/{video_id}/thumbnail")
def get_video_thumbnail(video_id: str):
    """JPEG thumbnail bytes from the blob store (thumbnails/<video_id>.jpg)."""
    try:
        data = get_blob_store().get(f"thumbnails/{video_id}.jpg")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no thumbnail for {video_id}")
    return Response(content=data, media_type="image/jpeg")


@router.get("/videos/{video_id}/segments")
def list_video_segments(video_id: str):
    """Shots of a video, ordered by idx — backs the Data Viewer timeline."""
    return storage.get_segments(video_id)


@router.get("/segments/{segment_id}")
def get_segment(segment_id: str):
    seg = storage.get_segment(segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown segment {segment_id}")
    return seg


@router.get("/segments/{segment_id}/keyframe")
def get_segment_keyframe(segment_id: str):
    """One representative JPEG keyframe sampled from a segment's clip. Reuses the
    labelling frame sampler (imported lazily to keep media/model deps off the
    module import path). 404 if the segment or a decodable frame is missing."""
    seg = storage.get_segment(segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown segment {segment_id}")
    from labelling.tools import sample_frames

    frames = sample_frames(seg, 1)
    if not frames:
        raise HTTPException(
            status_code=404, detail=f"no keyframe for segment {segment_id}"
        )
    return Response(content=frames[0], media_type="image/jpeg")


@router.get("/labels")
def list_labels(segment_id: str | None = None):
    """Labels include their full trace (rationale, cited policies, tool_trace)."""
    return storage.list_labels(segment_id)
