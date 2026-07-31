"""Pipeline run triggers: ingestion + labelling."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/ingest")
def ingest(video_ids: list[str]):
    """Kick off the fixed ingestion pipeline for the given YouTube video ids."""
    raise NotImplementedError


@router.post("/label")
def label(video_ids: list[str]):
    """Run the labelling agent over already-ingested videos."""
    raise NotImplementedError


@router.get("/runs/{run_id}")
def run_status(run_id: str):
    raise NotImplementedError
