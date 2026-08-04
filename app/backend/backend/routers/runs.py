"""Pipeline run triggers: ingestion + labelling.

Runs execute as FastAPI background tasks; status is tracked in a simple
in-process registry (PoC — not durable across restarts).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/api", tags=["runs"])

# run_id -> run record. In-process only.
_RUNS: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run(kind: str, video_ids: list[str]) -> dict:
    run_id = f"{kind}-{uuid.uuid4().hex[:8]}"
    run = {
        "run_id": run_id, "kind": kind, "status": "queued",
        "video_ids": video_ids, "done": [], "failed": {},
        "started_at": _now(), "finished_at": None,
    }
    _RUNS[run_id] = run
    return run


async def _run_ingest(run_id: str, video_ids: list[str]) -> None:
    from ingestion import ingest_video  # heavy import; kept out of module load
    run = _RUNS[run_id]
    run["status"] = "running"
    for vid in video_ids:
        try:
            await ingest_video(vid)
            run["done"].append(vid)
        except Exception as exc:  # noqa: BLE001
            run["failed"][vid] = str(exc)
    run["status"] = "failed" if run["failed"] else "completed"
    run["finished_at"] = _now()


def _run_label(run_id: str, video_ids: list[str]) -> None:
    from labelling import label_video
    run = _RUNS[run_id]
    run["status"] = "running"
    for vid in video_ids:
        try:
            label_video(vid)
            run["done"].append(vid)
        except Exception as exc:  # noqa: BLE001
            run["failed"][vid] = str(exc)
    run["status"] = "failed" if run["failed"] else "completed"
    run["finished_at"] = _now()


@router.post("/ingest")
def ingest(video_ids: list[str], background: BackgroundTasks):
    """Kick off the fixed ingestion pipeline for the given YouTube video ids."""
    run = _new_run("ingest", video_ids)
    background.add_task(_run_ingest, run["run_id"], video_ids)
    return run


@router.post("/label")
def label(video_ids: list[str], background: BackgroundTasks):
    """Run the labelling agent over already-ingested videos."""
    run = _new_run("label", video_ids)
    background.add_task(_run_label, run["run_id"], video_ids)
    return run


@router.get("/runs/{run_id}")
def run_status(run_id: str):
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return run
