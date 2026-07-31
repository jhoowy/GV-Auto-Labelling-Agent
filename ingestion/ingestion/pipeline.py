"""Ingestion pipeline — fixed batch, no agent.

    split into fixed-length av clips (video+audio) via ffmpeg -> store Video/Segment

Media is already in the blob store (source acquisition documented via yt-dlp).
This stage produces the shot structure and per-segment av clips; the DB keeps
only pointers. Summary, ASR transcript, base_attributes and embeddings are
filled in once the model providers are wired. Keyframes are not pre-extracted —
the Omni/ASR models consume the clip directly.
"""
from __future__ import annotations

import json
import os
import subprocess
from math import ceil
from pathlib import Path

from models import base_config
from schemas import Segment, Video, VideoMetadata
from schemas.enums import VideoStatus
from tools import storage

ROOT = Path(__file__).resolve().parents[2]
BLOB = Path(os.getenv("BLOB_LOCAL_DIR", ROOT / "blobs"))
MEDIA = BLOB / "media"
CLIPS = BLOB / "clips"
MANIFEST = ROOT / "data" / "manifest" / "ingest_ready.jsonl"

_manifest: dict[str, dict] | None = None


def _manifest_row(video_id: str) -> dict:
    global _manifest
    if _manifest is None:
        _manifest = {}
        if MANIFEST.exists():
            for line in MANIFEST.read_text().splitlines():
                r = json.loads(line)
                _manifest[r["video_id"]] = r
    return _manifest.get(video_id, {})


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _extract_clip(media: Path, start: float, dur: float, dest: Path) -> None:
    """Cut an av clip (video+audio) with stream copy — fast and lossless.
    Container is .mkv so it accepts the source codecs (e.g. vp9/opus)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(media),
         "-t", str(dur), "-c", "copy", str(dest)],
        check=True,
    )


def ingest_video(video_id: str) -> Video:
    """Split one video into fixed-length av clips and store Video/Segment rows."""
    cfg = base_config()["ingestion"]
    seg_s = cfg["segment_seconds"]
    media = MEDIA / f"{video_id}.mp4"
    if not media.exists():
        raise FileNotFoundError(media)

    duration = _probe_duration(media)
    row = _manifest_row(video_id)
    video = Video(
        video_id=video_id,
        metadata=VideoMetadata(title=row.get("title"), channel_id=row.get("channel_id")),
        duration_s=duration, source_blob=f"media/{video_id}.mp4",
        status=VideoStatus.INGESTED,
    )

    segments: list[Segment] = []
    for idx in range(ceil(duration / seg_s)):
        start = idx * seg_s
        end = min(start + seg_s, duration)
        clip = CLIPS / video_id / f"seg{idx:04d}.mkv"
        _extract_clip(media, start, end - start, clip)
        segments.append(Segment(
            segment_id=f"{video_id}_{idx:04d}", video_id=video_id, idx=idx,
            t_start=start, t_end=end, clip_blob=str(clip.relative_to(BLOB)),
            status="ingested",
        ))

    storage.upsert_video(video)
    storage.upsert_segments(segments)
    return video


def build_global_overview(video_id: str) -> str:
    """Aggregate shot summaries + metadata into a video overview, injected into
    every labelling window as global context."""
    row = _manifest_row(video_id)
    summaries = [s.summary for s in storage.get_segments(video_id) if s.summary]
    parts = [f"Title: {row.get('title', '')}"]
    if summaries:
        parts.append("Shots: " + " ".join(summaries))
    return "\n".join(parts)
