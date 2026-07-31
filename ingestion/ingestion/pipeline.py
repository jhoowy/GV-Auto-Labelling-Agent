"""Ingestion pipeline — lightweight ASR-only preprocessing (no agent).

Per video: split into fixed-length windows, and for each window call the ASR
server, which returns the transcript + word-level timestamps in one HTTP call
(ASR + forced alignment). Offset each window's words onto the full timeline and
store them as `utterances`.

    video → 30s windows(wav) → ASR+align server → word timestamps
                             → offset by window start → utterances table

Segment boundaries, clips, summaries and embeddings are produced later by the
labelling agent (which merges the utterances in each refined segment's range).

The ASR+align server runs in the isolated qwen-asr venv (scripts/serve_asr.sh);
this code only speaks HTTP to it via the async client.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from math import ceil
from pathlib import Path

from models import base_config, get_asr
from schemas import Utterance, Video, VideoMetadata
from schemas.enums import VideoStatus
from tools import storage

ROOT = Path(__file__).resolve().parents[2]
BLOB = Path(os.getenv("BLOB_LOCAL_DIR", ROOT / "blobs"))
MEDIA = BLOB / "media"
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
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _window_wav(media: Path, start: float, dur: float) -> str:
    """16kHz mono wav for one window (temp file)."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(media),
         "-t", str(dur), "-ar", "16000", "-ac", "1", path], check=True)
    return path


async def ingest_video(video_id: str, concurrency: int = 8) -> Video:
    """Transcribe + word-align the whole video into the utterances table."""
    win_s = base_config()["ingestion"]["segment_seconds"]
    media = MEDIA / f"{video_id}.mp4"
    if not media.exists():
        raise FileNotFoundError(media)

    duration = _probe_duration(media)
    row = _manifest_row(video_id)
    video = Video(
        video_id=video_id,
        metadata=VideoMetadata(title=row.get("title"), channel_id=row.get("channel_id")),
        duration_s=duration, source_blob=f"media/{video_id}.mp4",
        status=VideoStatus.INGESTED)

    windows = [(i * win_s, _window_wav(media, i * win_s, min(win_s, duration - i * win_s)))
               for i in range(ceil(duration / win_s))]

    asr = get_asr()
    sem = asyncio.Semaphore(concurrency)

    async def process(start: float, wav: str) -> tuple[float, list[dict]]:
        async with sem:
            try:
                _text, words = await asr.transcribe(wav)
            finally:
                os.unlink(wav)
        return start, words

    # gather preserves window order → utterance idx is in timeline order.
    results = await asyncio.gather(*(process(s, w) for s, w in windows))

    utterances: list[Utterance] = []
    for start, words in results:
        for w in words:
            utterances.append(Utterance(
                video_id=video_id, idx=len(utterances),
                t_start=w["t_start"] + start, t_end=w["t_end"] + start, text=w["text"]))

    storage.upsert_video(video)
    storage.replace_utterances(video_id, utterances)
    return video


def build_global_overview(video_id: str) -> str:
    """Global overview is produced later by the agent; return what's stored."""
    v = storage.get_video(video_id)
    return (v.global_overview or "") if v else ""
