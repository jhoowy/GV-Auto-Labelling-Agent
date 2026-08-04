"""Ingestion pipeline — ASR + Omni scene segmentation (no agent).

Per video:
  1) ASR word-level utterances (ASR+align server), fixed windows.
  2) Omni scene boundaries: overlapping windows, per-window segmentation (Pass1),
     reconcile each overlap region (Pass2), assemble into FIXED segments.
  3) Per segment: extract clip -> visual + text embedding, merge overlapping
     utterances as the segment transcript. Store segments + utterances.

The labelling agent consumes these fixed segments (moderation only; no
re-segmentation).

NOTE: Omni-based boundary detection here is a PoC stand-in. Replace with a
dedicated shot/scene-boundary model (e.g. OmniShotCut) later — more accurate
boundaries and no MLLM cost for segmentation.

Needs the ASR+align server (:8810, scripts/serve_asr.sh) and the Omni/embed
servers (scripts/serve_vllm.sh).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from math import ceil
from pathlib import Path

from models import base_config, get_asr, get_image_embedder, get_mllm, get_text_embedder
from schemas import Attribute, Segment, Utterance, Video, VideoMetadata
from schemas.enums import AttributeLayer, VideoStatus
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


def _rm(p: str) -> None:
    try:
        os.unlink(p)
    except OSError:
        pass


def _tmp_wav(media: Path, start: float, dur: float) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(media),
                    "-t", str(dur), "-ar", "16000", "-ac", "1", path], check=True)
    return path


def _tmp_clip(media: Path, start: float, dur: float) -> str:
    # -avoid_negative_ts make_zero resets the clip PTS to 0 (without it, -c copy
    # keeps the source PTS and the model reports out-of-range absolute times).
    fd, path = tempfile.mkstemp(suffix=".mkv")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(media),
                    "-t", str(dur), "-c", "copy", "-avoid_negative_ts", "make_zero", path], check=True)
    return path


def _merge_utts(utts: list[Utterance], a: float, b: float) -> str:
    return " ".join(u.text for u in utts if u.t_start < b and u.t_end > a)


async def _asr_utterances(video_id, media, duration, win_s, asr) -> list[Utterance]:
    starts = [i * win_s for i in range(ceil(duration / win_s))]
    wavs = [_tmp_wav(media, st, min(win_s, duration - st)) for st in starts]
    # One batched request — the ASR engine must not be called concurrently, so
    # all windows go together and are batched server-side (not serialized).
    try:
        results = await asr.transcribe(wavs)
    finally:
        for w in wavs:
            _rm(w)

    utts: list[Utterance] = []
    for st, res in zip(starts, results):
        for w in res["words"]:
            utts.append(Utterance(video_id=video_id, idx=len(utts),
                                  t_start=w["t_start"] + st, t_end=w["t_end"] + st, text=w["text"]))
    return utts


async def _omni_segments(media, duration, W, O, omni) -> list[dict]:
    """Overlapping-window segmentation with per-overlap reconcile. Returns a
    contiguous list of {t_start, t_end, summary}."""
    step = max(W - O, 1.0)
    starts, s = [], 0.0
    while s < duration:
        starts.append(s)
        s += step
    win = [(st, min(W, duration - st)) for st in starts]
    n = len(starts)

    clips = [_tmp_clip(media, st, dur) for st, dur in win]
    auds = [_tmp_wav(media, st, dur) for st, dur in win]
    try:
        # Omni reports ABSOLUTE times (H:MM:SS.ff) — pass the clip's [start, end].
        pass1 = await asyncio.gather(*(
            omni.segment_window(c, st, st + dur, a)
            for c, a, (st, dur) in zip(clips, auds, win)))
    finally:
        for p in clips + auds:
            _rm(p)

    abs_segs = []  # clamp each window's scenes to its bounds (already absolute)
    for (st, dur), segs in zip(win, pass1):
        aw = []
        for sg in segs:
            a, b = max(st, sg["start"]), min(st + dur, sg["end"])
            if b > a:
                aw.append({"t_start": a, "t_end": b, "summary": sg["summary"]})
        abs_segs.append(aw)

    final: list[dict] = []

    def _append(seg):
        cover = final[-1]["t_end"] if final else 0.0
        if seg["t_end"] > cover + 0.5:
            final.append({"t_start": max(seg["t_start"], cover), "t_end": seg["t_end"],
                          "summary": seg["summary"]})

    for i in range(n):
        st, dur = win[i]
        win_end = st + dur
        hi = starts[i + 1] if i < n - 1 else duration
        for sg in abs_segs[i]:
            if sg["t_start"] < hi:
                _append(sg)
        if i < n - 1:
            ov_s, ov_e = starts[i + 1], win_end
            inside_i = [s for s in abs_segs[i] if s["t_end"] > ov_s]
            inside_j = [s for s in abs_segs[i + 1] if s["t_start"] < ov_e]
            ovc, ova = _tmp_clip(media, ov_s, ov_e - ov_s), _tmp_wav(media, ov_s, ov_e - ov_s)
            try:
                rec = await omni.reconcile_overlap(ovc, ov_s, ov_e, inside_i, inside_j, ova)
            finally:
                _rm(ovc); _rm(ova)
            for r in rec:
                _append({"t_start": r["start"], "t_end": r["end"], "summary": r["summary"]})
    if final:
        final[-1]["t_end"] = duration
    return final


def _base_attributes(summary: str | None, asr_text: str, t_start: float, t_end: float) -> list[Attribute]:
    """Policy-independent raw observations, derived only from data already
    produced (no extra model call). Deliberately factual, not judgemental —
    category verdicts are the agent's (policy layer) job.

    NOTE: richer VISION/OCR base attributes (on-screen text, scene objects,
    logos) would come from a dedicated Omni extraction pass here. Deferred as a
    deliberate follow-up — it adds an MLLM call per shot (cost).
    """
    summary = summary or ""
    asr_text = asr_text or ""
    words = asr_text.split()
    obs: list[tuple[str, str | float | bool]] = [
        ("has_speech", bool(words)),
        ("shot_seconds", round(t_end - t_start, 3)),
        ("asr_word_count", len(words)),
        ("summary_len", len(summary)),
    ]
    return [
        Attribute(key=k, value=v, layer=AttributeLayer.BASE, source="ingestion")
        for k, v in obs
    ]


async def _build_segments(video_id, media, bounds, utts, temb, vemb, sem) -> list[Segment]:
    async def build(idx, b):
        async with sem:
            clip_key = f"clips/{video_id}/seg{idx:04d}.mkv"
            dest = BLOB / clip_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = _tmp_clip(media, b["t_start"], b["t_end"] - b["t_start"])
            shutil.move(tmp, dest)
            transcript = _merge_utts(utts, b["t_start"], b["t_end"])
            text_in = f"{b['summary']}\n{transcript}".strip()
            text_v = (await temb.embed([text_in]))[0]
            vis_v = (await vemb.embed([str(dest)]))[0]
            return Segment(
                segment_id=f"{video_id}_{idx:04d}", video_id=video_id, idx=idx,
                t_start=b["t_start"], t_end=b["t_end"], clip_blob=clip_key,
                transcript=transcript or None, summary=b["summary"],
                base_attributes=_base_attributes(b["summary"], transcript, b["t_start"], b["t_end"]),
                text_embedding=text_v, image_embedding=vis_v, status="ingested")

    return list(await asyncio.gather(*(build(i, b) for i, b in enumerate(bounds))))


async def ingest_video(video_id: str, concurrency: int = 8) -> Video:
    cfg = base_config()["ingestion"]
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

    sem = asyncio.Semaphore(concurrency)
    asr, omni = get_asr(), get_mllm()
    temb, vemb = get_text_embedder(), get_image_embedder()

    utts = await _asr_utterances(video_id, media, duration, cfg["asr_window_seconds"], asr)
    bounds = await _omni_segments(media, duration, cfg["omni_window_seconds"],
                                  cfg["omni_overlap_seconds"], omni)
    segments = await _build_segments(video_id, media, bounds, utts, temb, vemb, sem)

    if segments:
        joined = "\n".join(f"[{s.idx}] {s.summary}" for s in segments)
        video.global_overview = await omni.chat(
            "Summarize the whole video in 3-5 sentences from these scene summaries:\n" + joined)
        video.text_embedding = (await temb.embed([video.global_overview]))[0]

    storage.upsert_video(video)
    storage.replace_utterances(video_id, utts)
    storage.upsert_segments(segments)
    return video


def build_global_overview(video_id: str) -> str:
    v = storage.get_video(video_id)
    return (v.global_overview or "") if v else ""


def backfill_base_attributes(video_id: str | None = None) -> int:
    """Recompute base_attributes for already-ingested segments and re-save them.

    Model-free: reuses each segment's stored summary and its merged word-level
    utterances (the same ASR text ingestion would have merged). Scoped to one
    video, or all videos when video_id is None. Returns the number of segments
    updated.
    """
    video_ids = [video_id] if video_id else [v.video_id for v in storage.list_videos()]
    updated = 0
    for vid in video_ids:
        utts = storage.get_utterances(vid)
        segments = storage.get_segments(vid)
        for seg in segments:
            asr_text = seg.transcript or _merge_utts(utts, seg.t_start, seg.t_end)
            seg.base_attributes = _base_attributes(
                seg.summary, asr_text, seg.t_start, seg.t_end)
        if segments:
            storage.upsert_segments(segments)
            updated += len(segments)
    return updated
