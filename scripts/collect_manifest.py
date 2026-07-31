"""Collect a bootstrap dataset manifest from YouTube via yt-dlp search.

Reads config/dataset_queries.yaml, runs `ytsearchN:<query>` for each query
(metadata only, no download), de-duplicates by video id, and writes
data/manifest/bootstrap.jsonl. Records the Video ID + light metadata only;
actual media is fetched later by the ingestion step.

    python scripts/collect_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "config" / "dataset_queries.yaml"
OUT = ROOT / "data" / "manifest" / "bootstrap.jsonl"

YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,   # search results only; no per-video page fetch
    "skip_download": True,
}


def main() -> None:
    cfg = yaml.safe_load(QUERIES.read_text())
    n = cfg.get("results_per_query", 10)
    queries: list[str] = cfg["queries"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []

    with YoutubeDL(YDL_OPTS) as ydl:
        for q in queries:
            try:
                info = ydl.extract_info(f"ytsearch{n}:{q}", download=False)
            except Exception as e:  # noqa: BLE001 — keep going on a bad query
                print(f"  ! query failed: {q!r}: {e}")
                continue
            entries = info.get("entries") or []
            kept = 0
            for e in entries:
                vid = e.get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                rows.append({
                    "video_id": vid,
                    "title": e.get("title"),
                    "channel_id": e.get("channel_id") or e.get("uploader_id"),
                    "uploader": e.get("uploader"),
                    "duration_s": e.get("duration"),
                    "view_count": e.get("view_count"),
                    "query": q,
                })
                kept += 1
            print(f"  {q!r}: +{kept} (total {len(rows)})")

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} unique videos -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
