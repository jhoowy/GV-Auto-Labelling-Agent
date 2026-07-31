"""Reconcile the manifest against downloaded media.

Cross-checks data/manifest/bootstrap.jsonl with the files present in the blob
store, reports coverage, and writes data/manifest/ingest_ready.jsonl — the
videos that are both downloaded and within the duration limit, i.e. the actual
ingestion work-list.

    python scripts/reconcile.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest" / "bootstrap.jsonl"
BLOB_DIR = Path(os.getenv("BLOB_LOCAL_DIR", ROOT / "blobs")) / "media"
OUT = ROOT / "data" / "manifest" / "ingest_ready.jsonl"
MAX_DURATION = 1800


def main() -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines()]
    have = {p.stem for p in BLOB_DIR.glob("*.mp4")}

    downloaded = [r for r in rows if r["video_id"] in have]
    ready = [r for r in downloaded
             if r.get("duration_s") and r["duration_s"] <= MAX_DURATION]
    missing = [r for r in rows if r["video_id"] not in have]

    with OUT.open("w") as f:
        for r in ready:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_gb = sum((BLOB_DIR / f"{v}.mp4").stat().st_size for v in have) / 1e9
    print(f"manifest videos      : {len(rows)}")
    print(f"downloaded (in blob) : {len(downloaded)}  ({total_gb:.1f} GB)")
    print(f"ingestion-ready      : {len(ready)}  (downloaded & <= {MAX_DURATION}s)")
    print(f"missing / failed     : {len(missing)}")
    if missing:
        print("  missing video_ids:", ", ".join(r["video_id"] for r in missing[:20]))
    # blob files with no manifest row (orphans)
    orphans = have - {r["video_id"] for r in rows}
    if orphans:
        print(f"orphan blobs (no manifest row): {len(orphans)}")
    print(f"\nwrote work-list -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
