"""Monitoring endpoints — back the Monitoring page. Consistency reproducibility
is stubbed so the UI contract exists early."""
from __future__ import annotations

import math
from collections import defaultdict

from fastapi import APIRouter

from tools import policy_store, storage

router = APIRouter(prefix="/api", tags=["monitoring"])


def _tally(items, key) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for it in items:
        out[str(key(it))] += 1
    return dict(out)


@router.get("/metrics")
def metrics():
    """Counts of videos / segments / labels by status & category."""
    videos = storage.list_videos()
    segments = [s for v in videos for s in storage.get_segments(v.video_id)]
    labels = storage.list_labels()
    return {
        "videos": {
            "total": len(videos),
            "by_status": _tally(videos, lambda v: v.status.value),
        },
        "segments": {
            "total": len(segments),
            "by_status": _tally(segments, lambda s: s.status),
        },
        "labels": {
            "total": len(labels),
            "by_category": _tally(labels, lambda x: x.category.value),
            "by_score": _tally(labels, lambda x: x.score),
        },
    }


@router.get("/queue")
def queue():
    """Policy change-request queue summary for the operator."""
    by_status = {
        st: policy_store.list_change_requests(st)
        for st in ("queued", "approved", "rejected")
    }
    return {
        "counts": {st: len(v) for st, v in by_status.items()},
        "queued": by_status["queued"],
    }


@router.get("/consistency")
def consistency():
    """Cross-sample score dispersion (primary) + reproducibility (stub).

    Cross-sample here is the per-category entropy of the score distribution over
    all stored labels — a PoC proxy for label dispersion. A neighbourhood-level
    metric (via find_similar_segments) and re-run reproducibility are the last
    PoC milestone and remain stubbed.
    """
    labels = storage.list_labels()
    dist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for lb in labels:
        dist[lb.category.value][lb.score] += 1

    per_category = {}
    for cat, scores in dist.items():
        total = sum(scores.values())
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in scores.values() if c
        )
        per_category[cat] = {
            "n": total,
            "score_distribution": {str(k): scores[k] for k in sorted(scores)},
            "score_entropy_bits": round(entropy, 4),
        }

    return {
        "cross_sample": {"per_category": per_category},
        "reproducibility": {
            "status": "stub",
            "note": "re-run agreement not yet measured (PoC last milestone)",
        },
    }
