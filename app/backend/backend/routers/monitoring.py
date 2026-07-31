"""Monitoring endpoints — back the Monitoring page. Consistency metrics are
stubbed so the UI contract exists early."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.get("/metrics")
def metrics():
    raise NotImplementedError


@router.get("/queue")
def queue():
    """Policy change-request queue summary for the operator."""
    raise NotImplementedError


@router.get("/consistency")
def consistency():
    """Cross-sample (primary) + reproducibility (secondary)."""
    raise NotImplementedError
