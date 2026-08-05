"""Policy + change-request queue endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from tools import db_browser, policy_store, tracking, translate

router = APIRouter(prefix="/api", tags=["policy"])


@router.get("/policies")
def get_policy_tree(category: str):
    return policy_store.get_policy_tree(category)


@router.post("/policies/{category}/translate")
def translate_category(category: str):
    """Populate Korean (presentation-only) translations for a category's policy
    nodes. Idempotent + version-cached: only nodes changed since their last
    translation are re-run. Returns {translated, skipped} node-id summary."""
    return translate.translate_category(category)


@router.get("/policies/{category}/rule/{rule_index}/segments")
def rule_segments(category: str, rule_index: int):
    """Segments labelled via decision-tree rule `rule_index` of `category`."""
    return tracking.segments_for_rule(category, rule_index)


@router.get("/policies/{category}/attribute/{name}/segments")
def attribute_segments(category: str, name: str, value: str):
    """Segments whose label recorded attribute `name == value` for `category`."""
    return tracking.segments_for_attribute_value(category, name, value)


@router.get("/policies/{category}/attribute/{name}/examples")
def attribute_examples(category: str, name: str, value: str, limit: int = 3):
    """Up to `limit` representative example segments labelled with attribute
    `name == value` — each with its ingestion summary — for the attribute-detail
    panel. Keyframes come from GET /api/segments/{segment_id}/keyframe."""
    return tracking.attribute_value_examples(category, name, value, limit)


@router.get("/policy-sets")
def list_policy_sets():
    """Policy-set snapshots, newest version first."""
    return db_browser.list_policy_sets()


@router.post("/policy-sets")
def snapshot(note: str | None = None):
    return policy_store.snapshot_policy_set(note)


@router.get("/policy-change-requests")
def list_queue(status: str = "queued"):
    return policy_store.list_change_requests(status)


@router.post("/policy-change-requests/{req_id}/resolve")
def resolve(req_id: str, approve: bool):
    policy_store.resolve_change_request(req_id, approve)
    return {"req_id": req_id, "approved": approve}
