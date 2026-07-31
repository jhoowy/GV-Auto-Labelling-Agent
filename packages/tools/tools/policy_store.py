"""Policy store service layer.

Manages the policy-node tree, per-node versions + policy-set snapshots, and
the human-review change-request queue.
"""
from __future__ import annotations

from schemas import Policy, PolicyChangeRequest, PolicySet


def get_policy_tree(category: str) -> list[Policy]:
    """All active nodes for a category (rubric + attribute defs + edge rules)."""
    raise NotImplementedError


def upsert_policy(policy: Policy) -> Policy:
    """Create/edit a node; bumps node version."""
    raise NotImplementedError


def snapshot_policy_set(note: str | None = None) -> PolicySet:
    """Tag the whole tree as policy-set vN."""
    raise NotImplementedError


def resolve_policy(policy_id: str, version: int) -> Policy:
    """Fetch the exact (id, version) a label pinned."""
    raise NotImplementedError


def enqueue_change_request(req: PolicyChangeRequest) -> None:
    raise NotImplementedError


def list_change_requests(status: str = "queued") -> list[PolicyChangeRequest]:
    raise NotImplementedError


def resolve_change_request(req_id: str, approve: bool) -> None:
    raise NotImplementedError
