"""Policy store service layer.

Manages the policy-node tree, per-node versions + policy-set snapshots, and
the human-review change-request queue.

Versioning model: the `policies` table keeps a single row per `policy_id`
(policy_id is the sole PK), so each row holds the node's *current* version.
`upsert_policy` bumps that integer on every edit and appends the corresponding
`policy_versions` snapshot, so `resolve_policy` can reproduce the exact text a
Label pinned. A `PolicySet` snapshot records the {policy_id -> version} map that
was active when it was tagged.
"""
from __future__ import annotations

import logging
import uuid

from db import SessionLocal
from db import models as m
from schemas import Policy, PolicyChangeRequest, PolicySet
from schemas.enums import Category, ChangeRequestStatus, PolicyType

log = logging.getLogger(__name__)


# --- mapping helpers -------------------------------------------------------

def _to_schema(o: "m.Policy") -> Policy:
    return Policy(
        policy_id=o.policy_id,
        type=PolicyType(o.type),
        category=o.category,
        version=o.version,
        parent_id=o.parent_id,
        text=o.text,
        embedding=o.embedding,
        structured_ref=o.structured_ref,
        structured_data=o.structured_data,
        status=o.status,
    )


def _ver_to_schema(o: "m.PolicyVersion") -> Policy:
    """Map a historical policy_versions row to the Policy schema.

    History rows don't carry an embedding or status; those fields are
    policy-independent and not needed to reproduce the pinned text.
    """
    return Policy(
        policy_id=o.policy_id,
        type=PolicyType(o.type),
        category=o.category,
        version=o.version,
        parent_id=o.parent_id,
        text=o.text,
        embedding=None,
        structured_ref=o.structured_ref,
        structured_data=o.structured_data,
        status="active",
    )


def _req_to_schema(o: "m.PolicyChangeRequest") -> PolicyChangeRequest:
    return PolicyChangeRequest(
        req_id=o.req_id,
        proposed_change=o.proposed_change,
        rationale=o.rationale,
        category=o.category,
        node_type=o.node_type,
        target_policy_id=o.target_policy_id,
        affected_segments=list(o.affected_segments or []),
        similar_policies=list(o.similar_policies or []),
        status=ChangeRequestStatus(o.status),
    )


def _embed_or_none(text: str) -> list[float] | None:
    """Embed policy text, but never let a down embedding server block a write.

    Returns None (and logs) on any failure; the node is stored text-only and
    can be re-embedded later.
    """
    try:
        from tools.embeddings import embed_text

        return embed_text([text])[0]
    except Exception as e:  # embedding server down / not configured
        log.warning("embed_text failed for policy node; storing embedding=None (%s)", e)
        return None


# --- tree -----------------------------------------------------------------

def get_policy_tree(category: str) -> list[Policy]:
    """All active nodes for a category (rubric + attribute defs + edge rules)."""
    with SessionLocal() as s:
        objs = (
            s.query(m.Policy)
            .filter_by(category=category, status="active")
            .order_by(m.Policy.type, m.Policy.policy_id)
            .all()
        )
        return [_to_schema(o) for o in objs]


def upsert_policy(policy: Policy) -> Policy:
    """Create or edit a node; bumps the node version on edit.

    - Empty policy_id => new node, version 1, id generated.
    - Existing policy_id => edit; version = current + 1.
    - Embedding is recomputed from the node text, guarded so the write still
      succeeds (embedding=None) when the embedding server is unavailable.
    """
    with SessionLocal() as s:
        pid = policy.policy_id or f"{policy.category}.{policy.type.value}.{uuid.uuid4().hex[:8]}"
        obj = s.get(m.Policy, pid)

        embedding = _embed_or_none(policy.text)

        if obj is None:
            obj = m.Policy(policy_id=pid)
            obj.version = 1
            s.add(obj)
        else:
            obj.version = obj.version + 1  # per-node bump on edit

        obj.type = policy.type.value
        obj.category = getattr(policy.category, "value", policy.category)
        obj.parent_id = policy.parent_id
        obj.text = policy.text
        obj.embedding = embedding
        obj.structured_ref = policy.structured_ref
        obj.structured_data = policy.structured_data
        obj.status = policy.status or "active"

        # Append the per-version snapshot (idempotent on policy_id+version).
        exists = (
            s.query(m.PolicyVersion)
            .filter_by(policy_id=pid, version=obj.version)
            .first()
        )
        if exists is None:
            s.add(m.PolicyVersion(
                policy_id=pid, version=obj.version, type=obj.type,
                category=obj.category, parent_id=obj.parent_id,
                text=obj.text, structured_ref=obj.structured_ref,
                structured_data=obj.structured_data,
            ))

        s.commit()
        return _to_schema(obj)


def upsert_structured_attribute(
    category: str,
    name: str,
    levels: dict[str, list[str]],
    description: str | None = None,
) -> Policy:
    """Create/edit an ATTRIBUTE node holding structured term-level data.

    The node id is deterministic (`<category>.attr.<name>`) so re-running edits
    in place and bumps the version. `levels` maps PEGI score band -> terms; a
    term matched at level L is evidence toward score L for the category. Used by
    the bootstrap drafting skill; not a human-review-gated change.
    """
    cat = getattr(category, "value", category)
    text = description or (
        f"Structured '{name}' attribute for {cat}: terms are organised by PEGI "
        "score level; a term matched at level L is evidence toward score L. The "
        "term lists live in structured_data, not in this text."
    )
    return upsert_policy(Policy(
        policy_id=f"{cat}.attr.{name}",
        type=PolicyType.ATTRIBUTE,
        category=Category(cat),
        parent_id=f"{cat}.scoring",
        text=text,
        structured_data={"kind": "term_levels", "levels": levels},
    ))


def _coerce_values(values: list | None) -> list[dict] | None:
    """Normalise a `values` enum into the rich dict form.

    Each value becomes `{"value","label","description","examples"}`. A plain
    `list[str]` (legacy callers) is coerced with value=label=str and empty
    copy; a list of dicts is filled in for any missing keys. `None` stays None
    (boolean / unbounded attributes carry no closed value set)."""
    if not values:
        return None
    out: list[dict] = []
    for v in values:
        if isinstance(v, dict):
            val = str(v.get("value", v.get("label", "")))
            out.append({
                "value": val,
                "label": str(v.get("label", val)),
                "description": str(v.get("description", "") or ""),
                "examples": list(v.get("examples") or []),
            })
        else:
            s = str(v)
            out.append({"value": s, "label": s, "description": "", "examples": []})
    return out


def upsert_attribute_definition(
    category: str,
    name: str,
    value_type: str,
    guidelines: str,
    scores_informed: list[int],
    values: list | None = None,
    examples: list | None = None,
) -> Policy:
    """Create/edit an ATTRIBUTE node holding an attribute *definition*.

    The node id is deterministic (`<category>.attr.<name>`), parented under the
    scoring rubric. `value_type` is one of boolean/categorical/ordinal; `values`
    is a closed enum, stored as a list of
    `{"value","label","description","examples"}` dicts (a plain `list[str]` is
    coerced to that form). For `ordinal`, values are kept in ascending order so
    rules can compare with `>=`. `guidelines` describe how to detect the
    attribute from a shot; `scores_informed` are the PEGI score bands this
    attribute is evidence for. Reuses the versioned upsert path (bump +
    history)."""
    cat = getattr(category, "value", category)
    scores = [int(s) for s in scores_informed]
    coerced = _coerce_values(values)
    # Fold a flat top-level `examples` (legacy callers) into the first value.
    if examples and coerced:
        coerced[0]["examples"] = list(coerced[0]["examples"]) + list(examples)
    text = (
        f"Attribute '{name}' ({value_type}) for {cat}. {guidelines} "
        f"Informs scores: {', '.join(str(s) for s in scores) or 'none'}."
    )
    return upsert_policy(Policy(
        policy_id=f"{cat}.attr.{name}",
        type=PolicyType.ATTRIBUTE,
        category=Category(cat),
        parent_id=f"{cat}.scoring",
        text=text,
        structured_data={
            "kind": "attribute_def",
            "value_type": value_type,
            "values": coerced,
            "guidelines": guidelines,
            "scores_informed": scores,
        },
    ))


def upsert_decision_rule(
    category: str,
    rules: list[dict],
    default: int = 0,
) -> Policy:
    """Create/edit the category's DECISION_RULE node — a priority-ordered,
    attribute-based decision tree. The node id is deterministic
    (`<category>.rules`), parented under the scoring rubric. `rules` are
    evaluated in order; the first fully-matching rule's `score` wins, else
    `default`. Reuses the versioned upsert path (bump + history)."""
    cat = getattr(category, "value", category)
    text = (
        f"Decision rule tree for {cat}: {len(rules)} priority-ordered rule(s) "
        f"over policy attributes; first fully-matching rule wins, else default "
        f"score {int(default)}. The rules live in structured_data."
    )
    return upsert_policy(Policy(
        policy_id=f"{cat}.rules",
        type=PolicyType.DECISION_RULE,
        category=Category(cat),
        parent_id=f"{cat}.scoring",
        text=text,
        structured_data={
            "kind": "decision_tree",
            "default": int(default),
            "rules": list(rules),
        },
    ))


def snapshot_policy_set(note: str | None = None) -> PolicySet:
    """Tag the whole active tree as policy-set vN.

    N = max existing set version + 1. Stores the {policy_id -> version} map of
    every active node so a label's pins can be interpreted against the set.
    """
    with SessionLocal() as s:
        last = s.query(m.PolicySet).order_by(m.PolicySet.version.desc()).first()
        version = (last.version + 1) if last else 1

        active = s.query(m.Policy).filter_by(status="active").all()
        policy_versions = {p.policy_id: p.version for p in active}

        obj = m.PolicySet(version=version, policy_versions=policy_versions, note=note)
        s.add(obj)
        s.commit()
        return PolicySet(version=version, policy_versions=policy_versions, note=note)


def resolve_policy(policy_id: str, version: int) -> Policy:
    """Fetch the exact (id, version) a label pinned.

    Reads the historical node from `policy_versions`, reproducing the text as it
    was at that version. If the pinned (id, version) predates history capture,
    fall back to the current `policies` row and log the divergence, so
    audit/trace never hard-crashes on an older pin.
    """
    with SessionLocal() as s:
        hist = (
            s.query(m.PolicyVersion)
            .filter_by(policy_id=policy_id, version=version)
            .first()
        )
        if hist is not None:
            return _ver_to_schema(hist)

        obj = s.get(m.Policy, policy_id)
        if obj is None:
            raise KeyError(f"policy '{policy_id}' not found")
        log.warning(
            "resolve_policy: pinned %s v%s absent from history; returning current v%s",
            policy_id, version, obj.version,
        )
        return _to_schema(obj)


# --- change-request queue -------------------------------------------------

def enqueue_change_request(req: PolicyChangeRequest) -> None:
    """Queue a policy edit proposal for human review. Generates req_id if empty."""
    with SessionLocal() as s:
        req_id = req.req_id or f"pcr-{uuid.uuid4().hex[:12]}"
        s.merge(m.PolicyChangeRequest(
            req_id=req_id,
            proposed_change=req.proposed_change,
            rationale=req.rationale,
            category=req.category,
            node_type=req.node_type,
            target_policy_id=req.target_policy_id,
            affected_segments=list(req.affected_segments),
            similar_policies=list(req.similar_policies),
            status=getattr(req.status, "value", req.status) or "queued",
        ))
        s.commit()


def list_change_requests(status: str = "queued") -> list[PolicyChangeRequest]:
    with SessionLocal() as s:
        q = s.query(m.PolicyChangeRequest)
        if status:
            q = q.filter_by(status=status)
        return [_req_to_schema(o) for o in q.order_by(m.PolicyChangeRequest.req_id).all()]


def resolve_change_request(req_id: str, approve: bool) -> None:
    """Human decision on a queued request. Approving MATERIALISES the proposal
    into a policy node (ATTRIBUTE / EDGE_CASE) under its category; rejecting only
    marks it. This is what lets the bootstrap queue converge into the tree."""
    with SessionLocal() as s:
        obj = s.get(m.PolicyChangeRequest, req_id)
        if obj is None:
            raise KeyError(f"change request '{req_id}' not found")
        if obj.status != ChangeRequestStatus.QUEUED.value:
            return  # already resolved — idempotent
        obj.status = (
            ChangeRequestStatus.APPROVED.value if approve
            else ChangeRequestStatus.REJECTED.value
        )
        s.commit()
    if approve:
        _materialise_request(req_id)


def _materialise_request(req_id: str) -> Policy | None:
    """Create a policy node from an approved change request (parented under the
    category's scoring rubric)."""
    with SessionLocal() as s:
        obj = s.get(m.PolicyChangeRequest, req_id)
        if obj is None or not obj.category:
            return None
        cat, node_type = obj.category, (obj.node_type or "edge_case")
        proposed, rationale = obj.proposed_change, obj.rationale
    try:
        category = Category(cat)
    except ValueError:
        log.warning("request %s has unknown category %r; not materialised", req_id, cat)
        return None
    ptype = PolicyType.ATTRIBUTE if node_type == "attribute" else PolicyType.EDGE_CASE
    text = proposed if not rationale else f"{proposed}\n\nRationale: {rationale}"
    node = upsert_policy(Policy(
        policy_id=f"{cat}.{node_type}.{req_id.split('-')[-1][:8]}",
        type=ptype, category=category,
        parent_id=f"{cat}.scoring", text=text,
    ))
    log.info("materialised %s node %s from request %s", node_type, node.policy_id, req_id)
    return node
