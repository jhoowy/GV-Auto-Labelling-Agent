"""Node -> segment tracking (derived from existing labels; read-only).

Aggregates each label back into the policy nodes that drove it, so a reviewer
can click a decision-tree rule or an attribute value and see which segments were
labelled through it. Nothing is stored: both queries read `labels` (+ join
`segments` for the video id).

Labels carry NO per-label tool_trace. The signal both queries use is
`evidence_attributes` — one entry per defined attribute; a selected attribute
carries its extracted `value`, unselected ones store value == "". For the
rule -> segment map (`segments_for_rule`) the fired rule is RE-DERIVED by
re-applying the category's current decision tree to those evidence values via
the shared `tools.decision_tree` module (same code the agent scored with).
"""
from __future__ import annotations

from sqlalchemy import text

from db import SessionLocal

from . import decision_tree, policy_store

# Aggregators cap their result set; a full page equal to the cap signals the
# caller (endpoint/UI) that more matches may exist.
RESULT_CAP = 200


def _seg_row(segment_id: str, video_id: str, score: int) -> dict:
    """Shape one matched label into the segment record the UI consumes."""
    return {"segment_id": segment_id, "video_id": video_id, "score": score}


def _category_tree(category: str) -> tuple[list, int, dict] | None:
    """Load `category`'s current decision tree + ordinal order map from the live
    policy tree. Returns (rules, default, order) or None if no `{cat}.rules` node
    exists. `order` maps each ordinal attribute name -> ascending value keys, built
    from the attribute-def nodes exactly as `graph._attr_index` does, so `>=`/`<=`
    rules re-derive identically."""
    rules: list | None = None
    default = 0
    order: dict[str, list] = {}
    for node in policy_store.get_policy_tree(category):
        sd = node.structured_data or {}
        kind = sd.get("kind")
        if kind == "decision_tree" and node.policy_id == f"{category}.rules":
            rules = sd.get("rules") or []
            default = sd.get("default", 0)
        elif kind == "attribute_def" and sd.get("value_type") == "ordinal":
            name = (node.policy_id.split(".attr.", 1)[1]
                    if ".attr." in node.policy_id else node.policy_id)
            vals = sd.get("values") or []
            keys = [v.get("value") if isinstance(v, dict) else v for v in vals]
            if keys:
                order[name] = keys  # ascending -> lets rules compare with >=
    if rules is None:
        return None
    return rules, default, order


def segments_for_rule(category: str, rule_index: int) -> list[dict]:
    """Segments whose label fired decision-tree rule `rule_index` for `category`.

    No trace is stored, so the fired rule is re-derived: for each label, build
    `values` from its non-empty `evidence_attributes`, re-apply the category's
    CURRENT decision tree (`tools.decision_tree._apply_decision_tree`), and keep
    the segment when the matched rule's index == `rule_index`. Done in Python
    over the labels (counts are modest). Capped at RESULT_CAP, ordered by id."""
    tree = _category_tree(category)
    if tree is None:
        return []
    rules, default, order = tree

    sql = text(
        """
        SELECT l.segment_id AS segment_id, s.video_id AS video_id,
               l.score AS score, l.evidence_attributes AS evidence
        FROM labels l
        JOIN segments s ON s.segment_id = l.segment_id
        WHERE l.category = :category
        ORDER BY l.label_id
        """
    )
    out: list[dict] = []
    with SessionLocal() as sess:
        rows = sess.execute(sql, {"category": category}).mappings()
        for r in rows:
            values = {
                a.get("key"): a.get("value")
                for a in (r["evidence"] or [])
                if isinstance(a, dict) and a.get("key") and a.get("value") != ""
            }
            _score, matched = decision_tree._apply_decision_tree(
                rules, default, values, order)
            if matched is None:
                continue  # no rule fired -> not attributable to any rule_index
            if rules.index(matched) == rule_index:
                out.append(_seg_row(r["segment_id"], r["video_id"], r["score"]))
                if len(out) >= RESULT_CAP:
                    break
    return out


def rule_segment_counts(category: str) -> dict[int, int]:
    """Per-rule count of segments matching each decision-tree rule of `category`.

    Like `segments_for_rule` but a SINGLE pass over the category's labels,
    tallying every rule at once (uncapped) — feeds the tree-node badges. The
    fired rule is re-derived identically (re-apply the current tree to each
    label's evidence). Keys are rule indices; key -1 collects labels where no
    rule matched (the default). Missing `{cat}.rules` node -> {}."""
    tree = _category_tree(category)
    if tree is None:
        return {}
    rules, default, order = tree

    sql = text(
        """
        SELECT l.evidence_attributes AS evidence
        FROM labels l
        WHERE l.category = :category
        """
    )
    counts: dict[int, int] = {}
    with SessionLocal() as sess:
        rows = sess.execute(sql, {"category": category}).mappings()
        for r in rows:
            values = {
                a.get("key"): a.get("value")
                for a in (r["evidence"] or [])
                if isinstance(a, dict) and a.get("key") and a.get("value") != ""
            }
            _score, matched = decision_tree._apply_decision_tree(
                rules, default, values, order)
            idx = rules.index(matched) if matched is not None else -1
            counts[idx] = counts.get(idx, 0) + 1
    return counts


def segments_for_attribute_value(
    category: str, attribute: str, value: str,
) -> list[dict]:
    """Segments whose label recorded `attribute == value` for `category`. Matches
    any evidence_attributes entry with that key/value, compared as text (`->>`)
    so numeric/bool values coerce. Empty value matches nothing. Capped at
    RESULT_CAP, ordered by id."""
    if value == "":
        return []
    sql = text(
        """
        SELECT l.segment_id AS segment_id, s.video_id AS video_id,
               l.score AS score
        FROM labels l
        JOIN segments s ON s.segment_id = l.segment_id
        WHERE l.category = :category
          AND EXISTS (
              SELECT 1
              FROM jsonb_array_elements(l.evidence_attributes) e
              WHERE e->>'key' = :attribute AND e->>'value' = :value
          )
        ORDER BY l.label_id
        LIMIT :cap
        """
    )
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "category": category,
                "attribute": attribute,
                "value": value,
                "cap": RESULT_CAP,
            },
        ).mappings()
        return [_seg_row(r["segment_id"], r["video_id"], r["score"]) for r in rows]


def attribute_value_examples(
    category: str, attribute: str, value: str, limit: int = 3,
) -> list[dict]:
    """Up to `limit` representative segments labelled with `attribute == value`,
    each `{segment_id, video_id, summary, score}` — reuses
    `segments_for_attribute_value` then reads each segment's ingestion `summary`
    for display alongside a keyframe. Read-only. Empty value -> []."""
    if value == "" or limit <= 0:
        return []
    segs = segments_for_attribute_value(category, attribute, value)[:limit]
    if not segs:
        return []
    ids = [s["segment_id"] for s in segs]
    sql = text(
        "SELECT segment_id, summary FROM segments WHERE segment_id = ANY(:ids)"
    )
    with SessionLocal() as sess:
        summaries = {
            r["segment_id"]: r["summary"]
            for r in sess.execute(sql, {"ids": ids}).mappings()
        }
    return [
        {
            "segment_id": s["segment_id"],
            "video_id": s["video_id"],
            "summary": summaries.get(s["segment_id"]),
            "score": s["score"],
        }
        for s in segs
    ]
