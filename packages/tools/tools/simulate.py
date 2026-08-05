"""What-if simulation for a decision-tree rule change (read-only, #41).

Given a proposed decision-tree NODE op (the same `{op, rule_index, when, score,
note}` shape `policy_store.apply_tree_op` consumes), compute how the category's
metrics would change BEFORE anything is committed. Nothing is written: the tree
is modified only in memory via the pure `apply_tree_op`, every label is re-scored
under BOTH the current and the modified tree with the shared
`decision_tree._apply_decision_tree`, and a compact before/after summary is
returned.

Scoring/summary logic is factored into `_summarize` (pure, no DB) so it is
unit-testable on synthetic value dicts; the DB-facing `simulate_rule_change`
only loads the tree + the scoped labels and delegates to it.
"""
from __future__ import annotations

from sqlalchemy import text

from . import decision_tree, policy_store, tracking

# How many old->new score changes to echo back as concrete examples.
_MAX_EXAMPLES = 10


def _score_dist(scores) -> dict[int, int]:
    """Score -> count histogram over an iterable of scores."""
    dist: dict[int, int] = {}
    for s in scores:
        dist[s] = dist.get(s, 0) + 1
    return dist


def _summarize(records, current_rules, current_default,
               new_rules, new_default, order=None) -> dict:
    """Pure before/after summary over pre-built label records — no DB.

    Each record is `{"segment_id": str, "values": dict, "gt_present": bool|None}`
    (`gt_present` = ground-truth "category present", or None when unknown). Scores
    each record under the current tree and the modified tree, then reports the
    score distribution before/after, how many segments changed score (with a few
    examples), and GT agreement (fraction of GT-carrying segments whose predicted
    present == score>0 matches GT) for each tree."""
    cur_scores: list[int] = []
    new_scores: list[int] = []
    n_changed = 0
    changed_examples: list[dict] = []
    # GT agreement counters: hits over the segments that carry a GT `present`.
    n_gt = cur_gt_hits = new_gt_hits = 0

    for rec in records:
        values = rec.get("values") or {}
        cur, _ = decision_tree._apply_decision_tree(
            current_rules, current_default, values, order)
        new, _ = decision_tree._apply_decision_tree(
            new_rules, new_default, values, order)
        cur_scores.append(cur)
        new_scores.append(new)
        if cur != new:
            n_changed += 1
            if len(changed_examples) < _MAX_EXAMPLES:
                changed_examples.append(
                    {"segment_id": rec.get("segment_id"), "old": cur, "new": new})

        gt_present = rec.get("gt_present")
        if gt_present is not None:
            n_gt += 1
            cur_gt_hits += int(bool(gt_present) == (cur > 0))
            new_gt_hits += int(bool(gt_present) == (new > 0))

    return {
        "n_segments": len(records),
        "current_score_dist": _score_dist(cur_scores),
        "new_score_dist": _score_dist(new_scores),
        "n_changed": n_changed,
        "changed_examples": changed_examples,
        "n_gt": n_gt,
        "gt_agreement_current": (cur_gt_hits / n_gt) if n_gt else None,
        "gt_agreement_new": (new_gt_hits / n_gt) if n_gt else None,
    }


def _load_records(category: str, scope: str) -> list[dict]:
    """Load the category's labels in `scope` as `_summarize` records (read-only).

    `scope` is a `metadata_json.split` value (default the `bootstrap_train` set);
    `"all"` takes every split. For each label, `values` is built from its non-empty
    `evidence_attributes` (mirrors `tracking`), and `gt_present` is read from the
    video's `metadata_json.category_gt[category].present` when present."""
    from db import SessionLocal

    sql = text(
        """
        SELECT l.segment_id AS segment_id,
               l.evidence_attributes AS evidence,
               v.metadata_json AS meta
        FROM labels l
        JOIN segments s ON s.segment_id = l.segment_id
        JOIN videos v ON v.video_id = s.video_id
        WHERE l.category = :category
          AND (:scope = 'all' OR v.metadata_json->>'split' = :scope)
        ORDER BY l.label_id
        """
    )
    records: list[dict] = []
    with SessionLocal() as sess:
        for r in sess.execute(sql, {"category": category, "scope": scope}).mappings():
            values = {
                a.get("key"): a.get("value")
                for a in (r["evidence"] or [])
                if isinstance(a, dict) and a.get("key") and a.get("value") != ""
            }
            meta = r["meta"] or {}
            gt = (meta.get("category_gt") or {}).get(category) or {}
            gt_present = gt.get("present") if isinstance(gt, dict) else None
            records.append({
                "segment_id": r["segment_id"],
                "values": values,
                "gt_present": gt_present,
            })
    return records


def simulate_rule_change(category: str, op: dict,
                         scope: str = "bootstrap_train") -> dict:
    """Preview a decision-tree rule change's metric impact WITHOUT committing it.

    Loads `category`'s current tree (rules, default) + ordinal `order` map, derives
    the modified tree via `policy_store.apply_tree_op(rules, default, op)`, and
    re-scores every label in `scope` (a `metadata_json.split`; default
    `bootstrap_train`, or `"all"`) under both trees. Returns the before/after
    summary from `_summarize`: `n_segments`, `current_score_dist`/`new_score_dist`,
    `n_changed` (+ a few `changed_examples`), and GT agreement before/after. Purely
    read-only — the tree is never persisted (that stays a SIDE_FX/human step)."""
    tree = tracking._category_tree(category)
    if tree is None:
        # No decision-tree node yet -> nothing to simulate against.
        return {"category": category, "status": "no_decision_tree",
                "n_segments": 0}
    rules, default, order = tree
    new_rules, new_default = policy_store.apply_tree_op(rules, default, op)

    records = _load_records(category, scope)
    summary = _summarize(records, rules, default, new_rules, new_default, order)
    summary["category"] = category
    summary["scope"] = scope
    # Surface whether the op actually changed the tree (invalid ops no-op).
    summary["tree_changed"] = (new_rules != rules) or (new_default != default)
    return summary
