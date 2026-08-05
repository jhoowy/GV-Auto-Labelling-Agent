"""Learn a category's decision tree from bootstrap labels via CART (sklearn).

Bootstrap labelling (`labelling.graph._judge_bootstrap_attrs`) emits, per shot,
the FULL extracted attribute vector plus a FREE (non-tree) score — that is the
training set. Here we integer-encode the attributes, fit a shallow
`DecisionTreeClassifier`, and CONVERT it into the existing priority-rules
decision-tree format consumed by `graph._apply_decision_tree` (and the
vertical-cascade UI). The storage format is unchanged, so the JUDGE keeps
working: only the rules' PROVENANCE changes (learned from data, not GPT-guessed).

Read-only except the final `policy_store.upsert_decision_rule`. The sklearn ->
rules conversion (`_tree_to_rules`) is a pure function so it is unit-testable
without a DB.
"""
from __future__ import annotations

import logging
import math

from schemas.enums import SCORE_MAX, SCORE_MIN

log = logging.getLogger(__name__)

# One integer feature per attribute. Missing / empty / unknown value -> a
# reserved code that is distinct from every real value's code.
MISSING = -1

# Guards: too few rows or a single class -> a learned tree is meaningless.
_MIN_ROWS = 8


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def _clamp(score) -> int:
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = SCORE_MIN
    return max(SCORE_MIN, min(SCORE_MAX, s))


class _Encoder:
    """Encode ONE attribute's value as an integer code and decode back.

    - ordinal   -> ascending rank (values listed low..high -> 0,1,2,...).
    - categorical -> a stable code map (value's position in the def's list).
    - boolean   -> false=0, true=1.
    Missing / empty / unknown -> MISSING. `value_for_code` returns the canonical
    value for a code so the code sets a tree split implies translate back to the
    exact VALUES `_apply_decision_tree` compares against."""

    def __init__(self, name: str, value_type: str, values):
        self.name = name
        self.value_type = value_type
        self._code: dict[str, int] = {}
        self._value: dict[int, object] = {MISSING: ""}
        if value_type == "boolean":
            self._code = {"false": 0, "true": 1}
            self._value[0] = False
            self._value[1] = True
            return
        for i, v in enumerate(values or []):
            val = v.get("value") if isinstance(v, dict) else v
            if val is None:
                continue
            key = str(val).strip().lower()
            if key in self._code:
                continue
            self._code[key] = i
            self._value[i] = val

    def code(self, raw) -> int:
        if raw is None:
            return MISSING
        if isinstance(raw, bool):
            return self._code.get("true" if raw else "false", MISSING)
        s = str(raw).strip()
        if not s:
            return MISSING
        return self._code.get(s.lower(), MISSING)

    def value_for_code(self, c: int):
        return self._value.get(c, "")


def _attr_name(policy_id: str) -> str:
    return policy_id.split(".attr.", 1)[1] if ".attr." in policy_id else policy_id


def _encoders_from_defs(defs) -> tuple[list[str], dict[str, _Encoder]]:
    """Build the fixed feature order + per-attribute encoders from a category's
    ATTRIBUTE-def policy nodes (DB-free — reads only `structured_data`)."""
    feature_order: list[str] = []
    encoders: dict[str, _Encoder] = {}
    for d in defs:
        sd = d.structured_data or {}
        if sd.get("kind") != "attribute_def":
            continue
        name = _attr_name(d.policy_id)
        feature_order.append(name)
        encoders[name] = _Encoder(
            name, sd.get("value_type", "categorical"), sd.get("values"))
    return feature_order, encoders


# --------------------------------------------------------------------------- #
# sklearn tree -> priority rules  (pure, unit-tested without a DB)
# --------------------------------------------------------------------------- #
def _tree_to_rules(clf, feature_order, encoders, X) -> tuple[list[dict], int]:
    """Convert a fitted `DecisionTreeClassifier` into the priority-rules format
    (`{"when":[...],"score":int,"note":str}`) that `_apply_decision_tree` runs.

    Each root->leaf path is a conjunction of integer splits `feature_i <= thr`.
    Because feature_i is attribute A's integer code, the branch is `A's code
    <= floor(thr)` (left) or `> floor(thr)` (right); we intersect the allowed
    code set across every split on A along the path. At the leaf, each still-
    constrained attribute becomes one `in` condition over the translated VALUES
    (or `==` when a single value survives); an attribute whose allowed set spans
    all observed codes is unconstrained and dropped. The leaf's majority class is
    the rule score. Tree leaves partition the sample space, so the rules are
    mutually exclusive — order is not semantically load-bearing; we still sort by
    descending sample count for readability. `default` = the root majority class.

    `X` (the training feature matrix) supplies each feature's observed-code
    universe, so an `in` set only ever lists values that actually occur."""
    tree = clf.tree_
    n_features = len(feature_order)
    universe = {i: {row[i] for row in X} for i in range(n_features)}

    leaves: list[tuple[dict[int, set], int]] = []

    def recurse(node: int, allowed: dict[int, set]) -> None:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:  # leaf
            leaves.append((allowed, node))
            return
        f = tree.feature[node]
        cut = math.floor(tree.threshold[node])  # integer boundary; <= cut -> left
        la = dict(allowed)
        la[f] = {c for c in allowed[f] if c <= cut}
        ra = dict(allowed)
        ra[f] = {c for c in allowed[f] if c > cut}
        recurse(left, la)
        recurse(right, ra)

    recurse(0, {i: set(universe[i]) for i in range(n_features)})

    scored: list[dict] = []
    for allowed, leaf in leaves:
        n = int(tree.n_node_samples[leaf])
        if n <= 0:
            continue
        conds: list[dict] = []
        impossible = False
        for i, name in enumerate(feature_order):
            s = allowed[i]
            if s == universe[i]:  # unconstrained on this attribute
                continue
            if not s:  # path admits no observed code -> unreachable leaf
                impossible = True
                break
            vals = [encoders[name].value_for_code(c) for c in sorted(s)]
            if len(vals) == 1:
                conds.append({"attribute": name, "op": "==", "value": vals[0]})
            else:
                conds.append({"attribute": name, "op": "in", "value": vals})
        if impossible:
            continue
        score = int(clf.classes_[int(tree.value[leaf][0].argmax())])
        scored.append({"n": n, "when": conds, "score": score,
                       "note": f"CART leaf (n={n})"})

    scored.sort(key=lambda r: -r["n"])
    rules = [{"when": r["when"], "score": r["score"], "note": r["note"]}
             for r in scored]
    default = int(clf.classes_[int(tree.value[0][0].argmax())])
    return rules, default


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def bootstrap_train_video_ids() -> list[str]:
    """Video ids tagged `metadata_json.split == 'bootstrap_train'` — the training
    split held apart from the eval videos. Pass to `train_all(video_ids=...)`."""
    from db import SessionLocal
    from sqlalchemy import text

    with SessionLocal() as s:
        rows = s.execute(
            text("select video_id from videos "
                 "where metadata_json->>'split' = 'bootstrap_train'")
        ).all()
    return [r[0] for r in rows]


def _segments_for_videos(video_ids) -> set[str]:
    """Segment ids belonging to the given videos (to scope training to a split,
    e.g. the 100 `bootstrap_train` videos held apart from the eval set)."""
    from db import SessionLocal
    from sqlalchemy import text

    with SessionLocal() as s:
        rows = s.execute(
            text("select segment_id from segments where video_id = any(:v)"),
            {"v": list(video_ids)},
        ).all()
    return {r[0] for r in rows}


def _gather_rows(category, feature_order, encoders, video_ids=None):
    """Bootstrap-label training rows for a category: each attribute-path label ->
    (integer-encoded attribute vector, clamped score). Only labels carrying the
    defined attributes in `evidence_attributes` (i.e. the bootstrap attr path,
    not pure holistic) qualify. When `video_ids` is given, only labels from those
    videos' segments are used (training set / split scoping). Read-only."""
    from tools import storage

    cat = getattr(category, "value", category)
    fset = set(feature_order)
    allowed = _segments_for_videos(video_ids) if video_ids is not None else None
    X: list[list[int]] = []
    y: list[int] = []
    for lbl in storage.list_labels():
        if getattr(lbl.category, "value", lbl.category) != cat:
            continue
        if allowed is not None and lbl.segment_id not in allowed:
            continue
        ev = {a.key: a.value for a in lbl.evidence_attributes}
        if not (fset & set(ev)):  # not an attribute-path label -> skip
            continue
        X.append([encoders[n].code(ev.get(n)) for n in feature_order])
        y.append(_clamp(lbl.score))
    return X, y


def train_decision_tree(category, video_ids=None) -> dict:
    """Learn one category's decision tree from bootstrap labels and upsert it.

    `video_ids` (optional) scopes training to those videos' labels — pass the
    `bootstrap_train` split so the held-out eval videos never leak into the tree.

    Gathers attribute-vector -> score rows, fits
    `DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=0)`,
    converts it to the priority-rules format, and upserts it via
    `policy_store.upsert_decision_rule`. Skips (logs, no write) when the category
    has no attribute defs, too few rows, or a single class. Returns a summary
    dict with a `status`."""
    from sklearn.tree import DecisionTreeClassifier

    from tools import policy_store

    cat = getattr(category, "value", category)
    defs = [n for n in policy_store.get_policy_tree(cat)
            if (n.structured_data or {}).get("kind") == "attribute_def"]
    if not defs:
        log.info("tree_train: %s has no attribute defs; skipping", cat)
        return {"category": cat, "status": "no_attributes"}

    feature_order, encoders = _encoders_from_defs(defs)
    X, y = _gather_rows(cat, feature_order, encoders, video_ids)

    if len(X) < _MIN_ROWS:
        log.info("tree_train: %s has %d row(s) (<%d); skipping",
                 cat, len(X), _MIN_ROWS)
        return {"category": cat, "status": "insufficient_data", "n_rows": len(X)}
    if len(set(y)) < 2:
        log.info("tree_train: %s rows are a single class %s; skipping",
                 cat, y[0])
        return {"category": cat, "status": "single_class",
                "n_rows": len(X), "score": y[0]}

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=0)
    clf.fit(X, y)
    rules, default = _tree_to_rules(clf, feature_order, encoders, X)
    policy_store.upsert_decision_rule(cat, rules, default)

    log.info("tree_train: learned %d rule(s) for %s from %d row(s)",
             len(rules), cat, len(X))

    # Best-effort: author human-readable EN + KO descriptions for the learned
    # rules. A description failure (e.g. no OPENAI_API_KEY) must NEVER fail
    # training, so the tree is kept as-is (undescribed) and `train_all` still
    # works fully offline.
    try:
        from tools.tree_describe import describe_rules

        describe_rules(cat)
    except Exception as e:  # noqa: BLE001 - descriptions are best-effort
        log.warning("tree_train: describe_rules failed for %s (%s); "
                    "tree kept without descriptions", cat, e)
    return {"category": cat, "status": "trained", "n_rows": len(X),
            "n_rules": len(rules), "default": default,
            "attributes": feature_order}


def train_all(categories=None, video_ids=None) -> dict:
    """Learn decision trees for every active category (or the given subset).

    Categories default to the policy config's active set. `video_ids` scopes
    training to a video split (e.g. the `bootstrap_train` set) for every
    category. One category's failure is logged and recorded, never aborting the
    run. Returns {category: summary}."""
    if categories is None:
        from schemas.enums import Category

        from models import base_config
        categories = ((base_config().get("policy", {}) or {}).get("categories")
                      or [c.value for c in Category])
    out: dict = {}
    for c in categories:
        cat = getattr(c, "value", c)
        try:
            out[cat] = train_decision_tree(cat, video_ids)
        except Exception as e:  # noqa: BLE001 - one category must not abort the run
            log.warning("tree_train: training failed for %s (%s)", cat, e)
            out[cat] = {"category": cat, "status": "error", "error": str(e)}
    return out
