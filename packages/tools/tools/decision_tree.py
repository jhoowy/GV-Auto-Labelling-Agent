"""Deterministic decision-tree evaluation — pure functions, no DB/model deps.

The single source of truth for how a category's priority-ordered rule tree turns
extracted attribute values into a score. Shared by the labelling agent
(`labelling/graph.py`, which re-exports these) and by node->segment tracking
(`tools.tracking`, which re-derives the fired rule from a label's stored
`evidence_attributes` instead of a per-label trace).

The rule/tree storage format (`{when, score, note, ...}` + `default`) and the
first-fully-matching-rule-wins semantics must stay identical across both callers.
"""
from __future__ import annotations

from schemas.enums import SCORE_MAX, SCORE_MIN


def _clamp(score) -> int:
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = SCORE_MIN
    return max(SCORE_MIN, min(SCORE_MAX, s))


def _as_num(x):
    """Coerce to float for ordered comparison; bools/non-numerics -> None."""
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1")
    if isinstance(x, (int, float)):
        return bool(x)
    return None


def _val_eq(a, b) -> bool:
    """Type-robust equality: bool vs "true"/1, numeric vs "3", else case-fold str."""
    if isinstance(a, bool) or isinstance(b, bool):
        ba, bb = _as_bool(a), _as_bool(b)
        return ba is not None and ba == bb
    na, nb = _as_num(a), _as_num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a).strip().lower() == str(b).strip().lower()


def _ordinal_rank(keys: list, val) -> int | None:
    """Position of `val` in an ascending ordinal value list (case-insensitive)."""
    target = str(val).strip().lower()
    for i, k in enumerate(keys):
        if str(k).strip().lower() == target:
            return i
    return None


def _match_cond(cond: dict, values: dict, order: dict | None = None) -> bool:
    """One decision-tree condition against the extracted attribute values.

    `order` maps an ordinal attribute name to its ascending value keys, so `>=`
    / `<=` on a non-numeric ordinal compares by position in that order."""
    attr = cond.get("attribute")
    op = cond.get("op")
    target = cond.get("value")
    present = attr in values and values.get(attr) is not None
    if op == "present":
        return present
    if not present:
        return False
    v = values.get(attr)
    if op == "==":
        return _val_eq(v, target)
    if op in (">=", "<="):
        a, b = _as_num(v), _as_num(target)
        if a is None or b is None:  # non-numeric -> fall back to ordinal ranking
            keys = (order or {}).get(attr)
            if not keys:
                return False
            a, b = _ordinal_rank(keys, v), _ordinal_rank(keys, target)
            if a is None or b is None:
                return False
        return a >= b if op == ">=" else a <= b
    if op == "in":
        opts = target if isinstance(target, (list, tuple, set)) else [target]
        return any(_val_eq(v, o) for o in opts)
    return False


def _apply_decision_tree(rules: list, default: int, values: dict,
                         order: dict | None = None) -> tuple[int, dict | None]:
    """Evaluate a priority-ordered decision tree against extracted attribute
    values. First rule whose every `when` condition matches wins; otherwise the
    default. `order` supplies ascending value keys for ordinal attributes so
    rules can compare with `>=`. Returns (clamped score, matched rule | None)."""
    for rule in rules or []:
        conds = rule.get("when") or []
        if all(_match_cond(c, values, order) for c in conds):
            return _clamp(rule.get("score", default)), rule
    return _clamp(default), None
