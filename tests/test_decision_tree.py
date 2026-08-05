"""Shared decision-tree module (DB-free).

Covers the re-export contract (graph.py exposes the same names it used to define)
and the evidence_attributes -> values -> fired-rule-index derivation that
`tools.tracking.segments_for_rule` now uses in place of a stored per-label
tool_trace.
"""
from tools.decision_tree import _apply_decision_tree, _match_cond


def test_reexport_is_the_same_object():
    # graph.py must keep exposing the moved names for its own callers + tests.
    from labelling.graph import _apply_decision_tree as g_apply
    from labelling.graph import _match_cond as g_match
    assert g_apply is _apply_decision_tree
    assert g_match is _match_cond


def test_first_fully_matching_rule_wins():
    rules = [
        {"when": [{"attribute": "bet", "op": "==", "value": True}],
         "score": 4, "note": "real-money bet"},
        {"when": [{"attribute": "sim", "op": "==", "value": True}],
         "score": 2, "note": "simulated"},
    ]
    # both conditions true -> the FIRST rule wins (priority order).
    score, matched = _apply_decision_tree(
        rules, default=0, values={"bet": True, "sim": True})
    assert score == 4 and rules.index(matched) == 0
    # only the second holds -> rule #1.
    score, matched = _apply_decision_tree(
        rules, default=0, values={"sim": True})
    assert score == 2 and rules.index(matched) == 1
    # neither -> default, no rule.
    score, matched = _apply_decision_tree(rules, default=0, values={})
    assert score == 0 and matched is None


def _rule_index_for_evidence(evidence, rules, default, order=None):
    """The per-label derivation `segments_for_rule` performs: strip empty
    evidence entries to `values`, re-apply the tree, return the fired rule index
    (or None). Kept DB-free here so the mapping can be unit-tested directly."""
    values = {a["key"]: a["value"] for a in evidence
              if a.get("key") and a.get("value") != ""}
    _score, matched = _apply_decision_tree(rules, default, values, order)
    return None if matched is None else rules.index(matched)


def test_evidence_to_rule_index_ignores_empty_values():
    rules = [{"when": [{"attribute": "nudity", "op": "==", "value": "partial"}],
              "score": 3, "note": "partial nudity"}]
    # unselected attrs (value == "") drop out; the selected one fires rule #0.
    evidence = [{"key": "nudity", "value": "partial"},
                {"key": "intensity", "value": ""}]
    assert _rule_index_for_evidence(evidence, rules, 0) == 0
    # all empty -> nothing fires.
    assert _rule_index_for_evidence(
        [{"key": "nudity", "value": ""}], rules, 0) is None


def test_evidence_to_rule_index_ordinal_order():
    # ordinal >= compares by position in the ascending order map.
    rules = [{"when": [{"attribute": "intensity", "op": ">=", "value": "high"}],
              "score": 5, "note": "high intensity"}]
    order = {"intensity": ["low", "medium", "high"]}
    hi = [{"key": "intensity", "value": "high"}]
    lo = [{"key": "intensity", "value": "low"}]
    assert _rule_index_for_evidence(hi, rules, 0, order) == 0
    assert _rule_index_for_evidence(lo, rules, 0, order) is None
