"""What-if rule-change simulation tests (DB-free, #41).

Exercises the pure `simulate._summarize` — re-score every label record under the
current vs. the modified tree and report the before/after delta — plus the real
`policy_store.apply_tree_op` that produces the modified tree, so an add/delete op
is simulated exactly as production would. No DB: records are supplied as
pre-built value dicts, mirroring what `simulate._load_records` yields."""
from tools.policy_store import apply_tree_op
from tools.simulate import _summarize

# A one-rule gambling-ish tree: a "loot box present" shot scores 3, else default 0.
_RULES = [{"when": [{"attribute": "loot", "op": "==", "value": "present"}],
           "score": 3, "note": "loot box present"}]
_DEFAULT = 0


def _rec(seg, values, gt_present=None):
    return {"segment_id": seg, "values": values, "gt_present": gt_present}


# loot-present segments score 3 under _RULES; the rest fall through to 0.
_RECORDS = [
    _rec("s1", {"loot": "present"}),
    _rec("s2", {"loot": "present"}),
    _rec("s3", {}),                     # default
    _rec("s4", {"other": "x"}),         # default
]


def test_delete_op_changes_n_segments():
    # Deleting rule 0 collapses the whole tree to the default -> the two
    # loot-present segments drop 3 -> 0; the two already-default ones are unchanged.
    new_rules, new_default = apply_tree_op(_RULES, _DEFAULT,
                                           {"op": "delete", "rule_index": 0})
    out = _summarize(_RECORDS, _RULES, _DEFAULT, new_rules, new_default)
    assert out["n_segments"] == 4
    assert out["n_changed"] == 2
    changed = {(e["segment_id"], e["old"], e["new"]) for e in out["changed_examples"]}
    assert changed == {("s1", 3, 0), ("s2", 3, 0)}


def test_add_op_pulls_segments_to_new_score():
    # Append a rule scoring `other == x` at 5; only s4 matches it (s1/s2 still fire
    # the earlier loot rule first -> first-match-wins is preserved).
    op = {"op": "add", "rule_index": None,
          "when": [{"attribute": "other", "op": "==", "value": "x"}],
          "score": 5, "note": "new"}
    new_rules, new_default = apply_tree_op(_RULES, _DEFAULT, op)
    out = _summarize(_RECORDS, _RULES, _DEFAULT, new_rules, new_default)
    assert out["n_changed"] == 1
    assert out["changed_examples"] == [{"segment_id": "s4", "old": 0, "new": 5}]


def test_score_distribution_before_and_after():
    new_rules, new_default = apply_tree_op(_RULES, _DEFAULT,
                                           {"op": "delete", "rule_index": 0})
    out = _summarize(_RECORDS, _RULES, _DEFAULT, new_rules, new_default)
    assert out["current_score_dist"] == {3: 2, 0: 2}
    assert out["new_score_dist"] == {0: 4}  # everything collapses to default


def test_gt_agreement_before_and_after():
    # GT `present` truth: s1/s2 truly present, s3/s4 truly absent. The current
    # tree matches all four (2 present via score 3, 2 absent via 0) -> 1.0.
    records = [
        _rec("s1", {"loot": "present"}, gt_present=True),
        _rec("s2", {"loot": "present"}, gt_present=True),
        _rec("s3", {}, gt_present=False),
        _rec("s4", {"other": "x"}, gt_present=False),
    ]
    # Deleting the only rule makes s1/s2 read as absent (score 0) -> 2 of 4 wrong.
    new_rules, new_default = apply_tree_op(_RULES, _DEFAULT,
                                           {"op": "delete", "rule_index": 0})
    out = _summarize(records, _RULES, _DEFAULT, new_rules, new_default)
    assert out["n_gt"] == 4
    assert out["gt_agreement_current"] == 1.0
    assert out["gt_agreement_new"] == 0.5


def test_gt_agreement_none_when_no_gt():
    # No record carries GT -> agreement is undefined (None), not a crash.
    out = _summarize(_RECORDS, _RULES, _DEFAULT, _RULES, _DEFAULT)
    assert out["n_gt"] == 0
    assert out["gt_agreement_current"] is None
    assert out["gt_agreement_new"] is None
    assert out["n_changed"] == 0  # identical trees -> nothing changes
