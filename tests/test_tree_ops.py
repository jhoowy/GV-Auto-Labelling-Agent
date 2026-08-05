"""Decision-tree NODE-op tests (DB-free).

Covers the pure `policy_store.apply_tree_op` — the only mutations the
post-bootstrap agent may request on a `{category}.rules` node (add / modify /
delete a rule) — plus the REVIEW parse that produces those ops in the graph.
`apply_tree_op` never touches the DB and preserves `default`; invalid ops are
no-ops so a bad request can never corrupt the tree."""
from labelling.graph import _clean_tree_op, _parse_reviews
from tools.policy_store import apply_tree_op

_R0 = {"when": [{"attribute": "nudity", "op": "==", "value": "partial"}],
       "score": 3, "note": "partial nudity"}
_R1 = {"when": [{"attribute": "nudity", "op": "==", "value": "full"}],
       "score": 4, "note": "full nudity"}


_DEFAULT_WHEN = [{"attribute": "intensity", "op": ">=", "value": "high"}]
_UNSET = object()


def _add(when=_UNSET, score=2, note="new", rule_index=None):
    return {"op": "add", "category": "sex", "rule_index": rule_index,
            "when": _DEFAULT_WHEN if when is _UNSET else when,
            "score": score, "note": note}


# --- add --------------------------------------------------------------------

def test_add_appends_when_index_none():
    rules, default = apply_tree_op([_R0], 0, _add())
    assert default == 0
    assert len(rules) == 2 and rules[0] == _R0
    assert rules[1]["score"] == 2 and rules[1]["note"] == "new"
    assert rules[1]["when"] == _DEFAULT_WHEN


def test_add_inserts_at_index():
    rules, _ = apply_tree_op([_R0, _R1], 0, _add(rule_index=1, note="mid"))
    assert [r.get("note") for r in rules] == ["partial nudity", "mid", "full nudity"]


def test_add_at_tail_index_allowed():
    rules, _ = apply_tree_op([_R0], 0, _add(rule_index=1, note="tail"))
    assert len(rules) == 2 and rules[1]["note"] == "tail"


def test_add_on_empty_creates_single_rule():
    rules, default = apply_tree_op([], 0, _add(note="first"))
    assert len(rules) == 1 and rules[0]["note"] == "first"
    assert default == 0


def test_add_out_of_bounds_index_is_noop():
    rules, _ = apply_tree_op([_R0], 0, _add(rule_index=5))
    assert rules == [_R0]


def test_add_malformed_when_is_noop():
    assert apply_tree_op([_R0], 0, _add(when=[]))[0] == [_R0]          # empty when
    assert apply_tree_op([_R0], 0, _add(when="nope"))[0] == [_R0]      # not a list
    assert apply_tree_op([_R0], 0, _add(when=[{"op": "=="}]))[0] == [_R0]  # no attribute


def test_add_missing_score_is_noop():
    op = _add()
    op.pop("score")
    assert apply_tree_op([_R0], 0, op)[0] == [_R0]


# --- modify -----------------------------------------------------------------

def test_modify_replaces_at_index():
    rules, default = apply_tree_op(
        [_R0, _R1], 2,
        {"op": "modify", "category": "sex", "rule_index": 0,
         "when": [{"attribute": "nudity", "op": "==", "value": "none"}],
         "score": 1, "note": "revised"})
    assert default == 2                       # default preserved
    assert rules[0] == {"when": [{"attribute": "nudity", "op": "==", "value": "none"}],
                        "score": 1, "note": "revised"}
    assert rules[1] == _R1                     # sibling untouched


def test_modify_out_of_bounds_is_noop():
    op = {"op": "modify", "category": "sex", "rule_index": 3,
          "when": _R0["when"], "score": 1, "note": "x"}
    assert apply_tree_op([_R0], 0, op)[0] == [_R0]


def test_modify_none_index_is_noop():
    op = {"op": "modify", "category": "sex", "rule_index": None,
          "when": _R0["when"], "score": 1, "note": "x"}
    assert apply_tree_op([_R0], 0, op)[0] == [_R0]


# --- delete -----------------------------------------------------------------

def test_delete_removes_at_index():
    rules, default = apply_tree_op([_R0, _R1], 5, {"op": "delete", "rule_index": 0})
    assert rules == [_R1] and default == 5


def test_delete_out_of_bounds_is_noop():
    assert apply_tree_op([_R0], 0, {"op": "delete", "rule_index": 9})[0] == [_R0]
    assert apply_tree_op([_R0], 0, {"op": "delete", "rule_index": None})[0] == [_R0]


# --- misc -------------------------------------------------------------------

def test_unknown_op_and_non_dict_are_noops():
    assert apply_tree_op([_R0], 0, {"op": "nuke"})[0] == [_R0]
    assert apply_tree_op([_R0], 0, "not-a-dict")[0] == [_R0]


def test_input_rules_not_mutated_in_place():
    original = [_R0]
    apply_tree_op(original, 0, _add())
    assert original == [_R0]   # caller's list is untouched (a copy is returned)


# --- REVIEW parse produces the constrained op -------------------------------

def test_clean_tree_op_only_keeps_the_three_ops():
    assert _clean_tree_op({"op": "add", "when": [], "score": 2}, "sex")["op"] == "add"
    assert _clean_tree_op({"op": "rubric_edit"}, "sex") is None      # not a node op
    assert _clean_tree_op("nope", "sex") is None
    # category is set authoritatively from the loop, never trusted from the model
    op = _clean_tree_op({"op": "delete", "category": "gambling", "rule_index": 1}, "sex")
    assert op == {"op": "delete", "category": "sex", "rule_index": 1}


def test_parse_reviews_attaches_op_only_when_flagged():
    parsed = {"reviews": [
        {"category": "sex", "needs_change": True,
         "op": {"op": "add", "rule_index": None,
                "when": [{"attribute": "nudity", "op": "==", "value": "full"}],
                "score": 4, "note": "n"}},
        {"category": "gambling", "needs_change": False,
         "op": {"op": "delete", "rule_index": 0}},   # ignored: not flagged
    ]}
    out = _parse_reviews(parsed, ["sex", "gambling"])
    assert out["sex"]["op"]["op"] == "add" and out["sex"]["op"]["category"] == "sex"
    assert out["gambling"]["op"] is None
