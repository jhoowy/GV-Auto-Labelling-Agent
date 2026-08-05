"""CART -> priority-rules conversion (DB-free).

Builds a tiny synthetic labelled set, fits a sklearn DecisionTreeClassifier,
converts it with `tree_train._tree_to_rules`, and asserts:
  (a) reversibility — the produced rules, applied via `_apply_decision_tree`,
      reproduce the tree's prediction on every training row (including a MISSING
      value, which round-trips through the reserved code + "" sentinel), and
  (b) the emitted `in`/`==` conditions are well-formed.
Also import-checks the new module + `graph.build_graph()`.
"""
from sklearn.tree import DecisionTreeClassifier

from labelling.graph import _apply_decision_tree, build_graph
from schemas import Policy
from schemas.enums import Category, PolicyType
from tools import tree_train
from tools.tree_train import _encoders_from_defs, _tree_to_rules


def _def(cat, name, value_type, values):
    return Policy(
        policy_id=f"{cat}.attr.{name}", type=PolicyType.ATTRIBUTE,
        category=Category(cat), version=1, parent_id=f"{cat}.scoring",
        text=f"Attribute '{name}' for {cat}.",
        structured_data={"kind": "attribute_def", "value_type": value_type,
                         "values": values, "guidelines": "detect it"})


def _synthetic():
    """A gambling-shaped set: `activity` (categorical none/simulated/real) is the
    strong signal, `stakes` (ordinal low/high) refines the `real` branch. Two
    `real` rows have a MISSING stakes value to exercise the reserved code."""
    defs = [
        _def("gambling", "activity", "categorical",
             [{"value": "none"}, {"value": "simulated"}, {"value": "real"}]),
        _def("gambling", "stakes", "ordinal",
             [{"value": "low"}, {"value": "high"}]),
    ]
    # (activity, stakes) raw values -> score
    rows = (
        [("none", "low", 0)] * 3 + [("none", "high", 0)] * 2
        + [("simulated", "low", 0)] * 3 + [("simulated", "high", 0)] * 2
        + [("real", "low", 4)] * 3 + [("real", "high", 5)] * 3
        + [("real", None, 4)] * 2          # MISSING stakes -> reserved code
    )
    return defs, rows


def _fit(defs, rows):
    feature_order, encoders = _encoders_from_defs(defs)
    X = [[encoders["activity"].code(a), encoders["stakes"].code(s)]
         for (a, s, _score) in rows]
    y = [score for (*_a, score) in rows]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=0)
    clf.fit(X, y)
    return feature_order, encoders, X, y, clf


def test_rules_reproduce_tree_predictions_on_training_rows():
    defs, rows = _synthetic()
    feature_order, encoders, X, y, clf = _fit(defs, rows)
    rules, default = _tree_to_rules(clf, feature_order, encoders, X)

    for row in X:
        pred = int(clf.predict([row])[0])
        values = {name: encoders[name].value_for_code(row[i])
                  for i, name in enumerate(feature_order)}
        applied, _matched = _apply_decision_tree(rules, default, values)
        assert applied == pred, (row, values, applied, pred)


def test_conditions_are_well_formed_and_both_ops_appear():
    defs, rows = _synthetic()
    feature_order, encoders, X, y, clf = _fit(defs, rows)
    rules, default = _tree_to_rules(clf, feature_order, encoders, X)

    assert rules and isinstance(default, int)
    ops_seen = set()
    for r in rules:
        assert isinstance(r["score"], int)
        assert r["note"].startswith("CART leaf (n=")
        for c in r["when"]:
            assert set(c) == {"attribute", "op", "value"}
            assert c["attribute"] in feature_order
            assert c["op"] in ("==", "in")
            ops_seen.add(c["op"])
            if c["op"] == "in":
                assert isinstance(c["value"], list)
                assert len(c["value"]) >= 2
                assert len(c["value"]) == len(set(map(str, c["value"])))
            else:
                assert not isinstance(c["value"], list)
    # this synthetic tree exercises BOTH a multi-value `in` and a single `==`.
    assert ops_seen == {"==", "in"}


def test_missing_value_encodes_to_reserved_code():
    _fo, encoders = _encoders_from_defs(_synthetic()[0])
    assert encoders["stakes"].code(None) == tree_train.MISSING
    assert encoders["stakes"].code("") == tree_train.MISSING
    assert encoders["stakes"].code("bogus") == tree_train.MISSING
    # ordinal ascending rank; missing decodes to the "" sentinel.
    assert encoders["stakes"].code("low") == 0
    assert encoders["stakes"].code("high") == 1
    assert encoders["stakes"].value_for_code(tree_train.MISSING) == ""


def test_build_graph_and_module_import():
    # build_graph() must still compile with the new bootstrap JUDGE path wired in.
    assert build_graph() is not None
    assert hasattr(tree_train, "train_all")
