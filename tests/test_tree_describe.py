"""EN + KO description merge onto decision-tree rules (DB-free, no live LLM).

Exercises `tree_describe._merge_descriptions` — the pure function that folds an
LLM response's per-index descriptions onto the rule dicts — asserting it:
  (a) merges aligned entries and leaves the `{when,score,note}` storage keys
      untouched (only ADDS description/description_ko);
  (b) is defensive against short / long / misaligned / malformed responses,
      never mutating the input list and never corrupting a rule.
Also import-checks the module + the guarded describe hook is wired in.
"""
from tools import tree_describe
from tools.tree_describe import _merge_descriptions


def _rules():
    return [
        {"when": [{"attribute": "activity", "op": "==", "value": "real"}],
         "score": 5, "note": "CART leaf (n=3)"},
        {"when": [{"attribute": "activity", "op": "==", "value": "none"}],
         "score": 0, "note": "CART leaf (n=5)"},
    ]


def test_merges_en_ko_by_index_without_touching_storage_keys():
    rules = _rules()
    resp = {"rules": [
        {"index": 0, "description": "Real gambling.", "description_ko": "실제 도박."},
        {"index": 1, "description": "No gambling.", "description_ko": "도박 없음."},
    ]}
    out = _merge_descriptions(rules, resp)

    assert out[0]["description"] == "Real gambling."
    assert out[0]["description_ko"] == "실제 도박."
    assert out[1]["description"] == "No gambling."
    assert out[1]["description_ko"] == "도박 없음."
    # storage keys are preserved verbatim.
    for i, r in enumerate(out):
        assert r["when"] == rules[i]["when"]
        assert r["score"] == rules[i]["score"]
        assert r["note"] == rules[i]["note"]
    # input list is not mutated in place.
    assert "description" not in rules[0]


def test_short_response_only_sets_aligned_rules():
    rules = _rules()
    resp = {"rules": [{"index": 0, "description": "Only first."}]}
    out = _merge_descriptions(rules, resp)
    assert out[0]["description"] == "Only first."
    assert "description" not in out[1]
    assert "description_ko" not in out[0]  # missing KO not invented


def test_misaligned_and_malformed_entries_are_ignored():
    rules = _rules()
    resp = {"rules": [
        {"index": 9, "description": "out of range"},      # dropped
        {"index": -1, "description": "negative"},          # dropped
        {"index": True, "description": "bool not int"},    # dropped
        {"index": "0", "description": "string index"},     # dropped
        "not a dict",                                       # dropped
        {"index": 1, "description": "   ", "description_ko": ""},  # blank -> skip
        {"index": 0, "description": "kept"},               # applied
    ]}
    out = _merge_descriptions(rules, resp)
    assert out[0]["description"] == "kept"
    assert "description" not in out[1]
    assert "description_ko" not in out[1]


def test_non_dict_and_empty_responses_are_safe():
    rules = _rules()
    for resp in (None, {}, {"rules": None}, {"rules": []}, [], "junk"):
        out = _merge_descriptions(rules, resp)
        assert out == rules  # equal content, undescribed
        assert all("description" not in r for r in out)


def test_module_wires_describe_api_and_guarded_hook():
    assert hasattr(tree_describe, "describe_rules")
    assert hasattr(tree_describe, "describe_all")
    # the guarded describe hook is present in train_decision_tree's source.
    import inspect

    from tools import tree_train
    src = inspect.getsource(tree_train.train_decision_tree)
    assert "describe_rules" in src and "try:" in src
