"""SELECT -> EXTRACT -> DECIDE -> REVIEW -> STORE pipeline helpers (pure, DB-free).

Covers the lenient SELECT/REVIEW parsers, the empty-safe DECIDE fall-through
(a category with zero selected attributes -> tree default), and the STORE
representation of unselected attributes (stored empty, not dropped)."""
from labelling.graph import (
    _apply_decision_tree,
    _attr_index,
    _build_label,
    _parse_reviews,
    _parse_selections,
)
from schemas import Policy, Segment
from schemas.enums import AttributeLayer, Category, PolicyType


# --- fixtures ---------------------------------------------------------------

def _def(cat, name, value_type, values, guidelines="detect it", version=1):
    return Policy(
        policy_id=f"{cat}.attr.{name}", type=PolicyType.ATTRIBUTE,
        category=Category(cat), version=version, parent_id=f"{cat}.scoring",
        text=f"Attribute '{name}' for {cat}.",
        structured_data={"kind": "attribute_def", "value_type": value_type,
                         "values": values, "guidelines": guidelines,
                         "scores_informed": []})


def _rule_node(cat, rules, default=0, version=1):
    return Policy(
        policy_id=f"{cat}.rules", type=PolicyType.DECISION_RULE,
        category=Category(cat), version=version, parent_id=f"{cat}.scoring",
        text=f"Decision tree for {cat}.",
        structured_data={"kind": "decision_tree", "default": default,
                         "rules": rules})


def _seg(idx=0):
    return Segment(segment_id=f"s{idx}", video_id="v", idx=idx,
                   t_start=0.0, t_end=5.0)


def _sex_defs():
    # one categorical attr with per-value rules, one ordinal attr.
    nudity = _def("sex", "nudity", "categorical",
                  [{"value": "none", "description": "no nudity",
                    "rules": ["case fully clothed -> this value"]},
                   {"value": "partial", "description": "partial nudity",
                    "rules": ["case swimwear/underwear -> this value"]}])
    intensity = _def("sex", "intensity", "ordinal",
                     [{"value": "low"}, {"value": "high"}])
    return [nudity, intensity]


# --- _parse_selections ------------------------------------------------------

def test_selections_keeps_known_and_defaults_missing_category():
    attr_cats = ["sex", "gambling"]
    names_by_cat = {"sex": ["nudity", "intensity"], "gambling": ["bet_present"]}
    parsed = {"selections": [
        {"category": "sex", "attributes": ["nudity", "bogus_attr"]},
        # gambling omitted entirely -> defaults to []
        {"category": "violence", "attributes": ["gore"]},   # unknown cat dropped
    ]}
    out = _parse_selections(parsed, attr_cats, names_by_cat)
    assert out == {"sex": ["nudity"], "gambling": []}


def test_selections_dedupe_and_empty_parsed():
    attr_cats = ["sex"]
    names_by_cat = {"sex": ["nudity"]}
    dup = {"selections": [{"category": "sex", "attributes": ["nudity", "nudity"]}]}
    assert _parse_selections(dup, attr_cats, names_by_cat) == {"sex": ["nudity"]}
    # a garbage / empty payload -> every category defaults to []
    assert _parse_selections({}, attr_cats, names_by_cat) == {"sex": []}
    assert _parse_selections({"selections": "nope"}, attr_cats,
                             names_by_cat) == {"sex": []}


# --- _parse_reviews ---------------------------------------------------------

def test_reviews_keeps_known_flags_and_notes():
    parsed = {"reviews": [
        {"category": "sex", "needs_change": True, "change_note": "too low"},
        {"category": "gambling", "needs_change": False},
        {"category": "violence", "needs_change": True},   # unknown cat dropped
    ]}
    out = _parse_reviews(parsed, ["sex", "gambling"])
    assert out["sex"] == {"needs_change": True, "change_note": "too low"}
    assert out["gambling"] == {"needs_change": False, "change_note": ""}
    assert "violence" not in out


# --- empty-safe DECIDE ------------------------------------------------------

def test_decide_zero_selected_attrs_falls_through_to_default():
    # a rule that requires an attribute; with NO extracted values the condition
    # can never be present -> the tree returns its default score.
    rules = [{"when": [{"attribute": "nudity", "op": "==", "value": "partial"}],
              "score": 3, "note": "partial nudity"}]
    score, matched = _apply_decision_tree(rules, default=0, values={})
    assert (score, matched) == (0, None)
    # and once the attribute is present the same tree does fire.
    score2, matched2 = _apply_decision_tree(
        rules, default=0, values={"nudity": "partial"})
    assert score2 == 3 and matched2 is rules[0]


# --- STORE: empty-value storage for unselected attributes -------------------

def test_build_label_stores_unselected_attribute_empty():
    defs = _sex_defs()
    rule_node = _rule_node(
        "sex", [{"when": [{"attribute": "nudity", "op": "==", "value": "partial"}],
                 "score": 3, "note": "partial nudity"}])
    idx = _attr_index(defs)

    selected = ["nudity"]  # 'intensity' was NOT selected
    extracted = {"nudity": {"value": "partial", "evidence": "frame 2 torso"}}
    trajectory = {"selected": selected, "extracted": {"nudity": "partial"},
                  "rule_index": 0, "rule_note": "partial nudity", "score": 3}

    lbl = _build_label(_seg(), "sex", 3, trajectory,
                       matched=rule_node.structured_data["rules"][0],
                       defs=defs, rule_node=rule_node, idx=idx,
                       selected=selected, extracted=extracted)

    ev = {a.key: a for a in lbl.evidence_attributes}
    # EVERY defined attribute is represented (considered-and-empty distinguishable)
    assert set(ev) == {"nudity", "intensity"}
    assert ev["nudity"].value == "partial"
    assert ev["nudity"].source == "judge/extract"
    assert ev["nudity"].evidence == "frame 2 torso"
    # unselected attribute -> stored EMPTY, not dropped
    assert ev["intensity"].value == ""
    assert ev["intensity"].source == "judge/unselected"
    assert ev["intensity"].evidence is None
    assert ev["intensity"].layer == AttributeLayer.POLICY
    # trajectory rides in tool_trace; pins cover both attr defs + the rule node
    assert lbl.tool_trace == [{"decision": trajectory}]
    assert set(lbl.cited_policy_ids) == {
        "sex.attr.nudity,v1", "sex.attr.intensity,v1", "sex.rules,v1"}
    assert "rule #0 matched" in lbl.rationale


def test_build_label_no_match_uses_default_rationale():
    defs = _sex_defs()
    rule_node = _rule_node("sex", [], default=0)
    idx = _attr_index(defs)
    trajectory = {"selected": [], "extracted": {}, "rule_index": None,
                  "rule_note": "", "score": 0}
    lbl = _build_label(_seg(), "sex", 0, trajectory, matched=None,
                       defs=defs, rule_node=rule_node, idx=idx,
                       selected=[], extracted={})
    # both attributes considered-and-empty
    assert all(a.value == "" and a.source == "judge/unselected"
               for a in lbl.evidence_attributes)
    assert "no decision rule matched" in lbl.rationale


# --- _attr_index / _attr_line render per-value rules ------------------------

def test_attr_index_line_includes_per_value_rules():
    idx = _attr_index(_sex_defs())
    assert idx["names"] == ["nudity", "intensity"]
    assert idx["order"] == {"intensity": ["low", "high"]}   # ordinal keeps order
    line = idx["line_by_name"]["nudity"]
    assert "rule: case fully clothed -> this value" in line
    assert "rule: case swimwear/underwear -> this value" in line
