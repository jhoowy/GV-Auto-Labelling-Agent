"""Pure field mapping/merge for policy Korean translation (#22).

DB-free: exercises which fields are extracted for translation and how a
translation dict merges back into `structured_data.i18n.ko`, preserving the
English source + structure and stamping the per-version cache key.
"""
from tools.translate import (
    extract_translatable,
    is_up_to_date,
    merge_translation,
)


def _attr_def():
    return {
        "kind": "attribute_def",
        "value_type": "categorical",
        "guidelines": "Detect visible gambling activity in the shot.",
        "scores_informed": [2, 3],
        "values": [
            {"value": "none", "label": "None", "description": "No gambling.",
             "examples": ["a chess match"], "rules": ["no wagering shown"]},
            {"value": "simulated", "label": "Simulated", "description": "Play-money only.",
             "examples": [], "rules": ["chips but no real money", "arcade style"]},
        ],
    }


def _decision_tree():
    return {
        "kind": "decision_tree",
        "default": 0,
        "rules": [
            {"when": [{"attribute": "gambling", "op": "==", "value": "real"}],
             "score": 3, "note": "Real-money wagering present."},
            {"when": [{"attribute": "gambling", "op": "==", "value": "simulated"}],
             "score": 1},  # no note
        ],
    }


# --- extract: which fields are translatable --------------------------------

def test_extract_attribute_def_fields():
    src = extract_translatable(_attr_def())
    assert src["guidelines"].startswith("Detect")
    # value keys mirror the enum `value`, NOT translated; only prose extracted
    assert set(src["values"]) == {"none", "simulated"}
    assert src["values"]["none"] == {
        "label": "None", "description": "No gambling.", "rules": ["no wagering shown"],
    }
    # multi-rule list preserved in order; examples are NOT extracted
    assert src["values"]["simulated"]["rules"] == ["chips but no real money", "arcade style"]
    assert "examples" not in src["values"]["simulated"]


def test_extract_decision_tree_notes_index_aligned():
    src = extract_translatable(_decision_tree())
    # placeholder keeps rule index alignment even though rule[1] has no note
    assert src["rules"] == [{"note": "Real-money wagering present."}, {}]
    assert "default_note" not in src


def test_extract_skips_non_translatable():
    assert extract_translatable({"kind": "term_levels", "levels": {"3": ["poker"]}}) is None
    assert extract_translatable({"kind": "attribute_def", "values": []}) is None
    assert extract_translatable(None) is None
    assert extract_translatable("nope") is None


# --- merge: preserve English + structure, stamp version --------------------

def test_merge_preserves_english_and_adds_ko():
    sd = _attr_def()
    ko = {
        "guidelines": "샷에서 도박 행위를 탐지합니다.",
        "values": {
            "none": {"label": "없음", "description": "도박 없음.", "rules": ["배팅 없음"]},
            "simulated": {"label": "모의", "description": "가상 화폐만.",
                          "rules": ["실제 돈 없음", "아케이드 방식"]},
        },
    }
    out = merge_translation(sd, ko, src_version=4)

    # English source untouched (deep copy — original dict unchanged too)
    assert out["guidelines"] == "Detect visible gambling activity in the shot."
    assert out["values"] == sd["values"]
    assert "i18n" not in _attr_def()  # merge did not mutate a fresh source

    ko_out = out["i18n"]["ko"]
    assert ko_out["_src_version"] == 4
    assert ko_out["guidelines"] == "샷에서 도박 행위를 탐지합니다."
    assert ko_out["values"]["none"]["label"] == "없음"
    assert ko_out["values"]["simulated"]["rules"] == ["실제 돈 없음", "아케이드 방식"]


def test_merge_drops_invented_keys_and_mismatched_rule_lengths():
    sd = _attr_def()
    ko = {
        "guidelines": "가이드",
        "values": {
            # model dropped one rule -> length mismatch, rules omitted; label kept
            "none": {"label": "없음", "rules": []},
            # model invented an unknown value key -> ignored
            "made_up": {"label": "??"},
        },
    }
    out = merge_translation(sd, ko, src_version=1)
    ko_out = out["i18n"]["ko"]
    assert "made_up" not in ko_out["values"]
    assert ko_out["values"]["none"] == {"label": "없음"}  # rules dropped (len mismatch)
    assert "simulated" not in ko_out["values"]  # model omitted it -> fall back to EN in UI


def test_merge_decision_tree_notes_keep_alignment():
    sd = _decision_tree()
    ko = {"rules": [{"note": "실제 배팅이 존재합니다."}, {}]}
    out = merge_translation(sd, ko, src_version=2)
    ko_rules = out["i18n"]["ko"]["rules"]
    assert ko_rules == [{"note": "실제 배팅이 존재합니다."}, {}]
    # original English notes preserved
    assert out["rules"][0]["note"] == "Real-money wagering present."


def test_merge_preserves_existing_sibling_i18n():
    sd = _attr_def()
    sd["i18n"] = {"ja": {"guidelines": "既存"}}
    out = merge_translation(sd, {"guidelines": "가이드"}, src_version=1)
    assert out["i18n"]["ja"] == {"guidelines": "既存"}
    assert out["i18n"]["ko"]["guidelines"] == "가이드"


# --- version cache ---------------------------------------------------------

def test_is_up_to_date_version_cache():
    sd = merge_translation(_attr_def(), {"guidelines": "가이드"}, src_version=5)
    assert is_up_to_date(sd, 5) is True
    assert is_up_to_date(sd, 6) is False  # node changed -> re-translate
    assert is_up_to_date(_attr_def(), 5) is False  # no ko yet
