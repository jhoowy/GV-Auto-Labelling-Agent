"""Pure LangGraph helpers: lenient JSON, ASR overlap merge, segment rendering,
window loading and the terminal route decision."""
from langgraph.graph import END

from labelling.graph import (
    _load,
    _merged_asr,
    _parse_json,
    _route,
    _segment_block,
)
from schemas import Attribute, Segment, Utterance
from schemas.enums import AttributeLayer


def _seg(idx, t_start, t_end, **kw):
    return Segment(segment_id=f"s{idx}", video_id="v", idx=idx,
                   t_start=t_start, t_end=t_end, **kw)


def _utt(idx, t_start, t_end, text):
    return Utterance(video_id="v", idx=idx, t_start=t_start, t_end=t_end, text=text)


# --- _parse_json ------------------------------------------------------------

def test_parse_clean_object():
    assert _parse_json('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_parse_empty_returns_empty_dict():
    assert _parse_json("") == {}
    assert _parse_json(None) == {}


def test_parse_object_embedded_in_prose():
    text = 'Sure, here is the result: {"score": 4} — hope that helps!'
    assert _parse_json(text) == {"score": 4}


def test_parse_object_between_code_fences():
    text = '```json\n{"judgements": [], "need_more_frames": false}\n```'
    assert _parse_json(text) == {"judgements": [], "need_more_frames": False}


def test_parse_unrecoverable_garbage_returns_empty():
    assert _parse_json("no json here at all") == {}
    assert _parse_json("{not: valid, json}") == {}


def test_parse_spanning_regex_grabs_outermost_braces():
    # greedy \{.*\} spans from the first { to the last }
    assert _parse_json('lead {"a": {"b": 2}} trail') == {"a": {"b": 2}}


# --- _merged_asr ------------------------------------------------------------

def test_merge_empty_utterances():
    assert _merged_asr(_seg(0, 0.0, 10.0), []) == ""


def test_merge_only_overlapping_in_order():
    seg = _seg(0, 10.0, 20.0)
    utts = [
        _utt(0, 0.0, 5.0, "before"),      # ends before window -> excluded
        _utt(1, 8.0, 12.0, "spanning"),   # straddles start -> included
        _utt(2, 14.0, 16.0, "inside"),    # fully inside -> included
        _utt(3, 20.0, 25.0, "after"),     # starts at window end -> excluded
    ]
    assert _merged_asr(seg, utts) == "spanning inside"


def test_merge_boundary_touching_excluded():
    # strict inequalities: an utterance that merely touches an edge does not merge
    seg = _seg(0, 10.0, 20.0)
    assert _merged_asr(seg, [_utt(0, 5.0, 10.0, "left")]) == ""
    assert _merged_asr(seg, [_utt(0, 20.0, 30.0, "right")]) == ""


# --- _segment_block ---------------------------------------------------------

def test_segment_block_with_no_attributes():
    seg = _seg(2, 20.0, 30.5, summary="a car")
    block = _segment_block(seg, "hello there", [])
    assert "[shot 2 | s2 | 20.0-30.5s]" in block
    assert "summary: a car" in block
    assert "asr: hello there" in block
    assert "base_attributes: none" in block
    assert "policy_attributes: none" in block


def test_segment_block_renders_attributes_and_placeholders():
    base = Attribute(key="scene", value="casino", layer=AttributeLayer.BASE,
                     source="ingestion")
    pol = Attribute(key="gambling_structured_match", value=True,
                    layer=AttributeLayer.POLICY, source="judge/derive")
    seg = _seg(0, 0.0, 5.0)  # no summary
    block = _segment_block(seg, "", [pol])
    assert "summary: n/a" in block
    assert "asr: n/a" in block
    assert "policy_attributes: gambling_structured_match=True" in block
    # this segment carries no base attributes -> placeholder
    assert "base_attributes: none" in block
    # base attributes render off seg.base_attributes, not policy_attrs
    seg.base_attributes = [base]
    assert "base_attributes: scene=casino" in _segment_block(seg, "", [pol])


# --- _load ------------------------------------------------------------------

def test_load_slices_window_and_confirm_head():
    segs = [_seg(i, i * 10.0, i * 10.0 + 10) for i in range(7)]
    state = {"cursor": 0, "window_size": 5, "window_stride": 3,
             "all_segments": segs, "utterances": []}
    out = _load(state)
    assert [s.segment_id for s in out["window"]] == ["s0", "s1", "s2", "s3", "s4"]
    assert [s.segment_id for s in out["confirm"]] == ["s0", "s1", "s2"]


def test_load_confirm_falls_back_to_window_at_tail():
    segs = [_seg(i, i * 10.0, i * 10.0 + 10) for i in range(7)]
    # cursor past where a full stride remains: window shorter than stride
    state = {"cursor": 6, "window_size": 5, "window_stride": 3,
             "all_segments": segs, "utterances": []}
    out = _load(state)
    assert [s.segment_id for s in out["window"]] == ["s6"]
    assert [s.segment_id for s in out["confirm"]] == ["s6"]


def test_load_merges_asr_per_window_segment():
    segs = [_seg(0, 0.0, 10.0), _seg(1, 10.0, 20.0)]
    utts = [_utt(0, 5.0, 12.0, "hello")]
    out = _load({"cursor": 0, "window_size": 5, "window_stride": 3,
                 "all_segments": segs, "utterances": utts})
    assert out["asr_by_segment"] == {"s0": "hello", "s1": "hello"}


def test_load_records_stage_trace():
    segs = [_seg(0, 0.0, 10.0)]
    out = _load({"cursor": 0, "all_segments": segs, "utterances": []})
    assert out["tool_trace"][0]["stage"] == "LOAD"


# --- _route -----------------------------------------------------------------

def test_route_loops_until_cursor_exhausts_segments():
    segs = [_seg(i, 0.0, 1.0) for i in range(3)]
    assert _route({"cursor": 0, "all_segments": segs}) == "load"
    assert _route({"cursor": 2, "all_segments": segs}) == "load"
    assert _route({"cursor": 3, "all_segments": segs}) == END
    assert _route({"cursor": 99, "all_segments": segs}) == END
