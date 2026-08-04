"""_minmax: per-modality [0,1] normalisation used to fuse dense + lexical scores."""
import math

from tools.retrieval import _minmax


def test_empty_input():
    assert _minmax({}) == {}


def test_normalises_to_unit_range():
    out = _minmax({"a": 0.0, "b": 5.0, "c": 10.0})
    assert out == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_min_maps_to_zero_max_to_one():
    out = _minmax({"lo": 2.0, "mid": 4.0, "hi": 6.0})
    assert out["lo"] == 0.0 and out["hi"] == 1.0
    assert math.isclose(out["mid"], 0.5)


def test_degenerate_all_equal_maps_to_one():
    # hi <= lo: every candidate is equally (maximally) relevant
    assert _minmax({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}


def test_single_element_maps_to_one():
    assert _minmax({"only": 0.7}) == {"only": 1.0}


def test_negative_scores_are_shifted():
    out = _minmax({"a": -2.0, "b": 0.0, "c": 2.0})
    assert out == {"a": 0.0, "b": 0.5, "c": 1.0}
