"""_clamp: coerce + bound a raw model score into the PEGI 0..5 range."""
import pytest

from labelling.graph import _clamp
from schemas.enums import SCORE_MAX, SCORE_MIN


def test_bounds_are_the_pegi_range():
    assert (SCORE_MIN, SCORE_MAX) == (0, 5)


@pytest.mark.parametrize("raw,expected", [(0, 0), (3, 3), (5, 5)])
def test_in_range_passthrough(raw, expected):
    assert _clamp(raw) == expected


def test_above_max_clamps_down():
    assert _clamp(9) == SCORE_MAX
    assert _clamp(100) == SCORE_MAX


def test_below_min_clamps_up():
    assert _clamp(-1) == SCORE_MIN
    assert _clamp(-50) == SCORE_MIN


def test_float_and_numeric_string_coerce():
    assert _clamp(4.9) == 4
    assert _clamp("3") == 3


@pytest.mark.parametrize("bad", [None, "high", "", object()])
def test_uncoercible_falls_back_to_min(bad):
    assert _clamp(bad) == SCORE_MIN
