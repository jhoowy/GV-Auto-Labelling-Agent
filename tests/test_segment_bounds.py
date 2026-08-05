"""_cap_and_fill: pure segment bound-transform — contiguity (gap-fill) + 30s cap.

Tests the transform in isolation (no Omni/DB). The editable install resolves
``ingestion`` to a *different* checkout, so prepend this worktree's ingestion
root to make the code under test authoritative (mirrors conftest for tools/
labelling).
"""
import sys
from pathlib import Path

_ING_ROOT = str(Path(__file__).resolve().parent.parent / "ingestion")
if _ING_ROOT not in sys.path:
    sys.path.insert(0, _ING_ROOT)
sys.modules.pop("ingestion", None)
sys.modules.pop("ingestion.pipeline", None)

from ingestion.pipeline import _cap_and_fill  # noqa: E402


def _check_contiguous(out, duration):
    assert out[0]["t_start"] == 0.0
    assert out[-1]["t_end"] == duration
    for a, b in zip(out, out[1:]):
        assert b["t_start"] == a["t_end"]


def test_empty_returns_empty():
    assert _cap_and_fill([], 100.0, 30.0) == []


def test_already_contiguous_under_cap_passthrough():
    final = [{"t_start": 0.0, "t_end": 20.0, "summary": "a"},
             {"t_start": 20.0, "t_end": 40.0, "summary": "b"}]
    out = _cap_and_fill(final, 40.0, 30.0)
    _check_contiguous(out, 40.0)
    assert [s["summary"] for s in out] == ["a", "b"]


def test_leading_gap_is_filled():
    # First scene starts at 60s -> [0, 60) must be back-filled onto it.
    final = [{"t_start": 60.0, "t_end": 90.0, "summary": "scene"}]
    out = _cap_and_fill(final, 90.0, 0)  # no cap, isolate gap-fill
    _check_contiguous(out, 90.0)
    assert out[0]["t_start"] == 0.0 and out[0]["t_end"] == 90.0
    assert out[0]["summary"] == "scene"


def test_mid_gap_is_filled():
    final = [{"t_start": 0.0, "t_end": 10.0, "summary": "a"},
             {"t_start": 25.0, "t_end": 40.0, "summary": "b"}]  # gap [10, 25)
    out = _cap_and_fill(final, 40.0, 0)
    _check_contiguous(out, 40.0)
    assert out[1]["t_start"] == 10.0  # b extended back to fill the gap


def test_tail_forced_to_duration():
    final = [{"t_start": 0.0, "t_end": 30.0, "summary": "a"}]
    out = _cap_and_fill(final, 50.0, 0)
    assert out[-1]["t_end"] == 50.0


def test_no_segment_exceeds_max():
    final = [{"t_start": 0.0, "t_end": 100.0, "summary": "long"}]
    out = _cap_and_fill(final, 100.0, 30.0)
    for s in out:
        assert s["t_end"] - s["t_start"] <= 30.0 + 1e-9


def test_split_piece_count_and_bounds():
    # 100s / 30 -> ceil = 4 equal pieces of 25s, contiguous over [0, 100].
    final = [{"t_start": 0.0, "t_end": 100.0, "summary": "long"}]
    out = _cap_and_fill(final, 100.0, 30.0)
    assert len(out) == 4
    _check_contiguous(out, 100.0)
    for s in out:
        assert s["summary"] == "long"  # summary inherited by every piece
        assert abs((s["t_end"] - s["t_start"]) - 25.0) < 1e-9


def test_split_exact_multiple():
    # 60s / 30 -> exactly 2 pieces, not 3.
    final = [{"t_start": 0.0, "t_end": 60.0, "summary": "x"}]
    out = _cap_and_fill(final, 60.0, 30.0)
    assert len(out) == 2
    _check_contiguous(out, 60.0)


def test_max_zero_disables_cap():
    final = [{"t_start": 0.0, "t_end": 200.0, "summary": "x"}]
    out = _cap_and_fill(final, 200.0, 0)
    assert len(out) == 1
    _check_contiguous(out, 200.0)


def test_negative_max_disables_cap():
    final = [{"t_start": 0.0, "t_end": 200.0, "summary": "x"}]
    out = _cap_and_fill(final, 200.0, -5.0)
    assert len(out) == 1


def test_gap_fill_then_cap_combined():
    # Leading 0-60 gap AND an over-long final -> filled and split, all <= max.
    final = [{"t_start": 60.0, "t_end": 150.0, "summary": "s"}]
    out = _cap_and_fill(final, 150.0, 30.0)
    _check_contiguous(out, 150.0)
    for s in out:
        assert s["t_end"] - s["t_start"] <= 30.0 + 1e-9
        assert s["summary"] == "s"
