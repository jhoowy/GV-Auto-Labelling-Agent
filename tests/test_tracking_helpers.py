"""_seg_row: pure row -> segment-record shaping for node->segment tracking.

The shared editable install resolves ``tools`` to the primary checkout; prepend
THIS worktree's source root (and drop any cached ``tools``) so the test exercises
the code under review. PathFinder is consulted before the editable finder, so a
sys.path entry wins.
"""
import sys

_WT_TOOLS = "/home/iji/video-labelling-iji-wt-track/packages/tools"
if _WT_TOOLS not in sys.path:
    sys.path.insert(0, _WT_TOOLS)
for _m in [k for k in list(sys.modules) if k == "tools" or k.startswith("tools.")]:
    del sys.modules[_m]

from tools.tracking import _seg_row  # noqa: E402  (after the path fix-up above)


def test_shapes_the_segment_record():
    assert _seg_row("seg-1", "vid-9", 3) == {
        "segment_id": "seg-1",
        "video_id": "vid-9",
        "score": 3,
    }


def test_keeps_zero_score():
    # score 0 is a real value (not "missing") and must survive shaping.
    assert _seg_row("s", "v", 0)["score"] == 0
