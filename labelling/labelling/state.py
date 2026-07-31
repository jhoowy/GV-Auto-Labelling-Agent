"""Labelling agent state, carried through the state machine as the shot window
slides over a video's segments."""
from __future__ import annotations

from typing import TypedDict

from schemas import Label, Policy, Segment


class LabellingState(TypedDict, total=False):
    video_id: str
    global_overview: str            # always-injected video context

    all_segments: list[Segment]     # ordered by idx
    cursor: int                     # index of first shot in the current window
    window: list[Segment]           # shots visible this step (confirm + neighbours)

    carry_over: str                 # rolling summary of confirmed judgements
    retrieved_policies: list[Policy]
    precedents: list[dict]          # similar segments + their confirmed labels

    draft_labels: list[Label]       # produced in SCORE, finalised in COMMIT
    tool_trace: list[dict]          # every tool call, for label traces
