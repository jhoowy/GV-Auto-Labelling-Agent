"""Labelling agent state, carried through the state machine as the shot window
slides over a video's segments."""
from __future__ import annotations

from typing import TypedDict

from schemas import Attribute, Label, Policy, Segment, Utterance


class LabellingState(TypedDict, total=False):
    video_id: str
    global_overview: str            # always-injected video context
    bootstrap: bool                 # bootstrap mode: no precedent retrieval; propose gaps

    all_segments: list[Segment]     # ordered by idx
    utterances: list[Utterance]     # word-level ASR on the video timeline
    cursor: int                     # index of first shot in the current window
    window: list[Segment]           # shots visible this step (confirm + neighbours)
    confirm: list[Segment]          # shots actually labelled this step (window head)

    window_size: int                # shots visible per step
    window_stride: int              # shots committed per step

    asr_by_segment: dict[str, str]  # segment_id -> merged ASR text for the window
    policy_attributes: dict[str, list[Attribute]]   # segment_id -> policy-layer attrs

    carry_over: str                 # rolling summary of confirmed judgements
    retrieved_policies: list[Policy]
    precedents: list[dict]          # similar segments + their confirmed labels

    draft_labels: list[Label]       # produced in SCORE, finalised in COMMIT
    proposals: list[dict]           # bootstrap-only: policy gaps for this window
    tool_trace: list[dict]          # every tool call, for label traces
