"""LangGraph orchestrator — structured state machine with fixed stages, each
with a restricted tool set. The window slides until all segments are labelled.

    LOAD -> RETRIEVE -> DERIVE -> SCORE -> CHECK -> SIDE_FX -> COMMIT
                                            |
                               (loop to next window until done)

  RETRIEVE  policies + precedents
  DERIVE    policy-layer attributes (e.g. profanity via word list)
  CHECK     if a judgement contradicts precedent, re-examine; if still
            divergent, record why in the trace
  SIDE_FX   revise_ingestion (auto+log) / propose_policy_change (queue)
  COMMIT    emit labels + update the carry-over rolling summary
"""
from __future__ import annotations

from .state import LabellingState

STAGES = ["LOAD", "RETRIEVE", "DERIVE", "SCORE", "CHECK", "SIDE_FX", "COMMIT"]


def build_graph():
    """Construct and compile the StateGraph over LabellingState: one node per
    stage, with a conditional edge from COMMIT that advances the window cursor
    and loops to LOAD until all_segments is exhausted."""
    raise NotImplementedError


def label_video(video_id: str) -> None:
    """Run the compiled graph across one video's shot windows."""
    _ = LabellingState
    raise NotImplementedError
