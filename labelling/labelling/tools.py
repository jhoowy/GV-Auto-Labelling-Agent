"""Agent tools — thin LangChain wrappers over the shared service layer. They
add no business logic, only expose service functions to the orchestrator and
record into tool_trace.

  Retrieval:  search_policies, find_similar_segments, lookup_structured
  Context:    expand_window, get_frames, get_video_overview
  Mutation:   revise_ingestion, propose_policy_change, emit_label
  External:   web_search
"""
from __future__ import annotations

from schemas import Label, PolicyChangeRequest
from tools import policy_store, retrieval, storage


def search_policies(query: str, category: str | None = None):
    return retrieval.search_policies(query, category)


def find_similar_segments(segment_id: str):
    """Similar segments + their confirmed labels (precedent lookup)."""
    raise NotImplementedError


def lookup_structured(ref: str, text: str) -> bool:
    return retrieval.lookup_structured(ref, text)


def expand_window(direction: str, n: int):
    """Pull more neighbouring shots when context is insufficient."""
    raise NotImplementedError


def get_frames(segment_id: str, t_start: float, t_end: float):
    """Inspect a specific frame range via the vision MLLM."""
    raise NotImplementedError


def get_video_overview(video_id: str) -> str:
    from ingestion import build_global_overview
    return build_global_overview(video_id)


def revise_ingestion(segment_id: str, patch: dict) -> None:
    storage.revise_ingestion(segment_id, patch)


def propose_policy_change(change: str, rationale: str, affected: list[str]) -> None:
    """Always queued for human approval; never auto-applied."""
    policy_store.enqueue_change_request(
        PolicyChangeRequest(req_id="", proposed_change=change,
                            rationale=rationale, affected_segments=affected)
    )


def emit_label(label: Label) -> None:
    storage.save_label(label)
