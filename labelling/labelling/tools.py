"""Agent tools — thin LangChain wrappers over the shared service layer. They
add no business logic, only expose service functions to the orchestrator and
record into tool_trace.

  Retrieval:  search_policies, find_similar_segments, lookup_structured
  Context:    expand_window, get_frames, get_video_overview
  Mutation:   revise_ingestion, propose_policy_change, emit_label
  External:   web_search
"""
from __future__ import annotations

import asyncio

from schemas import Label, PolicyChangeRequest, Segment

# NOTE: the service layer (`tools`) pulls in `db`, so it is imported lazily inside
# each wrapper — a plain import of this module (and build_graph) needs no DB.


def search_policies(query: str, category: str | None = None):
    from tools import retrieval
    return retrieval.search_policies(query, category)


def find_similar_segments(segment: Segment, top_k: int = 5):
    """Similar segments + their confirmed labels (precedent lookup).

    Returns list[(Segment, list[Label])] — the primary consistency signal.
    """
    from tools import retrieval
    return retrieval.find_similar_segments(segment, top_k)


def lookup_structured(ref: str, text: str) -> bool:
    from tools import retrieval
    return retrieval.lookup_structured(ref, text)


def define_structured_attribute(category: str, name: str,
                                levels: dict[str, list[str]],
                                description: str | None = None):
    """Bootstrap-only skill: draft/edit a structured ATTRIBUTE node (e.g. a
    profanity term list organised by score level). Directly upserts the node —
    the whole draft tree is human-reviewed before the policy-set v1 snapshot."""
    from tools import policy_store
    return policy_store.upsert_structured_attribute(category, name, levels, description)


def expand_window(
    all_segments: list[Segment], lo: int, hi: int, direction: str, n: int
) -> list[Segment]:
    """Pull more neighbouring shots when context is insufficient.

    `all_segments` is the full ordered list; `[lo, hi)` is the current window.
    Returns the neighbour shots just outside that window on the requested side
    ("left" | "right" | "both"), clamped to the video bounds.
    """
    left = all_segments[max(0, lo - n):lo] if direction in ("left", "both") else []
    right = all_segments[hi:hi + n] if direction in ("right", "both") else []
    return left + right


def get_frames(segment: Segment, t_start: float, t_end: float,
               prompt: str | None = None) -> str:
    """Inspect a specific frame range via the vision MLLM.

    Resolves the segment's clip blob to a local path and asks the omni MLLM to
    describe what happens in [t_start, t_end]. Model/blob access happens here,
    not at import time.
    """
    if not segment.clip_blob:
        raise ValueError(f"segment {segment.segment_id} has no clip_blob to inspect")
    from models import get_mllm
    from tools import blob

    clip_path = blob.get_blob_store().url(segment.clip_blob)
    ask = prompt or (
        f"Describe in detail what happens between {t_start:.2f}s and {t_end:.2f}s "
        "of this clip, focusing on any moderation-relevant content."
    )
    mllm = get_mllm()
    return asyncio.run(mllm.describe(clip_path, None, ask))


def sample_frames(segment: Segment, n: int = 5) -> list[bytes]:
    """Uniformly sample up to n JPEG frames from the shot (light percept).

    Prefers the shot's own av clip; falls back to the source video offset into
    the shot span. Returns raw JPEG bytes for the multimodal orchestrator.
    """
    import os
    import subprocess
    import tempfile

    from tools import blob, storage

    if segment.clip_blob:
        path = blob.get_blob_store().url(segment.clip_blob)
        base, dur = 0.0, max(0.1, segment.t_end - segment.t_start)
    else:
        video = storage.get_video(segment.video_id)
        if not video or not video.source_blob:
            return []
        path = blob.get_blob_store().url(video.source_blob)
        base, dur = segment.t_start, max(0.1, segment.t_end - segment.t_start)

    n = max(1, min(n, 8))
    frames: list[bytes] = []
    for i in range(n):
        t = base + dur * (i + 0.5) / n
        fd, out = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                            "-i", path, "-frames:v", "1", "-q:v", "3", out], check=True)
            data = open(out, "rb").read()
            if data:
                frames.append(data)
        except Exception:  # noqa: BLE001 - skip an unreadable frame
            pass
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass
    return frames


def expand_frames(segment: Segment, n: int = 10) -> list[bytes]:
    """Denser frame sampling when the initial percept was insufficient."""
    return sample_frames(segment, n)


def get_video_overview(video_id: str) -> str:
    from ingestion import build_global_overview
    return build_global_overview(video_id)


def revise_ingestion(segment_id: str, patch: dict) -> None:
    from tools import storage
    storage.revise_ingestion(segment_id, patch)


def propose_policy_change(change: str, rationale: str, affected: list[str],
                          category: str | None = None,
                          node_type: str | None = None,
                          target_policy_id: str | None = None) -> None:
    """Always queued for human approval; never auto-applied. category/node_type
    let an approver materialise the proposal into a policy node;
    target_policy_id names an existing node the change edits (if any)."""
    from tools import policy_store
    policy_store.enqueue_change_request(
        PolicyChangeRequest(req_id="", proposed_change=change, rationale=rationale,
                            category=category, node_type=node_type,
                            target_policy_id=target_policy_id,
                            affected_segments=affected)
    )


def define_attribute_definition(category: str, name: str, value_type: str,
                                guidelines: str, scores_informed: list[int],
                                values: list | None = None,
                                examples: list | None = None):
    """Bootstrap/redesign skill: draft/edit an ATTRIBUTE *definition* node
    (schema + detection guidelines). Directly upserts; not wired into JUDGE."""
    from tools import policy_store
    return policy_store.upsert_attribute_definition(
        category, name, value_type, guidelines, scores_informed, values, examples)


def define_decision_rule(category: str, rules: list[dict], default: int = 0):
    """Bootstrap/redesign skill: draft/edit the category's DECISION_RULE node
    (attribute-based decision tree). Directly upserts; not wired into JUDGE."""
    from tools import policy_store
    return policy_store.upsert_decision_rule(category, rules, default)


def emit_label(label: Label) -> None:
    from tools import storage
    storage.save_label(label)
