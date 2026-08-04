"""LangGraph orchestrator — structured state machine with fixed stages, each
with a restricted tool set. The window slides until all segments are labelled.

    LOAD -> RETRIEVE -> JUDGE -> SIDE_FX -> COMMIT
                          |
             (loop to next window until done)

  RETRIEVE  policies + precedents
  JUDGE     multimodal orchestrator (frames + summary + ASR) scores every
            category; derives policy-layer attributes and checks precedent
            consistency inline (divergences recorded as issues in the trace)
  SIDE_FX   revise_ingestion (auto+log) / propose_policy_change (queue)
  COMMIT    emit labels + update the carry-over rolling summary

See docs/AGENT_WORKFLOW.md for the full definition.
"""
from __future__ import annotations

import json
import math
import re
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from schemas import Attribute, Label
from schemas.enums import SCORE_MAX, SCORE_MIN, AttributeLayer, Category

from . import tools as agent_tools
from .state import LabellingState

# The service layer (`tools.*`, `models`) pulls in `db` / model clients, so it is
# imported lazily inside the nodes that use it — build_graph() needs no server.

STAGES = ["LOAD", "RETRIEVE", "JUDGE", "SIDE_FX", "COMMIT"]

_FRAMES_PER_SHOT = 5

_JUDGE_SYS = (
    "You are a content-moderation labelling agent for gameplay videos. Judge the "
    "TARGET SHOT for EACH of these categories: {cats}. Score each on a 0-5 PEGI "
    "age score (0:3+, 1:7+, 2:12+, 3:16+, 4:18+, 5:blocked). Ground every "
    "judgement in the supplied policy nodes and cite the exact policy ids you "
    "relied on. Stay consistent with the precedent labels of similar shots unless "
    "the evidence clearly differs. You are given up to 5 sampled frames plus the "
    "shot summary and ASR text; if the frames are insufficient to judge, set "
    "need_more_frames=true. Return ONLY JSON of the form "
    '{{"judgements":[{{"category":str,"score":int,"rationale":str,'
    '"cited_policy_ids":[str],"confidence":float}}],"need_more_frames":bool}} '
    "with one judgement per category."
)

_BOOTSTRAP_SUFFIX = (
    " The policy set is still being bootstrapped from unlabelled data, so no "
    "precedents are available. Additionally report policy_gaps: content in this "
    "shot the current rubric does NOT cover well and that warrants a new "
    "ATTRIBUTE definition or EDGE_CASE rule. Add a top-level "
    '"policy_gaps":[{{"category":str,"kind":"attribute"|"edge_case",'
    '"suggestion":str,"rationale":str}}] (empty list if none).'
    " If a category would benefit from a STRUCTURED term list matched by score "
    "level (e.g. a profanity list where stronger terms map to higher scores), "
    'also add "structured_attributes":[{{"category":str,"name":str,'
    '"levels":{{"<score-level>":[terms]}},"description":str}}] (empty list if '
    "none). Levels are PEGI bands 0..5; a term at level L is evidence toward "
    "score L for that category."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _categories() -> list[str]:
    """Active category set — parameterised from policy config, not hardcoded."""
    from models import base_config

    cats = (base_config().get("policy", {}) or {}).get("categories")
    return list(cats) if cats else [c.value for c in Category]


def _clamp(score) -> int:
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = SCORE_MIN
    return max(SCORE_MIN, min(SCORE_MAX, s))


def _merged_asr(seg, utterances) -> str:
    words = [u.text for u in utterances
             if u.t_end > seg.t_start and u.t_start < seg.t_end]
    return " ".join(words).strip()


def _parse_json(text: str) -> dict:
    """Lenient parse — structured output is not forced on the model."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _normalise_pins(cited: list[str], version_by_id: dict[str, int]) -> list[str]:
    """Canonicalise the model's cited policy ids into (policy_id, version) pins.

    Keeps only ids that were actually retrieved (drops hallucinated ones) and
    re-attaches the TRUE retrieved version — the model's own ",vN" suffix, if
    any, is stripped and never trusted. Order preserved, deduped.
    """
    pins: list[str] = []
    seen: set[str] = set()
    for cid in cited:
        if not cid:
            continue
        base = re.sub(r",v\d+$", "", str(cid).strip())
        ver = version_by_id.get(base)
        if ver is None:
            continue
        pin = f"{base},v{ver}"
        if pin not in seen:
            seen.add(pin)
            pins.append(pin)
    return pins


def _segment_block(seg, asr: str, policy_attrs: list[Attribute]) -> str:
    base = ", ".join(f"{a.key}={a.value}" for a in seg.base_attributes) or "none"
    derived = ", ".join(f"{a.key}={a.value}" for a in policy_attrs) or "none"
    return (
        f"[shot {seg.idx} | {seg.segment_id} | {seg.t_start:.1f}-{seg.t_end:.1f}s]\n"
        f"summary: {seg.summary or 'n/a'}\n"
        f"asr: {asr or 'n/a'}\n"
        f"base_attributes: {base}\n"
        f"policy_attributes: {derived}"
    )


def _judge_prompt(state, seg, asr, policy_attrs, policy_text, precedents) -> str:
    seg_prec = [pr for pr in precedents if pr["segment_id"] == seg.segment_id]
    prec_text = "\n".join(
        f"- similar {pr['similar_id']}: " + ", ".join(
            f"{lbl['category']}={lbl['score']}" for lbl in pr["labels"]
        )
        for pr in seg_prec
    ) or "none"
    return (
        f"GLOBAL VIDEO OVERVIEW:\n{state.get('global_overview', '') or 'n/a'}\n\n"
        f"CARRY-OVER (confirmed so far):\n{state.get('carry_over', '') or 'none'}\n\n"
        f"TARGET SHOT (frames attached below):\n"
        f"{_segment_block(seg, asr, policy_attrs)}\n\n"
        f"RELEVANT POLICY NODES:\n{policy_text}\n\n"
        f"PRECEDENT LABELS (similar shots):\n{prec_text}\n\n"
        "Judge every category for the TARGET SHOT."
    )


# --------------------------------------------------------------------------- #
# stage nodes
# --------------------------------------------------------------------------- #
def _load(state: LabellingState) -> dict:
    """Populate the window; merge overlapping ASR. The head (window_stride) shots
    are committed; the rest are neighbour context. tool_trace resets per window."""
    cursor = state["cursor"]
    size = state.get("window_size", 5)
    stride = state.get("window_stride", 3)
    segs = state["all_segments"]
    utterances = state.get("utterances", [])

    window = segs[cursor:cursor + size]
    confirm = window[:stride] or window
    asr_by_segment = {s.segment_id: _merged_asr(s, utterances) for s in window}

    trace = [{"stage": "LOAD", "cursor": cursor,
              "window": [s.segment_id for s in window],
              "confirm": [s.segment_id for s in confirm]}]
    return {"window": window, "confirm": confirm,
            "asr_by_segment": asr_by_segment, "tool_trace": trace}


def _retrieve(state: LabellingState) -> dict:
    """search_policies + find_similar_segments -> retrieved_policies, precedents."""
    confirm = state.get("confirm", state.get("window", []))
    asr = state.get("asr_by_segment", {})
    query = " ".join(
        f"{s.summary or ''} {asr.get(s.segment_id, '')}" for s in confirm
    ).strip()

    policies = []
    for cat in _categories():
        policies += agent_tools.search_policies(query, cat)

    # Bootstrap has no confirmed labels yet -> precedent retrieval is disabled.
    precedents: list[dict] = []
    if not state.get("bootstrap"):
        for seg in confirm:
            for sim, labels in agent_tools.find_similar_segments(seg):
                precedents.append({
                    "segment_id": seg.segment_id,
                    "similar_id": sim.segment_id,
                    "labels": [lbl.model_dump(mode="json") for lbl in labels],
                })

    trace = state.get("tool_trace", []) + [{
        "stage": "RETRIEVE", "n_policies": len(policies),
        "n_precedents": len(precedents), "bootstrap": bool(state.get("bootstrap")),
    }]
    return {"retrieved_policies": policies, "precedents": precedents,
            "tool_trace": trace}


def _judge(state: LabellingState) -> dict:
    """Multimodal judging (absorbs DERIVE + CHECK).

    For each confirm shot: derive policy-layer attributes, sample ≤5 frames, ask
    the orchestrator to score every category (once, with a single frame-expansion
    retry if it asks), then record precedent divergences as issues on the trace.
    The orchestrator is resolved here, so build_graph() needs no server."""
    from models import get_agent_llm
    from tools import policy_store, retrieval

    orch = get_agent_llm()
    cats = _categories()
    bootstrap = bool(state.get("bootstrap"))
    confirm = state.get("confirm", [])
    asr = state.get("asr_by_segment", {})
    policies = state.get("retrieved_policies", [])
    precedents = state.get("precedents", [])
    system = _JUDGE_SYS.format(cats=", ".join(cats))
    if bootstrap:
        system += _BOOTSTRAP_SUFFIX

    policy_text = "\n".join(
        f"- ({p.policy_id},v{p.version}) [{p.type.value}/{p.category.value}] {p.text}"
        for p in policies
    ) or "none"

    # canonical (policy_id -> version) pinning source: what was actually retrieved
    version_by_id = {p.policy_id: p.version for p in policies}

    # DB-managed structured attributes (term lists by score level) per active
    # category. Not mandatory: if the tree has none, DERIVE emits nothing.
    term_nodes: list[tuple[str, str, dict, int]] = []
    for cat in cats:
        for node in policy_store.get_policy_tree(cat):
            sd = node.structured_data or {}
            if sd.get("kind") == "term_levels":
                prefix = f"{cat}.attr."
                name = (node.policy_id[len(prefix):]
                        if node.policy_id.startswith(prefix) else node.policy_id)
                term_nodes.append((cat, name, sd, node.version))

    drafts: list[Label] = []
    proposals: list[dict] = []
    extra_trace: list[dict] = []
    for seg in confirm:
        text = asr.get(seg.segment_id, "")

        # DERIVE (absorbed): graded policy-layer attributes from structured term
        # lists. Per matching node add ONE attribute at the max matched level.
        pattrs: list[Attribute] = []
        for cat, name, sd, ver in term_nodes:
            matched = retrieval.match_term_levels(sd, text)
            if not matched:
                continue
            max_level = max(int(lvl) for lvl in matched)
            pattrs.append(Attribute(
                key=f"{cat}.{name}_level", value=int(max_level),
                layer=AttributeLayer.POLICY, source="judge/derive",
                evidence=", ".join(matched.get(str(max_level), [])),
                policy_version=ver,
            ))

        frames = agent_tools.sample_frames(seg, _FRAMES_PER_SHOT)
        prompt = _judge_prompt(state, seg, text, pattrs, policy_text, precedents)
        parsed = _parse_json(orch.judge(system, prompt, frames))

        # active perception: one frame-expansion retry if the model asks
        if parsed.get("need_more_frames") and frames:
            more = agent_tools.expand_frames(seg, _FRAMES_PER_SHOT * 2)
            parsed = _parse_json(orch.judge(system, prompt, more)) or parsed
            extra_trace.append({"stage": "JUDGE", "segment_id": seg.segment_id,
                                "action": "expand_frames", "n_frames": len(more)})

        if bootstrap:
            for g in parsed.get("policy_gaps") or []:
                proposals.append({
                    "segment_id": seg.segment_id, "category": g.get("category"),
                    "kind": g.get("kind", "edge_case"),
                    "suggestion": g.get("suggestion", "") or "",
                    "rationale": g.get("rationale", "") or "",
                })
            # Structured term-level attributes: drafted directly in SIDE_FX (not
            # queued), so they are tagged separately from free-text gaps.
            for sa in parsed.get("structured_attributes") or []:
                proposals.append({"structured_attribute": {
                    "segment_id": seg.segment_id,
                    "category": sa.get("category"),
                    "name": sa.get("name"),
                    "levels": sa.get("levels") or {},
                    "description": sa.get("description"),
                }})

        seg_prec = [pr for pr in precedents if pr["segment_id"] == seg.segment_id]
        for j in parsed.get("judgements", []):
            cat = j.get("category")
            if cat not in cats:
                continue
            lbl = Label(
                label_id="", segment_id=seg.segment_id, category=cat,
                score=_clamp(j.get("score")), rationale=j.get("rationale", "") or "",
                cited_policy_ids=_normalise_pins(
                    list(j.get("cited_policy_ids") or []), version_by_id),
                evidence_attributes=pattrs, confidence=j.get("confidence"),
            )
            # CHECK (absorbed): precedent divergence -> issue on the trace
            prec_scores = [
                pl["score"] for pr in seg_prec for pl in pr["labels"]
                if pl.get("category") == cat and pl.get("score") is not None
            ]
            if prec_scores and all(abs(lbl.score - ps) >= 2 for ps in prec_scores):
                lbl.tool_trace.append({
                    "issue": "precedent_divergence", "segment_id": seg.segment_id,
                    "category": cat, "score": lbl.score,
                    "precedent_scores": prec_scores,
                    "note": "diverges from precedent; retained for human triage",
                })
            drafts.append(lbl)

    trace = state.get("tool_trace", []) + extra_trace + [{
        "stage": "JUDGE", "n_labels": len(drafts), "n_proposals": len(proposals),
    }]
    return {"draft_labels": drafts, "proposals": proposals, "tool_trace": trace}


def _side_fx(state: LabellingState) -> dict:
    """Side-effect slot: revise_ingestion (auto+log) / propose_policy_change
    (queued for human). In bootstrap mode the policy gaps found in JUDGE are
    enqueued as change requests here; the baseline labelling run fires nothing."""
    actions: list[dict] = []
    if state.get("bootstrap"):
        for p in state.get("proposals", []) or []:
            # Structured attribute: direct upsert (bootstrap drafting), not queued.
            sa = p.get("structured_attribute")
            if sa:
                if sa.get("category") and sa.get("name") and sa.get("levels"):
                    node = agent_tools.define_structured_attribute(
                        category=sa["category"], name=sa["name"],
                        levels=sa["levels"], description=sa.get("description"),
                    )
                    actions.append({"define_structured_attribute":
                                    f"{node.policy_id} levels={sorted(sa['levels'])}"})
                continue
            # Free-text gap: still human-gated via the change-request queue.
            cat = p.get("category")
            kind = p.get("kind", "edge_case")
            suggestion = p.get("suggestion", "")
            agent_tools.propose_policy_change(
                change=suggestion, rationale=p.get("rationale", ""),
                affected=[p["segment_id"]] if p.get("segment_id") else [],
                category=cat, node_type=kind,
            )
            actions.append({"propose_policy_change": f"[{kind}/{cat}] {suggestion}"[:80]})
    trace = state.get("tool_trace", []) + [{"stage": "SIDE_FX", "actions": actions}]
    return {"tool_trace": trace}


def _commit(state: LabellingState) -> dict:
    """Persist labels, refresh the rolling carry-over, advance the cursor."""
    drafts = state.get("draft_labels", [])
    window = state.get("window", [])
    base_trace = state.get("tool_trace", [])
    used_ids = [s.segment_id for s in window]

    committed: list[Label] = []
    for lbl in drafts:
        lbl.label_id = lbl.label_id or str(uuid4())
        lbl.used_segment_ids = [i for i in used_ids if i != lbl.segment_id]
        lbl.tool_trace = base_trace + lbl.tool_trace
        agent_tools.emit_label(lbl)
        committed.append(lbl)

    bits = [f"{lbl.segment_id}:{lbl.category.value}={lbl.score}" for lbl in committed]
    carry = (state.get("carry_over", "") + " | " + ", ".join(bits)).strip(" |")

    cursor = state["cursor"] + state.get("window_stride", 3)
    trace = base_trace + [{"stage": "COMMIT",
                           "committed": [lbl.label_id for lbl in committed]}]
    return {"cursor": cursor, "carry_over": carry, "draft_labels": [],
            "tool_trace": trace}


def _route(state: LabellingState) -> str:
    return "load" if state["cursor"] < len(state["all_segments"]) else END


# --------------------------------------------------------------------------- #
# graph construction / run
# --------------------------------------------------------------------------- #
def build_graph():
    """Compile the StateGraph: one node per stage, with a conditional edge from
    COMMIT that advances the window cursor and loops to LOAD until exhausted."""
    g = StateGraph(LabellingState)
    g.add_node("load", _load)
    g.add_node("retrieve", _retrieve)
    g.add_node("judge", _judge)
    g.add_node("side_fx", _side_fx)
    g.add_node("commit", _commit)

    g.add_edge(START, "load")
    g.add_edge("load", "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_edge("judge", "side_fx")
    g.add_edge("side_fx", "commit")
    g.add_conditional_edges("commit", _route, {"load": "load", END: END})
    return g.compile()


def label_video(video_id: str, bootstrap: bool = False) -> None:
    """Run the compiled graph across one video's shot windows.

    bootstrap=True disables precedent retrieval (no confirmed labels exist yet)
    and lets JUDGE propose policy gaps that SIDE_FX queues for human review."""
    from models import base_config
    from tools import storage

    segments = storage.get_segments(video_id)
    video = storage.get_video(video_id)
    utterances = storage.get_utterances(video_id)
    cfg = base_config().get("labelling", {})
    stride = int(cfg.get("window_stride", 3))

    init: LabellingState = {
        "video_id": video_id,
        "global_overview": (video.global_overview if video else "") or "",
        "bootstrap": bootstrap,
        "all_segments": segments,
        "utterances": utterances,
        "cursor": 0,
        "window_size": int(cfg.get("window_size", 5)),
        "window_stride": stride,
        "carry_over": "",
        "tool_trace": [],
    }
    n_windows = max(1, math.ceil(len(segments) / max(1, stride)))
    graph = build_graph()
    graph.invoke(init, config={"recursion_limit": n_windows * 6 + 10})

    from schemas.enums import VideoStatus
    storage.set_video_status(video_id, VideoStatus.LABELLED.value)
