"""Retrieval service layer — the RAG skill.

Hybrid = pgvector dense + BM25 lexical. Called by both the agent tools and the
FastAPI routers so retrieval logic lives in one place.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import sqlalchemy as sa

from db import SessionLocal
from db import models as m
from models import base_config, get_agent_llm
from schemas import Attribute, Label, Policy, Segment
from schemas.enums import Category, PolicyType
from tools.embeddings import embed_query


# --- config -----------------------------------------------------------------

def _rcfg() -> dict:
    return base_config().get("retrieval", {})


# --- ORM -> contract mappers ------------------------------------------------

def _to_policy(o: "m.Policy") -> Policy:
    return Policy(
        policy_id=o.policy_id, type=PolicyType(o.type), category=Category(o.category),
        version=o.version, parent_id=o.parent_id, text=o.text,
        embedding=list(o.embedding) if o.embedding is not None else None,
        structured_ref=o.structured_ref, status=o.status,
    )


def _to_segment(o: "m.Segment") -> Segment:
    return Segment(
        segment_id=o.segment_id, video_id=o.video_id, idx=o.idx,
        t_start=o.t_start, t_end=o.t_end, clip_blob=o.clip_blob,
        transcript=o.transcript, summary=o.summary,
        base_attributes=[Attribute(**a) for a in (o.base_attributes or [])],
        text_embedding=o.text_embedding, image_embedding=o.image_embedding,
        status=o.status,
    )


def _to_label(o: "m.Label") -> Label:
    return Label(
        label_id=o.label_id, segment_id=o.segment_id, category=Category(o.category),
        score=o.score, rationale=o.rationale, cited_policy_ids=o.cited_policy_ids or [],
        evidence_attributes=[Attribute(**a) for a in (o.evidence_attributes or [])],
        used_segment_ids=o.used_segment_ids or [], tool_trace=o.tool_trace or [],
        confidence=o.confidence, human_verified=o.human_verified,
    )


# --- score fusion -----------------------------------------------------------

def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise scores to [0, 1] so dense and lexical are comparable."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# --- policies ---------------------------------------------------------------

def search_policies(query: str, category: str | None = None, top_k: int = 8) -> list[Policy]:
    """Hybrid retrieval over policy nodes (rubric/attribute/edge).

    Dense = pgvector cosine over Policy.embedding (query via embed_query);
    lexical = Postgres full-text ts_rank over the 'simple' to_tsvector index on
    Policy.text. Scores are min-max normalised per modality then fused with
    config retrieval.hybrid_alpha (weight on dense). Only active nodes are
    considered, optionally filtered to `category`.
    """
    cat = getattr(category, "value", category)
    alpha = float(_rcfg().get("hybrid_alpha", 0.5))
    fetch = max(top_k * 4, top_k)

    with SessionLocal() as s:
        # dense candidates (cosine similarity = 1 - cosine distance)
        dist = m.Policy.embedding.cosine_distance(embed_query(query))
        dq = s.query(m.Policy.policy_id, dist.label("dist")).filter(
            m.Policy.status == "active", m.Policy.embedding.isnot(None))
        if cat:
            dq = dq.filter(m.Policy.category == cat)
        dense = {pid: 1.0 - float(d) for pid, d in dq.order_by(dist).limit(fetch).all()}

        # lexical candidates via the functional 'simple' tsvector index
        lex_sql = (
            "SELECT policy_id, "
            "ts_rank(to_tsvector('simple', text), plainto_tsquery('simple', :q)) AS rank "
            "FROM policies "
            "WHERE status = 'active' "
            "AND to_tsvector('simple', text) @@ plainto_tsquery('simple', :q)"
            + (" AND category = :cat" if cat else "")
            + " ORDER BY rank DESC LIMIT :lim"
        )
        params: dict = {"q": query, "lim": fetch}
        if cat:
            params["cat"] = cat
        lex = {r.policy_id: float(r.rank) for r in s.execute(sa.text(lex_sql), params)}

        dnorm, lnorm = _minmax(dense), _minmax(lex)
        ids = set(dense) | set(lex)
        fused = {
            pid: alpha * dnorm.get(pid, 0.0) + (1.0 - alpha) * lnorm.get(pid, 0.0)
            for pid in ids
        }
        ranked = sorted(ids, key=lambda p: fused[p], reverse=True)[:top_k]
        if not ranked:
            return []
        by_id = {
            o.policy_id: o
            for o in s.query(m.Policy).filter(m.Policy.policy_id.in_(ranked)).all()
        }
        return [_to_policy(by_id[p]) for p in ranked if p in by_id]


# --- precedent lookup -------------------------------------------------------

def _confirmed_labels(s, segment_id: str) -> list[Label]:
    """Committed/confirmed labels for a segment. Persisted labels are treated as
    committed (they are only written at the COMMIT stage); human_verified ones
    are surfaced first as the stronger precedent."""
    objs = (s.query(m.Label).filter(m.Label.segment_id == segment_id)
            .order_by(m.Label.human_verified.desc()).all())
    return [_to_label(o) for o in objs]


def find_similar_segments(
    segment: Segment, top_k: int = 5
) -> list[tuple[Segment, list[Label]]]:
    """Nearest segments by text_embedding cosine (excluding self) + each
    neighbour's confirmed labels. This precedent pairing is the primary
    consistency mechanism."""
    qvec = segment.text_embedding
    if qvec is None:
        text = segment.summary or segment.transcript or ""
        if not text.strip():
            return []
        qvec = embed_query(text)

    with SessionLocal() as s:
        dist = m.Segment.text_embedding.cosine_distance(qvec)
        rows = (s.query(m.Segment)
                .filter(m.Segment.text_embedding.isnot(None),
                        m.Segment.segment_id != segment.segment_id)
                .order_by(dist).limit(top_k).all())
        return [(_to_segment(o), _confirmed_labels(s, o.segment_id)) for o in rows]


# --- structured data lookup -------------------------------------------------

@lru_cache(maxsize=64)
def _load_word_list(ref: str) -> tuple[str, ...]:
    """Resolve a structured_ref to a list of terms. `ref` may be a filesystem
    path (JSON array / {"words": [...]} / newline-delimited) or a policy_id whose
    structured_ref points at such a file. Returns () if it can't be resolved."""
    p = Path(ref)
    if not p.exists():
        # ref may be a policy_id; resolve its structured_ref pointer instead
        with SessionLocal() as s:
            pol = s.get(m.Policy, ref)
        if pol is None or not pol.structured_ref or pol.structured_ref == ref:
            return ()
        p = Path(pol.structured_ref)
        if not p.exists():
            return ()
    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return tuple(line.strip() for line in raw.splitlines() if line.strip())
    if isinstance(data, dict):
        data = data.get("words") or list(data.keys())
    return tuple(str(x) for x in data)


def lookup_structured(ref: str, text: str) -> bool:
    """Lookup against a policy's attached data, e.g. a profanity word list.
    True if any resolved term matches `text` (case-insensitive, word-boundary)."""
    if not ref or not text:
        return False
    words = _load_word_list(ref)
    if not words:
        return False
    hay = text.lower()
    for w in words:
        w = w.strip().lower()
        if w and re.search(r"\b" + re.escape(w) + r"\b", hay):
            return True
    return False


# --- Q&A --------------------------------------------------------------------

def _qa_prompt(question: str, context: str) -> str:
    return (
        "You are a content-moderation analyst. Answer the question using ONLY the "
        "context below. If the context is insufficient, say so.\n\n"
        f"Context:\n{context or '(no context found)'}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _answer(prompt: str, fallback: str = "") -> str:
    """Invoke the agent LLM; degrade to the retrieved context if it is not
    configured/reachable (so an import/dry check needs no running server)."""
    try:
        llm = get_agent_llm()
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:  # noqa: BLE001 - graceful degradation
        return fallback or f"No answer available (agent LLM unavailable): {e}"


def _datapoint_context(target_id: str | None) -> str:
    if not target_id:
        return ""
    parts: list[str] = []
    with SessionLocal() as s:
        seg = s.get(m.Segment, target_id)
        if seg is not None:
            if seg.summary:
                parts.append(f"Summary: {seg.summary}")
            if seg.transcript:
                parts.append(f"Transcript: {seg.transcript}")
            return "\n".join(parts)
        vid = s.get(m.Video, target_id)
        if vid is not None:
            if vid.global_overview:
                parts.append(f"Overview: {vid.global_overview}")
            segs = (s.query(m.Segment).filter_by(video_id=target_id)
                    .order_by(m.Segment.idx).all())
            parts += [f"[shot {o.idx}] {o.summary}" for o in segs if o.summary]
    return "\n".join(parts)


def _dataset_context(question: str) -> str:
    parts: list[str] = []
    try:
        for p in search_policies(question, top_k=4):
            parts.append(f"[policy {p.policy_id}] {p.text}")
    except Exception:  # noqa: BLE001 - retrieval is best-effort here
        pass
    try:
        qvec = embed_query(question)
        with SessionLocal() as s:
            dist = m.Segment.text_embedding.cosine_distance(qvec)
            rows = (s.query(m.Segment)
                    .filter(m.Segment.text_embedding.isnot(None))
                    .order_by(dist).limit(6).all())
            parts += [f"[segment {o.segment_id}] {o.summary}" for o in rows if o.summary]
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(parts)


def qa(question: str, scope: str = "datapoint", target_id: str | None = None) -> str:
    """Data Viewer Q&A. scope=datapoint -> single segment/video;
    scope=dataset -> agentic RAG (retrieve -> aggregate -> answer)."""
    if scope == "datapoint":
        context = _datapoint_context(target_id)
    elif scope == "dataset":
        context = _dataset_context(question)
    else:
        raise ValueError(f"unknown scope: {scope!r}")
    return _answer(_qa_prompt(question, context), fallback=context)
