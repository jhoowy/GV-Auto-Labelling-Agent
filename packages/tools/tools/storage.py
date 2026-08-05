"""Persistence service layer over Postgres.

Thin repository functions used by ingestion, labelling, and the backend.
Maps between Pydantic contracts (schemas) and the SQLAlchemy models (db).
"""
from __future__ import annotations

from db import SessionLocal
from db import models as m
from schemas import Attribute, Label, Segment, Utterance, Video, VideoMetadata
from schemas.enums import VideoStatus

# Node -> segment tracking aggregators (read-only, derived from labels).
from tools.tracking import (  # noqa: F401  (re-exported for the service layer)
    segments_for_attribute_value,
    segments_for_rule,
)


def _apply_segment(obj: "m.Segment", seg: Segment) -> None:
    obj.video_id = seg.video_id
    obj.idx = seg.idx
    obj.t_start = seg.t_start
    obj.t_end = seg.t_end
    obj.clip_blob = seg.clip_blob
    obj.transcript = seg.transcript
    obj.summary = seg.summary
    obj.base_attributes = [a.model_dump() for a in seg.base_attributes]
    obj.text_embedding = seg.text_embedding
    obj.image_embedding = seg.image_embedding
    obj.status = seg.status


def upsert_video(video: Video) -> None:
    with SessionLocal() as s:
        obj = s.get(m.Video, video.video_id) or m.Video(video_id=video.video_id)
        # Merge, don't clobber: ingestion rebuilds a thin VideoMetadata (title /
        # channel / description / thumbnail), but the row may already carry rich
        # collected fields (dataset, tags, default_audio_language, pegi GT). Keep
        # existing keys and overlay only non-empty incoming values so a re-ingest
        # never wipes collection metadata.
        merged = dict(obj.metadata_json or {})
        for k, v in video.metadata.model_dump().items():
            if v not in (None, "", [], {}):
                merged[k] = v
        obj.metadata_json = merged
        obj.duration_s = video.duration_s
        obj.source_blob = video.source_blob
        obj.global_overview = video.global_overview
        obj.status = getattr(video.status, "value", video.status)
        obj.text_embedding = video.text_embedding
        s.merge(obj)
        s.commit()


def get_video(video_id: str) -> Video | None:
    with SessionLocal() as s:
        obj = s.get(m.Video, video_id)
        if obj is None:
            return None
        return Video(
            video_id=obj.video_id,
            metadata=VideoMetadata(**(obj.metadata_json or {})),
            duration_s=obj.duration_s,
            source_blob=obj.source_blob,
            global_overview=obj.global_overview,
            status=VideoStatus(obj.status),
            text_embedding=obj.text_embedding,
        )


def upsert_segments(segments: list[Segment]) -> None:
    with SessionLocal() as s:
        for seg in segments:
            obj = s.get(m.Segment, seg.segment_id)
            if obj is None:
                obj = m.Segment(segment_id=seg.segment_id)
                s.add(obj)
            _apply_segment(obj, seg)
        s.commit()


def replace_segments(video_id: str, segments: list[Segment]) -> None:
    """Replace a video's segments (delete-then-insert) — used by resegmentation,
    where new bounds change the segment count so stale rows must be dropped."""
    with SessionLocal() as s:
        s.query(m.Segment).filter_by(video_id=video_id).delete()
        for seg in segments:
            obj = m.Segment(segment_id=seg.segment_id)
            s.add(obj)
            _apply_segment(obj, seg)
        s.commit()


def get_segments(video_id: str) -> list[Segment]:
    """Ordered by idx."""
    with SessionLocal() as s:
        objs = (s.query(m.Segment).filter_by(video_id=video_id)
                .order_by(m.Segment.idx).all())
        return [
            Segment(
                segment_id=o.segment_id, video_id=o.video_id, idx=o.idx,
                t_start=o.t_start, t_end=o.t_end, clip_blob=o.clip_blob,
                transcript=o.transcript, summary=o.summary,
                base_attributes=[Attribute(**a) for a in (o.base_attributes or [])],
                text_embedding=o.text_embedding, image_embedding=o.image_embedding,
                status=o.status,
            )
            for o in objs
        ]


def replace_utterances(video_id: str, utterances: list[Utterance]) -> None:
    """Replace a video's word-level ASR utterances (ingestion output)."""
    with SessionLocal() as s:
        s.query(m.Utterance).filter_by(video_id=video_id).delete()
        for u in utterances:
            s.add(m.Utterance(video_id=u.video_id, idx=u.idx,
                              t_start=u.t_start, t_end=u.t_end, text=u.text))
        s.commit()


def get_utterances(video_id: str) -> list[Utterance]:
    """Word-level utterances on the video timeline, ordered by idx."""
    with SessionLocal() as s:
        objs = (s.query(m.Utterance).filter_by(video_id=video_id)
                .order_by(m.Utterance.idx).all())
        return [Utterance(video_id=o.video_id, idx=o.idx, t_start=o.t_start,
                          t_end=o.t_end, text=o.text) for o in objs]


def save_label(label: Label) -> None:
    """Persist a label. Append-only; revisions keep history."""
    with SessionLocal() as s:
        s.merge(m.Label(
            label_id=label.label_id, segment_id=label.segment_id,
            category=getattr(label.category, "value", label.category),
            score=label.score, rationale=label.rationale,
            cited_policy_ids=label.cited_policy_ids,
            evidence_attributes=[a.model_dump() for a in label.evidence_attributes],
            used_segment_ids=label.used_segment_ids, tool_trace=label.tool_trace,
            confidence=label.confidence, human_verified=label.human_verified,
        ))
        s.commit()


def revise_ingestion(segment_id: str, patch: dict) -> None:
    """Apply an agent correction to ingestion output (original preserved via
    label/segment history)."""
    with SessionLocal() as s:
        obj = s.get(m.Segment, segment_id)
        if obj is None:
            raise KeyError(segment_id)
        for k, v in patch.items():
            setattr(obj, k, v)
        s.commit()


def set_video_status(video_id: str, status: str) -> None:
    """Advance a video's lifecycle status (e.g. -> 'labelled') without touching
    its metadata_json or embeddings."""
    with SessionLocal() as s:
        obj = s.get(m.Video, video_id)
        if obj is not None:
            obj.status = getattr(status, "value", status)
            s.commit()


# --- Backend read helpers (Data Viewer / monitoring). Append-only. ---

def list_videos() -> list[Video]:
    """All videos, ordered by video_id."""
    with SessionLocal() as s:
        objs = s.query(m.Video).order_by(m.Video.video_id).all()
        return [
            Video(
                video_id=o.video_id,
                metadata=VideoMetadata(**(o.metadata_json or {})),
                duration_s=o.duration_s,
                source_blob=o.source_blob,
                global_overview=o.global_overview,
                status=VideoStatus(o.status),
                text_embedding=o.text_embedding,
            )
            for o in objs
        ]


def list_videos_page(
    search: str | None = None,
    page: int = 1,
    page_size: int = 24,
    dataset: str | None = None,
    status: str | None = None,
    category: str | None = None,
    score_min: int | None = None,
    score_max: int | None = None,
) -> dict:
    """Paginated video cards for the gallery view.

    Each item carries the title (from metadata_json), duration, status and the
    segment count. `search` is a case-insensitive substring match on the title.
    `dataset` filters on metadata_json->>'dataset' (None = all datasets).

    `status` filters on the labelling lifecycle derived from segments/labels
    (not the video row's own status column): 'ingested' = has >=1 segment;
    'labelled' = has >=1 label via its segments; 'unlabelled' = has segments but
    no labels. `category` + `score_min`/`score_max` keep videos with >=1 label in
    that category whose score falls in the (inclusive) range; when only one bound
    is given the other defaults to 0/5, and a bare `category` matches any score.
    All filters compose SQL-side and `total` respects them.
    """
    from sqlalchemy import and_, func

    with SessionLocal() as s:
        seg_counts = (
            s.query(
                m.Segment.video_id.label("video_id"),
                func.count(m.Segment.segment_id).label("n"),
            )
            .group_by(m.Segment.video_id)
            .subquery()
        )
        q = s.query(m.Video, seg_counts.c.n).outerjoin(
            seg_counts, m.Video.video_id == seg_counts.c.video_id
        )
        if search:
            q = q.filter(m.Video.metadata_json["title"].astext.ilike(f"%{search}%"))
        if dataset:
            q = q.filter(m.Video.metadata_json["dataset"].astext == dataset)

        # Lifecycle status via EXISTS over segments/labels (not the video's own
        # status column). A label is tied to a video through its segment.
        has_segment = (
            s.query(m.Segment.segment_id)
            .filter(m.Segment.video_id == m.Video.video_id)
            .exists()
        )
        has_label = (
            s.query(m.Label.label_id)
            .join(m.Segment, m.Label.segment_id == m.Segment.segment_id)
            .filter(m.Segment.video_id == m.Video.video_id)
            .exists()
        )
        if status == "ingested":
            q = q.filter(has_segment)
        elif status == "labelled":
            q = q.filter(has_label)
        elif status == "unlabelled":
            q = q.filter(has_segment, ~has_label)

        # Category + score-range: keep videos with a matching label. A bare
        # category matches any score; otherwise clamp the range to [0, 5].
        if category:
            label_match = (
                s.query(m.Label.label_id)
                .join(m.Segment, m.Label.segment_id == m.Segment.segment_id)
                .filter(
                    m.Segment.video_id == m.Video.video_id,
                    m.Label.category == category,
                )
            )
            if score_min is not None or score_max is not None:
                lo = score_min if score_min is not None else 0
                hi = score_max if score_max is not None else 5
                label_match = label_match.filter(
                    and_(m.Label.score >= lo, m.Label.score <= hi)
                )
            q = q.filter(label_match.exists())

        q = q.order_by(m.Video.video_id)
        total = q.count()
        rows = q.offset((page - 1) * page_size).limit(page_size).all()
        items = [
            {
                "video_id": v.video_id,
                "title": (v.metadata_json or {}).get("title"),
                "duration_s": v.duration_s,
                "thumbnail_url": f"/api/videos/{v.video_id}/thumbnail",
                "status": v.status,
                "n_segments": n or 0,
            }
            for v, n in rows
        ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


def get_segment(segment_id: str) -> Segment | None:
    """A single shot by id, or None."""
    with SessionLocal() as s:
        o = s.get(m.Segment, segment_id)
        if o is None:
            return None
        return Segment(
            segment_id=o.segment_id, video_id=o.video_id, idx=o.idx,
            t_start=o.t_start, t_end=o.t_end, clip_blob=o.clip_blob,
            transcript=o.transcript, summary=o.summary,
            base_attributes=[Attribute(**a) for a in (o.base_attributes or [])],
            text_embedding=o.text_embedding, image_embedding=o.image_embedding,
            status=o.status,
        )


def list_labels(segment_id: str | None = None) -> list[Label]:
    """Labels with their full trace; optionally scoped to one segment."""
    from schemas.enums import Category
    with SessionLocal() as s:
        q = s.query(m.Label)
        if segment_id is not None:
            q = q.filter_by(segment_id=segment_id)
        objs = q.order_by(m.Label.label_id).all()
        return [
            Label(
                label_id=o.label_id, segment_id=o.segment_id,
                category=Category(o.category), score=o.score,
                rationale=o.rationale, cited_policy_ids=o.cited_policy_ids or [],
                evidence_attributes=[
                    Attribute(**a) for a in (o.evidence_attributes or [])
                ],
                used_segment_ids=o.used_segment_ids or [],
                tool_trace=o.tool_trace or [],
                confidence=o.confidence, human_verified=o.human_verified,
            )
            for o in objs
        ]
