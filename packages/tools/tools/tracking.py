"""Node -> segment tracking (derived from existing labels; read-only).

Aggregates the JUDGE's per-label decision trace back into the policy nodes that
drove it, so a reviewer can click a decision-tree rule or an attribute value and
see which segments were labelled through it. Nothing is stored: both queries
read `labels` (+ join `segments` for the video id).

Trace shape (written by labelling/graph.py):
  - tool_trace = [{"decision": {"selected", "extracted", "rule_index",
    "rule_note", "score"}}]  -> rule_index picks the fired decision-tree rule.
  - evidence_attributes = one entry per defined attribute; a selected attribute
    carries its extracted `value`, unselected ones store value == "".
"""
from __future__ import annotations

from sqlalchemy import text

from db import SessionLocal

# Aggregators cap their result set; a full page equal to the cap signals the
# caller (endpoint/UI) that more matches may exist.
RESULT_CAP = 200


def _seg_row(segment_id: str, video_id: str, score: int) -> dict:
    """Shape one matched label into the segment record the UI consumes."""
    return {"segment_id": segment_id, "video_id": video_id, "score": score}


def segments_for_rule(category: str, rule_index: int) -> list[dict]:
    """Segments whose label trajectory fired decision-tree rule `rule_index`
    for `category`. Compared as text on tool_trace[0].decision.rule_index so a
    null (no rule matched) never coerces. Capped at RESULT_CAP, ordered by id."""
    sql = text(
        """
        SELECT l.segment_id AS segment_id, s.video_id AS video_id,
               l.score AS score
        FROM labels l
        JOIN segments s ON s.segment_id = l.segment_id
        WHERE l.category = :category
          AND (l.tool_trace #>> '{0,decision,rule_index}') = :rule_index
        ORDER BY l.label_id
        LIMIT :cap
        """
    )
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {"category": category, "rule_index": str(rule_index), "cap": RESULT_CAP},
        ).mappings()
        return [_seg_row(r["segment_id"], r["video_id"], r["score"]) for r in rows]


def segments_for_attribute_value(
    category: str, attribute: str, value: str,
) -> list[dict]:
    """Segments whose label recorded `attribute == value` for `category`. Matches
    any evidence_attributes entry with that key/value, compared as text (`->>`)
    so numeric/bool values coerce. Empty value matches nothing. Capped at
    RESULT_CAP, ordered by id."""
    if value == "":
        return []
    sql = text(
        """
        SELECT l.segment_id AS segment_id, s.video_id AS video_id,
               l.score AS score
        FROM labels l
        JOIN segments s ON s.segment_id = l.segment_id
        WHERE l.category = :category
          AND EXISTS (
              SELECT 1
              FROM jsonb_array_elements(l.evidence_attributes) e
              WHERE e->>'key' = :attribute AND e->>'value' = :value
          )
        ORDER BY l.label_id
        LIMIT :cap
        """
    )
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "category": category,
                "attribute": attribute,
                "value": value,
                "cap": RESULT_CAP,
            },
        ).mappings()
        return [_seg_row(r["segment_id"], r["video_id"], r["score"]) for r in rows]
