"""Retrieval / RAG skill endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools import retrieval, storage

router = APIRouter(prefix="/api/search", tags=["retrieval"])


class PolicyQuery(BaseModel):
    query: str
    category: str | None = None


class QARequest(BaseModel):
    question: str
    scope: str = "datapoint"      # datapoint | dataset
    target_id: str | None = None


@router.post("/policies")
def search_policies(q: PolicyQuery):
    return retrieval.search_policies(q.query, q.category)


@router.post("/segments")
def search_segments(segment_id: str):
    """Precedent retrieval: nearest segments + their confirmed labels."""
    seg = storage.get_segment(segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown segment {segment_id}")
    results = retrieval.find_similar_segments(seg)
    return [{"segment": s, "labels": labels} for s, labels in results]


@router.post("/qa")
def qa(req: QARequest):
    return {"answer": retrieval.qa(req.question, req.scope, req.target_id)}
