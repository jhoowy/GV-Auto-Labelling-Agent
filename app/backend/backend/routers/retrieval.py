"""Retrieval / RAG skill endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from tools import retrieval

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
    """Precedent retrieval + confirmed labels."""
    raise NotImplementedError


@router.post("/qa")
def qa(req: QARequest):
    return {"answer": retrieval.qa(req.question, req.scope, req.target_id)}
