"""Retrieval service layer — the RAG skill.

Hybrid = pgvector dense + BM25 lexical. Called by both the agent tools and the
FastAPI routers so retrieval logic lives in one place.
"""
from __future__ import annotations

from schemas import Label, Policy, Segment


def search_policies(query: str, category: str | None = None, top_k: int = 8) -> list[Policy]:
    """Hybrid retrieval over policy nodes (rubric/attribute/edge)."""
    raise NotImplementedError


def find_similar_segments(
    segment: Segment, top_k: int = 5
) -> list[tuple[Segment, list[Label]]]:
    """Nearest segments + their confirmed labels (precedent lookup)."""
    raise NotImplementedError


def lookup_structured(ref: str, text: str) -> bool:
    """Lookup against a policy's attached data, e.g. a profanity word list."""
    raise NotImplementedError


def qa(question: str, scope: str = "datapoint", target_id: str | None = None) -> str:
    """Data Viewer Q&A. scope=datapoint -> single segment/video;
    scope=dataset -> agentic RAG (retrieve -> aggregate -> answer)."""
    raise NotImplementedError
