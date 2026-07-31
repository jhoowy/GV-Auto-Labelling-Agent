"""FastAPI app — HTTP surface over the shared service layer.

Routers add no business logic; they call tools.* (the same functions the agent
tools call). Consumers: the Next.js UI and external callers.
"""
from __future__ import annotations

from fastapi import FastAPI

from .routers import data, monitoring, policy, retrieval, runs

app = FastAPI(title="Video Labelling — Content Moderation PoC")

app.include_router(data.router)
app.include_router(retrieval.router)
app.include_router(policy.router)
app.include_router(runs.router)
app.include_router(monitoring.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
