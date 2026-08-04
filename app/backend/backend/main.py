"""FastAPI app — HTTP surface over the shared service layer.

Routers add no business logic; they call tools.* (the same functions the agent
tools call). Consumers: the Next.js UI and external callers.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import data, db, monitoring, policy, retrieval, runs

app = FastAPI(title="Video Labelling — Content Moderation PoC")

# Next.js dev origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(db.router)
app.include_router(retrieval.router)
app.include_router(policy.router)
app.include_router(runs.router)
app.include_router(monitoring.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
