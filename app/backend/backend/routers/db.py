"""Read-only DB browser endpoints — back the Web UI "DB" view."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tools import db_browser

router = APIRouter(prefix="/api/db", tags=["db"])


@router.get("/tables")
def list_tables():
    """Whitelisted tables with row counts."""
    return db_browser.list_tables()


@router.get("/tables/{name}")
def browse_table(name: str, page: int = 1, page_size: int = 50, q: str | None = None):
    """Paginated rows of a whitelisted table (read-only). Vector columns are
    summarised as "vector(<dim>)"; `q` filters case-insensitively across text
    columns. Unknown tables 404."""
    try:
        return db_browser.browse_table(name, page=page, page_size=page_size, q=q)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown table {name}")
