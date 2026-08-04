"""Read-only DB table browser service.

Backs the Web UI "DB" view: a whitelisted set of tables, row counts, and
paginated / substring-filtered row dumps. Never writes. Embedding (pgvector)
columns are summarised as the string ``"vector(<dim>)"`` rather than dumping the
raw array; JSON(B) columns are returned as-is.
"""
from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Text, func, or_

from db import SessionLocal
from db import models as m

# Whitelist: exposed name -> SQLAlchemy model. Anything else is rejected (404).
_TABLES: dict[str, type] = {
    "videos": m.Video,
    "segments": m.Segment,
    "labels": m.Label,
    "policies": m.Policy,
    "policy_sets": m.PolicySet,
    "policy_change_requests": m.PolicyChangeRequest,
    "utterances": m.Utterance,
}


def _is_vector(col) -> bool:
    return isinstance(col.type, Vector)


def _cell(obj: Any, col) -> Any:
    """Serialise one column value; summarise vectors, pass everything else."""
    val = getattr(obj, col.name)
    if _is_vector(col):
        if val is None:
            return None
        dim = getattr(col.type, "dim", None) or len(val)
        return f"vector({dim})"
    return val


def list_tables() -> list[dict]:
    """Whitelisted tables with their row counts."""
    with SessionLocal() as s:
        return [
            {"name": name, "count": s.query(func.count()).select_from(model).scalar()}
            for name, model in _TABLES.items()
        ]


def browse_table(
    name: str,
    page: int = 1,
    page_size: int = 50,
    q: str | None = None,
) -> dict:
    """Paginated rows of one whitelisted table, ordered by primary key.

    `q` is a case-insensitive substring filter applied across text columns.
    Raises KeyError if the table is not whitelisted.
    """
    if name not in _TABLES:
        raise KeyError(name)
    model = _TABLES[name]
    table = model.__table__
    columns = [c.name for c in table.columns]
    text_cols = [c for c in table.columns if isinstance(c.type, (String, Text))]
    pk_cols = list(table.primary_key.columns)

    with SessionLocal() as s:
        query = s.query(model)
        if q:
            like = f"%{q}%"
            conds = [c.ilike(like) for c in text_cols]
            if conds:
                query = query.filter(or_(*conds))
        total = query.count()
        query = query.order_by(*pk_cols) if pk_cols else query
        objs = query.offset((page - 1) * page_size).limit(page_size).all()
        rows = [{c.name: _cell(o, c) for c in table.columns} for o in objs]

    return {
        "columns": columns,
        "rows": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def list_policy_sets() -> list[dict]:
    """Policy-set snapshots, newest version first."""
    with SessionLocal() as s:
        objs = s.query(m.PolicySet).order_by(m.PolicySet.version.desc()).all()
        return [
            {
                "version": o.version,
                "policy_versions": o.policy_versions or {},
                "note": o.note,
            }
            for o in objs
        ]
