"""Langfuse observability, wired as an opt-in no-op.

Tracing turns on only when `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set
(self-hosted: also `LANGFUSE_HOST`). When they are absent — or the `langfuse`
package is not installed — every hook here degrades to a plain passthrough, so
the labelling graph and provider calls run unchanged and untraced.

Two integration points:
- `get_langfuse_handler()` returns a LangChain `CallbackHandler` to inject into
  the compiled graph's `invoke` config, giving one trace per video with a span
  per LangGraph node (LOAD/RETRIEVE/JUDGE/SIDE_FX/COMMIT).
- `@observe(...)` decorates the direct-SDK LLM calls (Gemini orchestrator,
  OpenAI policy author) that bypass LangChain, so their I/O shows up as nested
  generation spans instead of opaque gaps under the node.
"""

from __future__ import annotations

import os
from typing import Callable


def _enabled() -> bool:
    """Tracing is active only when both Langfuse keys are present in the env."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def observe(*, name: str | None = None, as_type: str | None = None) -> Callable:
    """Decorator: wrap a function as a Langfuse span, else return it unchanged.

    Evaluated at import time, so the env must be set before the module using it
    is imported (the `set_api_key.sh` flow does this). `as_type="generation"`
    marks an LLM call so its input/output render as a generation in the trace.
    """

    def deco(fn: Callable) -> Callable:
        if not _enabled():
            return fn
        try:
            from langfuse import observe as _observe
        except Exception:
            return fn
        try:
            return _observe(name=name or fn.__name__, as_type=as_type)(fn)
        except Exception:
            return fn

    return deco


def get_langfuse_handler():
    """Return a LangChain `CallbackHandler` (reads keys/host from the env), or
    `None` when tracing is off or unavailable — safe to spread into a callbacks
    list either way. Per-trace attributes (name/metadata) are set on the graph's
    `invoke` config, not here — the v4 handler constructor takes no trace args."""
    if not _enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return None
    try:
        return CallbackHandler()
    except Exception:
        return None


def flush() -> None:
    """Flush buffered events (Langfuse ships them in a background batch). Called
    at the end of a labelling run so short-lived processes do not drop traces."""
    if not _enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
