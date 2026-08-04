"""Embedding helpers. Text and image use dedicated single-modality models,
selected by profile."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable

from models import get_image_embedder, get_text_embedder


def _run(coro: Awaitable[Any]) -> Any:
    """Run an async embedder call from sync code, even inside a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def embed_text(texts: list[str]) -> list[list[float]]:
    return _run(get_text_embedder().embed(texts))


def embed_images(frame_blobs: list[str]) -> list[list[float]]:
    return _run(get_image_embedder().embed(frame_blobs))


def embed_query(text: str) -> list[float]:
    """Embed a single query string; returns one dense text vector."""
    return embed_text([text])[0]
