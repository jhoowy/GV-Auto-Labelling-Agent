"""Embedding helpers. Text and image use dedicated single-modality models,
selected by profile."""
from __future__ import annotations

from models import get_image_embedder, get_text_embedder


def embed_text(texts: list[str]) -> list[list[float]]:
    return get_text_embedder().embed(texts)


def embed_images(frame_blobs: list[str]) -> list[list[float]]:
    return get_image_embedder().embed(frame_blobs)
