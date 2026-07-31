"""Blob store abstraction.

Large media (video, clips, keyframes, audio) live here; the DB only stores
the returned pointer/key. Default backend is the local filesystem; MinIO is
optional (BLOB_BACKEND=minio).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def url(self, key: str) -> str: ...


class LocalBlobStore:
    def __init__(self, root: str | None = None):
        self.root = Path(root or os.getenv("BLOB_LOCAL_DIR", "./blobs"))

    def put(self, key: str, data: bytes) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def url(self, key: str) -> str:
        return str((self.root / key).resolve())


def get_blob_store() -> BlobStore:
    backend = os.getenv("BLOB_BACKEND", "local")
    if backend == "local":
        return LocalBlobStore()
    raise NotImplementedError(f"blob backend '{backend}' not wired")
