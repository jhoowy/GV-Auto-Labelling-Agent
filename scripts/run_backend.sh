#!/usr/bin/env bash
# Launch the FastAPI backend.
#
# Uses a neutral --app-dir so the editable package `db` resolves to db/db.
# From the repo root, uvicorn puts the cwd on sys.path and the repo-root `db/`
# directory (a namespace package) would otherwise shadow the real `db` package.
set -euo pipefail

VENV="${VENV:-/tmp/iji-video-labelling-venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PROFILE="${MODEL_PROFILE:-local}"
export BLOB_LOCAL_DIR="${BLOB_LOCAL_DIR:-$REPO/blobs}"

exec "$VENV/bin/uvicorn" backend.main:app --app-dir /tmp --host "$HOST" --port "$PORT" "$@"
