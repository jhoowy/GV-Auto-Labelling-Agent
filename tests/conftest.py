"""Test bootstrap.

The repo-root ``db/`` directory (alembic config + the ``db/db`` package) shadows
the editable ``db`` package whenever the repo root sits on ``sys.path``: the stock
``PathFinder`` resolves ``import db`` to the root ``db/`` namespace dir, which has
no ``SessionLocal``. The editable install's finder is appended to ``sys.meta_path``
and only wins once the root is off ``sys.path``. So drop the root/cwd entries and
evict any half-resolved ``db`` module before the test modules import anything.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

sys.path[:] = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
sys.modules.pop("db", None)
