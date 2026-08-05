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

_WT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = str(_WT_ROOT)

sys.path[:] = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]

# The shared editable install points at a DIFFERENT checkout, so make THIS
# worktree's own source roots authoritative for the packages under test —
# otherwise `import labelling` / `import tools` resolve to the installed tree,
# not the code being validated here. (Unchanged packages — db, schemas, models,
# … — keep resolving via the editable finder.)
for _root in (_WT_ROOT / "packages" / "tools", _WT_ROOT / "labelling"):
    _sp = str(_root)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
sys.modules.pop("db", None)
