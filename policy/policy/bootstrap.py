"""Bootstrap — establish the initial policy set.

Reuses the normal mechanism (labelling agent + propose_policy_change queue),
run intensively over bootstrap videos, then converged.

    seed(PEGI -> rubric nodes = v0) -> label bootstrap videos -> cluster
    proposals into edge-case candidates -> human review -> iterate until
    cross-sample label variance converges -> policy-set v1
"""
from __future__ import annotations

from schemas import Category


def seed_from_pegi(categories: list[Category] | None = None) -> None:
    """Generate v0 scoring-rubric nodes from PEGI criteria."""
    raise NotImplementedError


def run_bootstrap(video_ids: list[str]) -> None:
    """Drive the normal labelling loop over bootstrap data and converge to
    policy-set v1."""
    raise NotImplementedError
