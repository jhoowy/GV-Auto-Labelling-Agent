"""Policy layer: tree management lives in tools.policy_store; this package
holds the bootstrap logic that establishes the initial policy set."""
from .bootstrap import run_bootstrap, seed_from_pegi

__all__ = ["seed_from_pegi", "run_bootstrap"]
