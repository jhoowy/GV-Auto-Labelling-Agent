"""Config + profile loader.

Base config (config/config.yaml) is profile-independent. Per-role model
providers come from config/profiles/<MODEL_PROFILE>.yaml. Profiles are looked
up here, never hard-coded, so the same code runs under different
prompt/model sets.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    return _load_yaml(_CONFIG_DIR / "config.yaml")


@lru_cache(maxsize=8)
def profile(name: str | None = None) -> dict[str, Any]:
    name = name or os.getenv("MODEL_PROFILE", "regular")
    return _load_yaml(_CONFIG_DIR / "profiles" / f"{name}.yaml")


def role_spec(role: str, profile_name: str | None = None) -> dict[str, Any]:
    """Return {provider, model, ...} for a model role (mllm/asr/...)."""
    spec = profile(profile_name).get(role)
    if spec is None:
        raise KeyError(f"model role '{role}' not defined in profile")
    return spec
