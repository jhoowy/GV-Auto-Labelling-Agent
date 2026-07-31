"""Provider-agnostic model clients.

Standard chat/text-embedding roles go through LangChain's unified interface.
Roles LangChain doesn't cover well — the vision+audio omni MLLM, ASR, image
embedding — get a thin local adapter. All factories read the active profile,
so switching proprietary <-> local vLLM is a config change.
"""
from __future__ import annotations

from typing import Any, Protocol

from .config import role_spec


class MLLMClient(Protocol):
    def describe(self, frames: list[str], audio: str | None, prompt: str) -> dict:
        """Vision+audio -> {summary, base_attributes}."""


class ASRClient(Protocol):
    def transcribe(self, audio_blob: str) -> str: ...


class Embedder(Protocol):
    def embed(self, items: list[Any]) -> list[list[float]]: ...


def get_agent_llm(profile_name: str | None = None):
    """LangChain chat model for the orchestrator."""
    spec = role_spec("agent_llm", profile_name)
    raise NotImplementedError(f"wire init_chat_model for {spec}")


def get_text_embedder(profile_name: str | None = None) -> Embedder:
    spec = role_spec("text_embedding", profile_name)
    raise NotImplementedError(f"wire text embedder for {spec}")


def get_image_embedder(profile_name: str | None = None) -> Embedder:
    spec = role_spec("image_embedding", profile_name)
    raise NotImplementedError(f"wire image embedder for {spec}")


def get_mllm(profile_name: str | None = None) -> MLLMClient:
    spec = role_spec("mllm", profile_name)
    raise NotImplementedError(f"wire vision+audio MLLM for {spec}")


def get_asr(profile_name: str | None = None) -> ASRClient:
    spec = role_spec("asr", profile_name)
    raise NotImplementedError(f"wire ASR for {spec}")
