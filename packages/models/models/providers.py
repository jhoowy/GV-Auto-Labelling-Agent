"""Provider-agnostic model clients.

Under MODEL_PROFILE=local these talk to the vLLM servers from
scripts/serve_vllm.sh (OpenAI-compatible). Each factory reads the active
profile (base_url + served_model_name), so switching endpoints is config-only.

Verified request shapes (Qwen local models):
  text embed  : POST /v1/embeddings input=[texts]
  visual embed: POST /v1/embeddings messages=[{image_url}]  (SDK input= is text-only)
  ASR         : chat + audio_url, output "language <L><asr_text><TEXT>"
  Omni MLLM   : chat + video_url + audio_url  (audio passed separately)
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from .config import role_spec


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _spec(role: str, profile: str | None):
    s = role_spec(role, profile)
    return s["base_url"], s["served_model_name"]


class MLLMClient(Protocol):
    async def describe(self, video_path: str, audio_path: str, prompt: str) -> str: ...


class ASRClient(Protocol):
    async def transcribe(self, audio_path: str) -> tuple[str, list[dict]]: ...


class Embedder(Protocol):
    async def embed(self, items: list[Any]) -> list[list[float]]: ...


class QwenTextEmbedder:
    def __init__(self, profile: str | None = None):
        base, self.model = _spec("text_embedding", profile)
        self.client = AsyncOpenAI(base_url=base, api_key="x")

    async def embed(self, items: list[str]) -> list[list[float]]:
        r = await self.client.embeddings.create(model=self.model, input=items)
        return [d.embedding for d in r.data]


class QwenVisualEmbedder:
    """Visual embedding of a clip (whole video). Multimodal input goes through
    the chat-style messages field, which the OpenAI SDK's embeddings.create does
    not expose, so POST directly."""

    def __init__(self, profile: str | None = None):
        self.base, self.model = _spec("image_embedding", profile)

    async def embed(self, items: list[str]) -> list[list[float]]:
        """items = clip file paths (av); returns one visual vector per clip."""
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=180) as c:
            for path in items:
                url = f"data:video/mp4;base64,{_b64(path)}"
                r = await c.post(f"{self.base}/embeddings", json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url": url}}]}],
                })
                r.raise_for_status()
                out.append(r.json()["data"][0]["embedding"])
        return out


class QwenASRAligner:
    """ASR + word-level timestamps via the isolated qwen-asr HTTP server
    (transcript and forced alignment in one call — see scripts/asr_align_server.py)."""

    def __init__(self, profile: str | None = None):
        self.base = role_spec("asr", profile)["base_url"]

    async def transcribe(self, audio_path: str,
                         language: str | None = None) -> tuple[str, list[dict]]:
        """Return (text, words) where words = [{text, t_start, t_end}, ...]."""
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{self.base}/transcribe",
                             json={"audio_path": audio_path, "language": language})
            r.raise_for_status()
            j = r.json()
            return j["text"], j["words"]


class QwenOmni:
    def __init__(self, profile: str | None = None):
        base, self.model = _spec("mllm", profile)
        self.client = AsyncOpenAI(base_url=base, api_key="x")

    async def describe(self, video_path: str, audio_path: str, prompt: str) -> str:
        video = f"data:video/mp4;base64,{_b64(video_path)}"
        audio = f"data:audio/wav;base64,{_b64(audio_path)}"
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=2048, temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video}},
                {"type": "audio_url", "audio_url": {"url": audio}},
                {"type": "text", "text": prompt}]}])
        return r.choices[0].message.content or ""

    async def chat(self, prompt: str) -> str:
        """Text-only chat (e.g. aggregate shot summaries into a global summary)."""
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=1024, temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


def get_agent_llm(profile_name: str | None = None):
    """LangChain chat model for the labelling orchestrator (not ingestion)."""
    spec = role_spec("agent_llm", profile_name)
    raise NotImplementedError(f"wire init_chat_model for {spec}")


def get_text_embedder(profile_name: str | None = None) -> Embedder:
    return QwenTextEmbedder(profile_name)


def get_image_embedder(profile_name: str | None = None) -> Embedder:
    return QwenVisualEmbedder(profile_name)


def get_mllm(profile_name: str | None = None) -> MLLMClient:
    return QwenOmni(profile_name)


def get_asr(profile_name: str | None = None) -> ASRClient:
    return QwenASRAligner(profile_name)
