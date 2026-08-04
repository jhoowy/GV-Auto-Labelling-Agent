"""Provider-agnostic model clients.

Under MODEL_PROFILE=local these talk to the vLLM servers from
scripts/serve_vllm.sh (OpenAI-compatible). Each factory dispatches on the role
spec's `provider` field: `vllm` -> the Qwen clients below, `google-genai` ->
GeminiOrchestrator (agent_llm). Any other provider (openai/whisper/anthropic/…)
is not wired in this PoC and raises NotImplementedError — only `local` runs.

Verified request shapes (Qwen local models):
  text embed  : POST /v1/embeddings input=[texts]
  visual embed: POST /v1/embeddings messages=[{image_url}]  (SDK input= is text-only)
  ASR         : chat + audio_url, output "language <L><asr_text><TEXT>"
  Omni MLLM   : chat + video_url, audio delivered via use_audio_in_video on the
                av clip (mm_processor_kwargs). Served by the isolated vllm-omni
                venv (scripts/serve_omni.sh); stock vLLM cannot ingest the audio
                track of a video, and a separate audio_url mismatches Qwen3-Omni's
                video/audio item count.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from .config import role_spec

_SEGMENT_SYS = (
    "You segment a video into contiguous, semantically coherent scenes based on "
    "general scene transitions — changes in setting, activity, subject, or visual/"
    "audio context. Prefer scenes roughly 5-30 seconds long."
)
_RECONCILE_SYS = (
    "You reconcile scene boundaries from two overlapping analyses of the same "
    "video region into a single contiguous segmentation."
)


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def sec_to_hms(t: float) -> str:
    """seconds -> H:MM:SS.ff"""
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _hms_to_sec(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    sec = 0.0
    for part in str(v).replace("s", "").split(":"):
        sec = sec * 60 + float(part)
    return sec


def _parse_scenes(text: str) -> list[dict]:
    """Extract [{start,end,summary}] from a model reply; times may be H:MM:SS.ff
    or plain seconds. Returns start/end in seconds."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for s in arr:
        if isinstance(s, dict) and {"start", "end"} <= s.keys():
            try:
                out.append({"start": _hms_to_sec(s["start"]), "end": _hms_to_sec(s["end"]),
                            "summary": str(s.get("summary", ""))})
            except (ValueError, TypeError):
                pass
    return out


def _spec(role: str, profile: str | None):
    s = role_spec(role, profile)
    return s["base_url"], s["served_model_name"]


def _provider(role: str, profile: str | None) -> str:
    prov = role_spec(role, profile).get("provider")
    if not prov:
        raise KeyError(f"model role '{role}' has no 'provider' in the active profile")
    return prov


def _unimplemented(prov: str, role: str) -> NotImplementedError:
    return NotImplementedError(
        f"model provider '{prov}' for role '{role}' is not wired in this PoC "
        "— use MODEL_PROFILE=local")


class MLLMClient(Protocol):
    async def describe(self, video_path: str, audio_path: str, prompt: str) -> str: ...


class ASRClient(Protocol):
    async def transcribe(self, audio_paths: list[str]) -> list[dict]: ...


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

    async def transcribe(self, audio_paths: list[str],
                         language: str | None = None) -> list[dict]:
        """Batch transcribe+align: returns [{language, text, words:[{text,t_start,
        t_end}]}, ...], one per path. All clips go in ONE request because the
        engine must not be called concurrently."""
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{self.base}/transcribe",
                             json={"audio_paths": audio_paths, "language": language})
            r.raise_for_status()
            return r.json()


class QwenOmni:
    def __init__(self, profile: str | None = None):
        base, self.model = _spec("mllm", profile)
        self.client = AsyncOpenAI(base_url=base, api_key="x")

    async def describe(self, video_path: str, audio_path: str | None, prompt: str) -> str:
        # The av clip carries its own audio; when audio is wanted we set
        # use_audio_in_video so Qwen3-Omni ingests the video's audio track (no
        # separate audio_url — that mismatches the video/audio item count). fps
        # is derived by the processor; passing it here would collide with that.
        content: list[dict] = [
            {"type": "video_url", "video_url": {
                "url": f"file://{video_path}",
                "min_pixels": 128 * 128, "max_pixels": 560 * 560}},
            {"type": "text", "text": prompt},
        ]
        # NB: no chat_template_kwargs enable_thinking — on Qwen3-Omni-Instruct
        # (not a thinking model) it degenerates output to bare newlines,
        # especially on the audio path. modalities text skips the TTS stages.
        extra_body: dict = {"modalities": ["text"]}
        if audio_path:
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=2048, temperature=0.2,
            extra_body=extra_body,
            messages=[{"role": "user", "content": content}])
        return r.choices[0].message.content or ""

    async def chat(self, prompt: str) -> str:
        """Text-only chat (e.g. aggregate shot summaries into a global summary)."""
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=1024, temperature=0.2,
            extra_body={"modalities": ["text"]},
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    async def _scene_call(self, clip_path: str, audio_path: str | None,
                          system: str, user_text: str) -> list[dict]:
        """Omni scene JSON with video (fps/pixels in the content block, per the
        reference Omni usage) + optional audio. The text MUST come AFTER the media
        — putting text before the video makes Omni degenerate into repeated
        newlines (verified empirically). Retries with rising temperature/
        repetition_penalty since structured output still degenerates occasionally."""
        content: list[dict] = [
            {"type": "video_url", "video_url": {
                "url": f"file://{clip_path}",
                "min_pixels": 128 * 128, "max_pixels": 560 * 560}},
            {"type": "text", "text": user_text},
        ]
        # av clip carries audio: deliver it via use_audio_in_video (served by
        # vllm-omni), not a separate audio_url which mismatches the video/audio
        # item count. fps is derived by the processor, so it is omitted here.
        aiv = {"use_audio_in_video": True} if audio_path else None
        for temp, rep in ((0.3, 1.05), (0.6, 1.1), (0.9, 1.3)):
            extra_body: dict = {"modalities": ["text"], "repetition_penalty": rep}
            if aiv:
                extra_body["mm_processor_kwargs"] = aiv
            r = await self.client.chat.completions.create(
                model=self.model, max_tokens=2048, temperature=temp,
                extra_body=extra_body,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": content}])
            scenes = _parse_scenes(r.choices[0].message.content or "")
            if scenes:
                return scenes
        return []

    async def segment_window(self, clip_path: str, start_abs: float, end_abs: float,
                             audio_path: str | None = None) -> list[dict]:
        """Segment a clip covering [start_abs, end_abs] of the video into scenes.
        Times are ABSOLUTE on the video timeline (H:MM:SS.ff)."""
        user = (
            f"This video clip covers {sec_to_hms(start_abs)} to {sec_to_hms(end_abs)}. "
            "Segment it into scenes. Output ONLY a JSON array "
            '[{"start":"H:MM:SS.ff","end":"H:MM:SS.ff","summary":"..."}] with start/end '
            "in that same H:MM:SS.ff time format, contiguous and covering the whole clip.")
        return await self._scene_call(clip_path, audio_path, _SEGMENT_SYS, user)

    async def reconcile_overlap(self, clip_path: str, ov_start_abs: float, ov_end_abs: float,
                                segs_a: list[dict], segs_b: list[dict],
                                audio_path: str | None = None) -> list[dict]:
        """Reconcile two segmentations of the overlap [ov_start_abs, ov_end_abs]
        (absolute times)."""
        def _fmt(segs):
            return [{"start": sec_to_hms(s["t_start"]), "end": sec_to_hms(s["t_end"]),
                     "summary": s["summary"]} for s in segs]
        user = (
            f"This clip covers {sec_to_hms(ov_start_abs)} to {sec_to_hms(ov_end_abs)}.\n"
            f"Analysis A: {json.dumps(_fmt(segs_a))}\nAnalysis B: {json.dumps(_fmt(segs_b))}\n"
            "Output the final contiguous scenes as ONLY a JSON array "
            '[{"start":"H:MM:SS.ff","end":"H:MM:SS.ff","summary":"..."}] covering the clip.')
        return await self._scene_call(clip_path, audio_path, _RECONCILE_SYS, user)


class GeminiOrchestrator:
    """Multimodal orchestrator for the labelling agent (google-genai).

    `judge(system, prompt, frames)` drives the JUDGE stage — sampled shot frames
    are attached as image parts and a JSON response is requested. `invoke(prompt)`
    returns plain text for simple text callers (e.g. Data Viewer QA). An
    audio-capable model is used so raw shot audio can be added later without an
    interface change."""

    def __init__(self, profile_name: str | None = None):
        import os

        from google import genai

        spec = role_spec("agent_llm", profile_name)
        self.model = spec["model"]
        self.temperature = float(spec.get("temperature", 0.0))
        self.max_output_tokens = int(spec.get("max_tokens", 2048))
        self._client = genai.Client(api_key=os.environ["GENAI_API_KEY"])

    def _generate(self, system: str | None, parts: list, json_out: bool) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction=system or None,
            response_mime_type="application/json" if json_out else None,
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=cfg,
        )
        return resp.text or ""

    def judge(self, system: str, prompt: str, frames: list[bytes] | None = None) -> str:
        from google.genai import types

        parts: list = [types.Part.from_text(text=prompt)]
        for fb in frames or []:
            parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
        return self._generate(system, parts, json_out=True)

    def invoke(self, prompt) -> str:
        from google.genai import types

        text = prompt if isinstance(prompt, str) else str(prompt)
        return self._generate(None, [types.Part.from_text(text=text)], json_out=False)


def get_agent_llm(profile_name: str | None = None) -> GeminiOrchestrator:
    """Multimodal orchestrator for the labelling agent (config `agent_llm`)."""
    prov = _provider("agent_llm", profile_name)
    if prov == "google-genai":
        return GeminiOrchestrator(profile_name)
    raise _unimplemented(prov, "agent_llm")


def get_text_embedder(profile_name: str | None = None) -> Embedder:
    prov = _provider("text_embedding", profile_name)
    if prov == "vllm":
        return QwenTextEmbedder(profile_name)
    raise _unimplemented(prov, "text_embedding")


def get_image_embedder(profile_name: str | None = None) -> Embedder:
    prov = _provider("image_embedding", profile_name)
    if prov == "vllm":
        return QwenVisualEmbedder(profile_name)
    raise _unimplemented(prov, "image_embedding")


def get_mllm(profile_name: str | None = None) -> MLLMClient:
    prov = _provider("mllm", profile_name)
    if prov == "vllm":
        return QwenOmni(profile_name)
    raise _unimplemented(prov, "mllm")


def get_asr(profile_name: str | None = None) -> ASRClient:
    prov = _provider("asr", profile_name)
    if prov == "vllm":
        return QwenASRAligner(profile_name)
    raise _unimplemented(prov, "asr")
