"""ASR + ForcedAligner HTTP server (word-level timestamps).

Runs in the ISOLATED qwen-asr venv (vllm 0.14 + qwen-asr's own torch/flash),
kept separate from the main venv (vllm 0.26) because qwen-asr pins vllm==0.14.
The main project calls POST /transcribe over HTTP.

Model weights come from env (set by scripts/serve_asr.sh):
    ASR_MODEL, ALIGNER_MODEL, ALIGNER_DEVICE, ASR_GPU_UTIL, PORT

    POST /transcribe {audio_path, language?} -> {language, text, words:[{text,t_start,t_end}]}
"""
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from qwen_asr import Qwen3ASRModel

app = FastAPI()
_model = None


def _load():
    global _model
    _model = Qwen3ASRModel.LLM(
        model=os.environ["ASR_MODEL"],
        gpu_memory_utilization=float(os.getenv("ASR_GPU_UTIL", "0.3")),
        max_inference_batch_size=8, max_new_tokens=4096,
        forced_aligner=os.environ["ALIGNER_MODEL"],
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16, device_map=os.getenv("ALIGNER_DEVICE", "cuda:0"),
            attn_implementation="sdpa"),
    )


class Req(BaseModel):
    audio_paths: list[str]        # batch: the engine must not be called concurrently,
    language: str | None = None   # so callers send many clips in ONE request instead.


def _one(r) -> dict:
    words = [{"text": u.text, "t_start": u.start_time, "t_end": u.end_time}
             for u in (r.time_stamps or [])]
    return {"language": r.language, "text": r.text or "", "words": words}


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading"}


@app.post("/transcribe")
def transcribe(req: Req):
    langs = [req.language] * len(req.audio_paths) if req.language else None
    try:
        rs = _model.transcribe(audio=req.audio_paths, language=langs, return_time_stamps=True)
    except Exception:  # degrade gracefully rather than 500
        import traceback
        traceback.print_exc()
        return [{"language": None, "text": "", "words": []} for _ in req.audio_paths]
    return [_one(r) for r in rs]


if __name__ == "__main__":
    import uvicorn
    _load()  # load under __main__ so vLLM's spawn works
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8810")))
