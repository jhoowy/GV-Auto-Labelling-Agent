#!/usr/bin/env bash
# Serve Qwen3-Omni (audio+vision understanding) from the ISOLATED vllm-omni
# venv on GPU0:8801. Stock pip vLLM cannot ingest audio-in-video; vllm-omni
# adds Qwen3-Omni's `use_audio_in_video` path, so audio is delivered inside the
# av clip (one video_url + mm_processor_kwargs.use_audio_in_video) — see
# packages/models/models/providers.py.
#
#   bash scripts/serve_omni.sh          # start (blocks until ready)
#
# Isolated from the main venv: the vllm-omni build (vllm 0.26 + vllm-omni) lives
# only here so it never perturbs the other ingestion servers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OMNI_VENV="${OMNI_VENV:-/tmp/iji-vllm-omni-venv}"
VLLM="$OMNI_VENV/bin/vllm"
PY="$OMNI_VENV/bin/python"
LOGDIR="${VLLM_LOGDIR:-/tmp/vllm-logs}"
PORT="${OMNI_PORT:-8801}"
mkdir -p "$LOGDIR"

# tail -1: importing vllm_omni below prints banner lines to stdout; take the
# real value from the last line only.
MODEL=$("$PY" -c "import yaml;print(yaml.safe_load(open('config/profiles/local.yaml'))['mllm']['model_path'])" | tail -1)

# vllm-omni runs Qwen3-Omni as a 3-stage pipeline (thinker/talker/code2wav).
# We only need understanding (thinker, text out) but the bundled deploy pins
# stages 1+2 to cuda:1; re-pin every stage to GPU0 and cap per-stage memory so
# all three co-locate on one B200. base_config inherits connectors/edges.
BASE_DEPLOY=$("$PY" -c "import vllm_omni,os;print(os.path.join(os.path.dirname(vllm_omni.__file__),'deploy','qwen3_omni_moe.yaml'))" | tail -1)
DEPLOY="$LOGDIR/omni_gpu0_deploy.yaml"
cat > "$DEPLOY" <<YAML
base_config: $BASE_DEPLOY
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.5
    max_model_len: 49152
    max_num_batched_tokens: 49152
  - stage_id: 1
    devices: "0"
    gpu_memory_utilization: 0.18
  - stage_id: 2
    devices: "0"
    gpu_memory_utilization: 0.1
YAML

echo "starting omni (:$PORT, GPU0) from $OMNI_VENV — log $LOGDIR/omni.log"
# NOTE: do NOT set a server-level --mm-processor-kwargs fps here. Qwen3-Omni's
# processor derives the video fps internally; a server fps collides with it
# ("fps passed two times") and 400s every use_audio_in_video request.
CUDA_VISIBLE_DEVICES=0 "$VLLM" serve "$MODEL" \
  --omni --served-model-name omni --port "$PORT" \
  --deploy-config "$DEPLOY" \
  --limit-mm-per-prompt '{"video":1,"audio":1}' \
  --allowed-local-media-path /tmp \
  --trust-remote-code \
  > "$LOGDIR/omni.log" 2>&1 &

for i in $(seq 1 300); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "  ✓ omni ready (:$PORT)"; exit 0; }
  grep -qiE "EngineCore failed|initialization failed|CUDA out of memory" "$LOGDIR/omni.log" 2>/dev/null \
    && { echo "  ✗ omni FAILED (see $LOGDIR/omni.log)"; exit 1; }
  sleep 3
done
echo "  ✗ omni timeout"; exit 1
