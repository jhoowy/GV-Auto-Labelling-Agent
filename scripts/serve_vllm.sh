#!/usr/bin/env bash
# Launch local vLLM servers for the ingestion models (single node, B200 x2).
# Model paths are read from config/profiles/local.yaml (git-excluded), so this
# script carries no paths and is safe to track. FlashAttention is vLLM default.
#
#   bash scripts/serve_vllm.sh          # start servers (blocks until all ready)
#
# These are the Agent-stage models. ASR+align runs separately in the qwen-asr
# venv (scripts/serve_asr.sh) because qwen-asr pins vllm==0.14.
# GPU layout: GPU0 = Omni (30B); GPU1 = text-embed + visual-embed.
# Models sharing a GPU are started SEQUENTIALLY — a same-GPU concurrent start
# makes each engine assume the full free memory and their KV caches collide.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${IJI_VENV:-/tmp/iji-video-labelling-venv}"
VLLM="$VENV/bin/vllm"
PY="$VENV/bin/python"
LOGDIR="${VLLM_LOGDIR:-/tmp/vllm-logs}"
mkdir -p "$LOGDIR"

path_of() { "$PY" -c "import yaml;print(yaml.safe_load(open('config/profiles/local.yaml'))['$1']['model_path'])"; }
TEMB=$(path_of text_embedding); VEMB=$(path_of image_embedding)

wait_ready() {  # port name
  local port=$1 name=$2 i
  for i in $(seq 1 150); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && { echo "  ✓ $name ready (:$port)"; return 0; }
    grep -q "EngineCore failed\|Engine core initialization failed" "$LOGDIR/$name.log" 2>/dev/null \
      && { echo "  ✗ $name FAILED (see $LOGDIR/$name.log)"; return 1; }
    sleep 3
  done
  echo "  ✗ $name timeout"; return 1
}

# GPU0 — Omni 30B (vision+audio summary). Served from the ISOLATED vllm-omni
# venv (audio-in-video needs vllm-omni, not stock pip vLLM), independent →
# start in background. See scripts/serve_omni.sh.
bash scripts/serve_omni.sh &

# GPU1 — three small models, started one at a time.
CUDA_VISIBLE_DEVICES=1 "$VLLM" serve "$TEMB" --served-model-name text-embed --port 8803 \
  --runner pooling --convert embed --pooler-config '{"pooling_type":"MEAN"}' \
  --max-model-len 8192 --trust-remote-code --gpu-memory-utilization 0.25 \
  > "$LOGDIR/text-embed.log" 2>&1 &
wait_ready 8803 text-embed

CUDA_VISIBLE_DEVICES=1 "$VLLM" serve "$VEMB" --served-model-name visual-embed --port 8804 \
  --runner pooling --convert embed --max-model-len 32768 \
  --trust-remote-code --gpu-memory-utilization 0.2 \
  > "$LOGDIR/visual-embed.log" 2>&1 &
wait_ready 8804 visual-embed

wait_ready 8801 omni

cat <<EOF
all servers up (Agent-stage models):
  omni :8801 (GPU0) | text-embed :8803 | visual-embed :8804 (GPU1)
ASR+align runs separately: bash scripts/serve_asr.sh (:8810, qwen-asr venv)
EOF
