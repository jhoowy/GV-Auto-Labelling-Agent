#!/usr/bin/env bash
# Start the ASR+aligner server in the ISOLATED qwen-asr venv (vllm 0.14),
# separate from the main vllm-0.26 servers. Word-level timestamps.
#
#   bash scripts/serve_asr.sh          # background; curl localhost:8810/health
#
# Runs on GPU1 alongside the small main-venv servers; Omni sits on GPU0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

QV="${IJI_QWEN_ASR_VENV:-/tmp/iji-qwen-asr-venv}"
CFG_PY="/tmp/iji-video-labelling-venv/bin/python"   # main venv has pyyaml
path_of() { "$CFG_PY" -c "import yaml;print(yaml.safe_load(open('config/profiles/local.yaml'))['$1']['model_path'])"; }

export ASR_MODEL="$(path_of asr)"
export ALIGNER_MODEL="$(path_of aligner)"
export ALIGNER_DEVICE="cuda:0" ASR_GPU_UTIL="0.3" PORT="8810"
export CUDA_VISIBLE_DEVICES="1" HF_HUB_OFFLINE="1" TRANSFORMERS_OFFLINE="1"
export LD_LIBRARY_PATH="$(find "$QV" -path '*nvidia/*/lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

mkdir -p /tmp/vllm-logs
echo "starting asr-align server (:8810, GPU1) — log /tmp/vllm-logs/asr-align.log"
exec "$QV/bin/python" scripts/asr_align_server.py > /tmp/vllm-logs/asr-align.log 2>&1
