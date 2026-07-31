#!/usr/bin/env bash
# Stop all local vLLM servers cleanly.
#
# A vLLM server is an APIServer (holds the port) plus EngineCore worker
# subprocesses (hold the GPU memory, no port). Killing only the port-holder
# leaks the EngineCore workers, which keep the GPU pinned — so kill both.
#
#   bash scripts/stop_vllm.sh
set -uo pipefail

# 1. APIServers by port.
for p in 8801 8802 8803 8804; do fuser -k -9 "${p}/tcp" 2>/dev/null || true; done
sleep 3

# 2. EngineCore workers by GPU process name (they have no port).
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | grep -i "VLLM" | cut -d',' -f1 | tr -d ' ' | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "stopped. GPU compute processes remaining:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "  (none)"
