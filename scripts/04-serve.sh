#!/usr/bin/env bash
# Serve Laguna S 2.1 INT4 with the poolside vLLM recipe.
# Env toggles:
#   DFLASH=1   enable speculative decoding (needs models/Laguna-S-2.1-DFlash)
#   KV_AUTO=1  override the checkpoint's FP8 KV cache if it fails on gfx1151
#   MAX_LEN    context length (default 262144)
#   PORT       default 8000
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL_DIR="${MODEL_DIR:-models/Laguna-S-2.1-INT4}"
EXTRA=()
if [ "${DFLASH:-0}" = "1" ]; then
  EXTRA+=(--speculative-config '{"model":"models/Laguna-S-2.1-DFlash","num_speculative_tokens":15,"method":"dflash"}')
fi
if [ "${KV_AUTO:-0}" = "1" ]; then
  EXTRA+=(--kv-cache-dtype auto)
fi

# --moe-backend triton is required twice over: DFlash is incompatible with the
# DeepGEMM backend, and Marlin W4A16 kernels are CUDA-only. Check the startup
# log actually says triton; vllm has been seen falling back to marlin silently.
exec vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --trust-remote-code \
  --max-model-len "${MAX_LEN:-262144}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --moe-backend triton \
  "${EXTRA[@]}"
