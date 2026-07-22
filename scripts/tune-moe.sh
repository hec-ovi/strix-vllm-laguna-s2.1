#!/usr/bin/env bash
# Generate the gfx1151-tuned fused-MoE Triton config for Laguna's expert shape
# (256 experts, top-10, intermediate 1024, W4A16). vLLM ships no config JSON for
# gfx1151, so its MoE kernels fall back to heuristic launch params; this grid
# search finds real ones. Output lands in cache/vllm/moe_configs/ (mounted to
# /root/.cache/vllm/moe_configs), which docker-compose points VLLM_TUNED_CONFIG_FOLDER at.
#
# Batch sizes default to decode-relevant M (single-stream decode is M=1). The
# search is long; M=1 alone is ~10 min on this box, the full 1 2 4 8 set is hours.
#
# Usage: scripts/tune-moe.sh [BATCH_SIZES...]   (default: 1)
set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_SIZES="${*:-1}"

# vLLM's benchmark_moe.py has no branch for the Laguna config class; add one
# inline in a throwaway container so we never mutate the image.
docker compose run --rm -T --entrypoint sh vllm -c '
python - <<EOF
path = "/opt/vllm-src/benchmarks/kernels/benchmark_moe.py"
src = open(path).read()
if "LagunaForCausalLM" not in src:
    src = src.replace(
        "    else:\n        # Support for llama4\n",
        "    elif architecture == \"LagunaForCausalLM\":\n"
        "        E = config.num_experts\n"
        "        topk = config.num_experts_per_tok\n"
        "        intermediate_size = config.moe_intermediate_size\n"
        "        hidden_size = config.hidden_size\n"
        "    else:\n        # Support for llama4\n",
        1,
    )
    open(path, "w").write(src)
    print("patched benchmark_moe.py with Laguna branch")
EOF
uv pip install -q ray
python /opt/vllm-src/benchmarks/kernels/benchmark_moe.py \
    --model /models/Laguna-S-2.1-INT4 --trust-remote-code \
    --dtype int4_w4a16 --tp-size 1 --tune \
    --batch-size '"$BATCH_SIZES"' \
    --save-dir /root/.cache/vllm/moe_configs
'
echo "tuned config(s) written to cache/vllm/moe_configs/"
ls -la cache/vllm/moe_configs/
