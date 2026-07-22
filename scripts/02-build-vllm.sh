#!/usr/bin/env bash
# Build vLLM (>= 0.25.1, first version with Laguna support) from source
# against the TheRock ROCm in ./.venv. gfx1151 is not an upstream-supported
# target, so this is a source build with PYTORCH_ROCM_ARCH pinned.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

VLLM_REF="${VLLM_REF:-v0.25.1}"
[ -d vllm-src ] || git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git fetch --tags
git checkout "$VLLM_REF"

# keep the TheRock torch instead of letting the build replace it
python use_existing_torch.py
pip install -r requirements/rocm.txt

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1151
pip install --no-build-isolation -e . 2>&1 | tail -20

python -c "import vllm; print('vllm', vllm.__version__)"
