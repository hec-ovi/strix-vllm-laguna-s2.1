#!/usr/bin/env bash
# TheRock nightly ROCm + PyTorch for gfx1151, into ./.venv
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/
pip install --index-url "$INDEX" "rocm[libraries,device-gfx1151]"
pip install --index-url "$INDEX" "torch[device-gfx1151]" "torchvision[device-gfx1151]" torchaudio

python - <<'EOF'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip, "| gpu:", torch.cuda.is_available() and torch.cuda.get_device_name(0))
EOF
