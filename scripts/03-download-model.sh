#!/usr/bin/env bash
# Laguna S 2.1 INT4 (~72 GB) plus the DFlash draft model (~2 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

pip install -q "huggingface_hub[cli]"
hf download poolside/Laguna-S-2.1-INT4  --local-dir models/Laguna-S-2.1-INT4
hf download poolside/Laguna-S-2.1-DFlash --local-dir models/Laguna-S-2.1-DFlash
du -sh models/*
