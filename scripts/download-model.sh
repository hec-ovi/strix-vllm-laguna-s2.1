#!/usr/bin/env bash
# Laguna S 2.1 INT4 (~72 GB) plus the DFlash draft model (~2 GB) into ./models,
# which docker-compose bind-mounts at /models. Resumes on rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
# HF_TOKEN in .env raises rate limits; hf_transfer parallelizes shard downloads
[ -f .env ] && set -a && source .env && set +a
export HF_HUB_ENABLE_HF_TRANSFER=1

uvx --from "huggingface_hub[cli,hf_transfer]" hf download poolside/Laguna-S-2.1-INT4  --local-dir models/Laguna-S-2.1-INT4
uvx --from "huggingface_hub[cli,hf_transfer]" hf download poolside/Laguna-S-2.1-DFlash --local-dir models/Laguna-S-2.1-DFlash
du -sh models/*
