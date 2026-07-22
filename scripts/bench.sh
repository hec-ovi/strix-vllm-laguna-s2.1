#!/usr/bin/env bash
# Prefill + decode throughput against the running server on :8000.
# Run once with the base compose, once with the DFlash overlay.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --with openai python bench.py
