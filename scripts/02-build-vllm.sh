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
pip install cmake ninja pkgconf
pip install -r requirements/rocm.txt

# clang refuses direct <mwaitxintrin.h> includes (gcc-only); use the umbrella header
sed -i 's|#include <mwaitxintrin.h>|#include <x86intrin.h>|' csrc/spinloop.cpp

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1151
# host has gcc but no g++; use TheRock's bundled LLVM toolchain
VENV_BIN="$(cd ../.venv/bin && pwd)"
export CC="$VENV_BIN/amdclang"
export CXX="$VENV_BIN/amdclang++"
# hipcc_cmake_linker_helper expects plain clang/clang++ names in the venv bin
ln -sf amdclang   "$VENV_BIN/clang"
ln -sf amdclang++ "$VENV_BIN/clang++"
# hip-lang and friends live in the devel SDK, not _rocm_sdk_core
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
pip install --no-build-isolation -e . -v

python -c "import vllm; print('vllm', vllm.__version__)"
