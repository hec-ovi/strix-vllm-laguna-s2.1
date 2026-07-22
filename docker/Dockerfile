FROM ubuntu:26.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential pkg-config libdrm-dev libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV VENV=/opt/venv
RUN uv venv --seed --python 3.13 $VENV
ENV PATH="$VENV/bin:$PATH" VIRTUAL_ENV=$VENV

# TheRock nightly ROCm + PyTorch for gfx1151
ARG ROCM_INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/
RUN uv pip install --index-url $ROCM_INDEX "rocm[libraries,devel,device-gfx1151]" && \
    uv pip install --index-url $ROCM_INDEX "torch[device-gfx1151]" "torchvision[device-gfx1151]" torchaudio

# vLLM from source: Laguna support starts at 0.25.0, gfx1151 is not an upstream target
ARG VLLM_REF=v0.25.1
RUN git clone --depth 1 --branch $VLLM_REF https://github.com/vllm-project/vllm.git /opt/vllm-src
WORKDIR /opt/vllm-src
RUN python use_existing_torch.py && \
    uv pip install cmake ninja && \
    uv pip install -r requirements/rocm.txt

ENV VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1151
# hip-lang and friends live in the devel SDK, not the core package
RUN CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" uv pip install --no-build-isolation .

# vLLM detects ROCm via the amdsmi python bindings, which TheRock ships in the
# SDK but does not install; without them platform detection fails at startup.
# Expose only libamd_smi via ldconfig; a global LD_LIBRARY_PATH to the SDK lib
# dir shadows the RPATH'd HIP runtime and breaks GPU enumeration entirely.
RUN uv pip install /opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/share/amd_smi && \
    ln -s /opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamd_smi.so.26 /usr/local/lib/libamd_smi.so && \
    printf '%s\n' /opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib \
                  /opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib \
                  > /etc/ld.so.conf.d/rocm-sdk.conf && \
    ldconfig

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
