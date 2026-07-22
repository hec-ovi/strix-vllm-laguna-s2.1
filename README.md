# strix-vllm-laguna-s2.1

Serve poolside's [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) (118B MoE, 8B active per token) on an AMD Strix Halo box (Ryzen AI Max+ 395, gfx1151, 128GB unified memory) with vLLM built against TheRock nightly ROCm. OpenAI-compatible endpoint, 256K context, optional DFlash speculative decoding. Follow-up to [vllm-qwen](https://github.com/hec-ovi/vllm-qwen), same stack, bigger model.

Status: scripts encode the full build path but have not been executed end to end on the target machine yet. The one open question is whether vLLM's Triton W4A16 MoE path loads the pack-quantized INT4 checkpoint on gfx1151 (the usual Marlin kernels are CUDA-only). The first run of `04-serve.sh` answers it. If it fails, the fallback is poolside's llama.cpp fork (branch `laguna`) with the Q4_K_M GGUF.

## Why INT4

The only variant that fits in 128GB unified memory:

| Variant | Size | Fits |
|---------|------|------|
| BF16    | 235 GB | no |
| FP8     | 121 GB | no (no headroom for KV) |
| INT4    | 72 GB  | yes |
| NVFP4   | 72 GB  | no (Blackwell only) |

INT4 quantizes only the MoE expert weights to 4 bit (compressed-tensors pack-quantized); attention stays BF16, KV cache ships FP8. KV stays small at long context because only 12 of 48 layers use global attention (the rest are sliding-window, capped at 512 tokens): about 13GB at 256K.

## Run

```bash
scripts/download-model.sh   # INT4 (~72 GB) + DFlash draft (~2 GB) into ./models
docker compose build        # Ubuntu 26.04 + TheRock ROCm + vLLM v0.25.1 source build
docker compose up -d        # serve on 127.0.0.1:8000
scripts/smoke-test.sh       # against the running server
```

The container gets the iGPU via `/dev/kfd` + `/dev/dri` passthrough; no ROCm install needed on the host. Build notes learned the hard way (kept here because any gfx1151 source build hits them): the HIP CMake packages live in TheRock's `devel` extra, `pkg-config` must exist, and clang rejects vLLM's direct `<mwaitxintrin.h>` include that gcc tolerates. The Dockerfile handles all three.

## Serving notes

- `--moe-backend triton` is mandatory: Marlin is CUDA-only and DeepGEMM is incompatible with DFlash. Verify the startup log says triton; vLLM has been seen falling back to Marlin despite the flag ([vllm#40357](https://github.com/vllm-project/vllm/issues/40357)).
- DFlash speculative decoding: `docker compose -f docker-compose.yml -f docker-compose.dflash.yml up -d`.
- If RDNA3.5 rejects the checkpoint's FP8 KV cache, add `--kv-cache-dtype auto` to the compose command.
- Thinking is off by default. Per request: `"chat_template_kwargs": {"enable_thinking": true}`.
- Recommended sampling: temp 0.7, top-p 0.95. Do not combine min_p with DFlash.

## Measured results

First published numbers for this model on this chip through vLLM (single stream, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, thinking off, prefix cache defeated per round; `scripts/bench.sh`). Baseline column is llama.cpp Vulkan serving the same model on the same machine.

| Context | vLLM prefill t/s | vLLM decode t/s | Vulkan prefill t/s | Vulkan decode t/s |
|---------|------------------|-----------------|--------------------|-------------------|
| 2k  | 752 | 10.9 | 293 | 22.7 |
| 8k  | 476 | 7.2  | 311 | 22.0 |
| 16k | 301 | 5.0  | 275 | 21.1 |

Prefill wins by 2.5x at 2k and 1.5x at 8k. Decode currently loses: vLLM's Triton fused-MoE kernels ship no tuned configs for gfx1151, so they run on heuristic fallbacks. A `benchmark_moe.py` grid search for this model's expert shape (256 experts, top-10, intermediate 1024) is the open work; theoretical decode ceiling from weight-read bandwidth is 35-40 t/s.

Other numbers: model load 67.5 GiB in 161s, KV cache 38.4 GiB = 1.63M tokens capacity (only 12 of 48 layers use global attention; the rest are 512-token sliding window with FP8 KV).

## Build notes for gfx1151 (the part that cost a day)

Every fix is a commit in this repo. The ones any vLLM-from-source build on Strix Halo will hit: TheRock ships its LLVM toolchain but the host needs none (the image installs `build-essential pkg-config libdrm-dev libnuma-dev`); the HIP CMake packages live in the `devel` extra, not `libraries`; clang rejects vLLM's direct `<mwaitxintrin.h>` include that gcc tolerates; TheRock's torchvision must match its torch or `torchvision::nms` fails at import; vLLM's ROCm platform detection hard-requires the `amdsmi` python bindings, which TheRock bundles at `_rocm_sdk_core/share/amd_smi` but does not install, and loading them needs the SDK's `rocm_sysdeps` on the loader path (ldconfig entries, not `LD_LIBRARY_PATH`, which shadows the RPATH'd HIP runtime and breaks GPU enumeration).

## Links

- [vLLM recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1)
- [Laguna-S-2.1-INT4](https://huggingface.co/poolside/Laguna-S-2.1-INT4)
- [GGUF fallback](https://huggingface.co/poolside/Laguna-S-2.1-GGUF)
- [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
