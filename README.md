# strix-vllm-laguna-s2.1

Serve poolside's [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) (118B MoE, 8B active per token) on an AMD Strix Halo box (Ryzen AI Max+ 395, gfx1151, 128GB unified memory) with vLLM built against TheRock nightly ROCm. OpenAI-compatible endpoint, 256K context, optional DFlash speculative decoding. Follow-up to [vllm-qwen](https://github.com/hec-ovi/vllm-qwen), same stack, bigger model.

Status: running. Laguna S 2.1 serves end to end on gfx1151 through vLLM 0.25.2 built from source against TheRock ROCm 7.15 nightly. The open question going in, whether vLLM's Triton W4A16 MoE path loads the pack-quantized INT4 checkpoint on this GPU (the usual Marlin kernels are CUDA-only), answered yes: it selects `CompressedTensorsWNA16MoEMethod` on the Triton backend. Measured numbers below.

## Why INT4

The only variant that fits in 128GB unified memory:

| Variant | Size | Fits |
|---------|------|------|
| BF16    | 235 GB | no |
| FP8     | 121 GB | no (no headroom for KV) |
| INT4    | 72 GB  | yes |
| NVFP4   | 72 GB  | no (Blackwell only) |

INT4 quantizes only the MoE expert weights to 4 bit (compressed-tensors pack-quantized); attention stays BF16, KV cache ships FP8. KV stays small at long context because only 12 of 48 layers use global attention (the rest are sliding-window, capped at 512 tokens): the full 262K context allocates 38 GiB of KV, room for 1.63M tokens.

## Run

```bash
scripts/download-model.sh   # INT4 (~72 GB) + DFlash draft (~2 GB) into ./models
docker compose build        # Ubuntu 26.04 + TheRock ROCm + vLLM v0.25.1 source build
docker compose up -d        # serve on 127.0.0.1:8000
scripts/smoke-test.sh       # against the running server
```

Overlays (compose merges them onto the base): `-f docker-compose.tuned.yml` loads the gfx1151-tuned decode config, `-f docker-compose.dflash.yml` adds DFlash speculative decoding, `-f docker-compose.rocmpatch.yml` applies the sliding-window block-skip fix to the ROCM_ATTN decode kernel (see the SWA section).

The container gets the iGPU via `/dev/kfd` + `/dev/dri` passthrough; no ROCm install needed on the host. The Dockerfile handles every gfx1151 build quirk (see the build-notes section).

## Serving notes

- `--moe-backend triton` is mandatory: Marlin is CUDA-only and DeepGEMM is incompatible with DFlash. Verify the startup log says triton; vLLM has been seen falling back to Marlin despite the flag ([vllm#40357](https://github.com/vllm-project/vllm/issues/40357)).
- DFlash speculative decoding: `docker compose -f docker-compose.yml -f docker-compose.dflash.yml up -d`. Currently a net loss on this stack, see the DFlash status section before using it.
- If RDNA3.5 rejects the checkpoint's FP8 KV cache, add `--kv-cache-dtype auto` to the compose command.
- Thinking is off by default. Per request: `"chat_template_kwargs": {"enable_thinking": true}`.
- Recommended sampling: temp 0.7, top-p 0.95. Do not combine min_p with DFlash.

## Measured results

First published numbers for this model on this chip through vLLM (single stream, ROCM_ATTN attention backend, thinking off, per-context warmup, prefix cache defeated with a fresh nonce; `scripts/bench.sh`). The Vulkan column is llama.cpp serving the same model on the same box. Note on the backend: vLLM 0.25.2 silently ignores the old `VLLM_ATTENTION_BACKEND` env var, so both recorded runs used the auto-selected ROCM_ATTN and differ only by noise; the working knob is `--attention-backend`, wired in the compose command as `ATTN_BACKEND=`.

| Context | vLLM prefill t/s | vLLM decode t/s (default) | vLLM decode t/s (tuned) | Vulkan prefill t/s | Vulkan decode t/s |
|---------|------------------|---------------------------|-------------------------|--------------------|-------------------|
| 512 | 380 | n/a | 15.2 | n/a | n/a |
| 2k  | 752 | 10.9 | 12.8 | 293 | 22.7 |
| 8k  | 476 | 7.2  | 8.0  | 311 | 22.0 |
| 16k | 301 | 5.0  | n/a  | 275 | 21.1 |

Where it stands, honestly:

- **Prefill wins**, 2.5x over Vulkan at 2k and 1.5x at 8k. This is the strong result.
- **Decode loses.** vLLM ships no tuned Triton fused-MoE configs for gfx1151, so the kernels run on heuristic fallbacks. Grid-searching this model's expert shape (256 experts, top-10, intermediate 1024) for single-stream decode (M=1) lifts decode ~17% (10.9 to 12.8 at 2k), still short of Vulkan's 22.7. Bandwidth ceiling for 8B active at INT4 is ~35-40 t/s, so the kernels are leaving most of it on the floor.
- **The tuned config is a trade, not a free win.** It currently has only an M=1 entry, so its decode tiling bleeds into prefill's large-M GEMMs and drops prefill (752 to 472 at 2k). That is why it is an opt-in overlay, not the default. Tuning the full M range (`scripts/tune-moe.sh 1 2 4 8 16 ... 2048`) is the path to a config that wins both; the committed `moe-configs/` file is the M=1 start.

### SWA decode kernel fix (kernel-level, measured)

The long-context decode decay traced to a concrete cause: the ROCM_ATTN decode kernel reads K/V for the entire context and applies the 512-token sliding window as a score mask after the loads, so the 36 windowed layers pay full-context bandwidth per decoded token. `docker-compose.rocmpatch.yml` mounts a one-hunk override that starts the kernel's block loop at the window edge instead. Kernel-level numbers on gfx1151 (`scripts/test-swa-skip.py`: bf16, GQA 4:1, window 512, output checked against a plain-torch reference, max abs err 0.002):

| seq len | stock us | patched us | speedup |
|---------|----------|------------|---------|
| 512   | 117  | 116 | 1.0x |
| 2k    | 221  | 114 | 1.9x |
| 8k    | 661  | 112 | 5.9x |
| 16k   | 1520 | 114 | 13.4x |

The patched kernel is flat across context, which is what a 512-token window is supposed to buy. End-to-end long-context decode numbers are pending (single-probe measurement). The no-patch alternative is `ATTN_BACKEND=TRITON_ATTN`, whose upstream kernel already bounds the tile loop to the window; note that neither backend has real end-to-end numbers here yet, since the old env var silently never switched backends.

Next levers, in order: end-to-end validation of the SWA fix, full-M-range MoE tuning, then a wave32-specialized fused gather+dequant+GEMV for the experts.

## DFlash status: measured, not recommended

The DFlash drafter loads and runs behind the overlay, but as measured here it is a net loss: 0% draft acceptance on this stack (every drafted token rejected), which drags decode from 10.9 to about 2-3 t/s at 2k because the server does draft work and throws all of it away. The ceiling is low even where DFlash is fully supported: DGX Spark users report mostly 2-3% acceptance for this same drafter on an NVFP4 stack. The overlay needs `--max-num-seqs 8` to start at all (parallel drafting reserves draft-token slots per sequence; the default sends the scheduler budget negative).

The exact 0% here is unresolved. The two custom GPU paths that feed the drafter both pass bit-accuracy tests on gfx1151 (`scripts/test-dflash-kernels.py`, run it inside the container), so what remains suspect is the compiled-model paths or the drafter's attention. Knobs for anyone digging further: `SPEC_TOKENS` (default 7, per the model card), `DRAFT_ATTN` to pin the drafter's attention backend (the drafter never inherits the target's `--attention-backend` flag), `scripts/spec-accept.py` to read acceptance from one greedy request. Until acceptance is real on this stack, run without the overlay.

## Build notes for gfx1151 (the part that cost a day)

Every fix is a commit in this repo. The ones any vLLM-from-source build on Strix Halo will hit: TheRock ships its LLVM toolchain but the host needs none (the image installs `build-essential pkg-config libdrm-dev libnuma-dev`); the HIP CMake packages live in the `devel` extra, not `libraries`; clang rejects vLLM's direct `<mwaitxintrin.h>` include that gcc tolerates; TheRock's torchvision must match its torch or `torchvision::nms` fails at import; vLLM's ROCm platform detection hard-requires the `amdsmi` python bindings, which TheRock bundles at `_rocm_sdk_core/share/amd_smi` but does not install, and loading them needs the SDK's `rocm_sysdeps` on the loader path (ldconfig entries, not `LD_LIBRARY_PATH`, which shadows the RPATH'd HIP runtime and breaks GPU enumeration).

## Links

- [vLLM recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1)
- [Laguna-S-2.1-INT4](https://huggingface.co/poolside/Laguna-S-2.1-INT4)
- [GGUF fallback](https://huggingface.co/poolside/Laguna-S-2.1-GGUF)
- [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
