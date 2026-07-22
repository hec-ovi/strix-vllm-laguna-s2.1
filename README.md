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
scripts/01-setup-rocm.sh      # TheRock nightly ROCm + gfx1151 PyTorch into .venv
scripts/02-build-vllm.sh      # vLLM v0.25.1 source build (Laguna support starts at 0.25.0)
scripts/03-download-model.sh  # INT4 (~72 GB) + DFlash draft (~2 GB)
scripts/04-serve.sh           # serve on 127.0.0.1:8000
scripts/05-smoke-test.sh      # against the running server
```

## Serving notes

- `--moe-backend triton` is mandatory: Marlin is CUDA-only and DeepGEMM is incompatible with DFlash. Verify the startup log says triton; vLLM has been seen falling back to Marlin despite the flag ([vllm#40357](https://github.com/vllm-project/vllm/issues/40357)).
- `DFLASH=1 scripts/04-serve.sh` enables speculative decoding (up to 15 draft tokens per step).
- `KV_AUTO=1` overrides the checkpoint's FP8 KV cache if RDNA3.5 rejects it.
- Thinking is off by default. Per request: `"chat_template_kwargs": {"enable_thinking": true}`.
- Recommended sampling: temp 0.7, top-p 0.95. Do not combine min_p with DFlash.

## Expected performance

Decode reads ~4.2GB of BF16 attention weights plus ~2.4GB of INT4 expert weights per token. At Strix Halo's 256GB/s that caps decode near 35-40 tok/s theoretical; DFlash is what makes it comfortable in practice.

## Links

- [vLLM recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1)
- [Laguna-S-2.1-INT4](https://huggingface.co/poolside/Laguna-S-2.1-INT4)
- [GGUF fallback](https://huggingface.co/poolside/Laguna-S-2.1-GGUF)
- [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
