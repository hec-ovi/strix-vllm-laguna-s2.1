## Purpose

`kernel_paged_attention_2d` in `chunked_prefill_paged_decode.py` iterates
KV blocks from 0 to `num_blocks` and applies the sliding window as a score
mask after K/V are already loaded:

```python
if SLIDING_WINDOW > 0:
    S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)
```

Decode is bandwidth-bound, so windowed layers pay full-context memory
traffic per decoded token even though everything outside the window is
discarded. For a model like poolside/Laguna-S-2.1 (36 of 48 layers with a
512-token window) at 16K context, roughly 97 percent of the K/V reads in
those layers are wasted. The unified attention kernel already bounds its
tile loop to the window (`compute_tile_loop_bounds`); this brings the
paged decode kernel used by the ROCM_ATTN backend up to par.

## Change

Start the block loop at the window edge instead of 0. The existing score
mask still trims the partial first block, and the masked-out contributions
it removes underflow to exactly 0 in the softmax accumulation, so outputs
are unchanged.

## Measurements

gfx1151 (Strix Halo), bf16, GQA 4:1, head_dim 128, window 512, single
decode token, kernel-level timing over 100 iterations. Outputs checked
against a plain PyTorch paged-attention reference (max abs err 0.002 for
both stock and patched, bf16 rounding):

| seq_len | stock | patched | speedup |
|---------|-------|---------|---------|
| 512     | 117 us | 116 us | 1.0x |
| 2048    | 221 us | 114 us | 1.9x |
| 8192    | 661 us | 112 us | 5.9x |
| 16384   | 1520 us | 114 us | 13.4x |

The patched kernel's latency is independent of context length, which is
the behavior the sliding window is supposed to provide.

## Notes

- No behavior change for `SLIDING_WINDOW == 0` (start_block stays 0).
- ALiBi and sink paths are unaffected: sinks are accumulated before the
  loop, and ALiBi biases only apply to tokens inside the window.
- The 2D kernel only runs for single-token queries (`query_len == 1`
  guard at the top), so `context_len = seq_len - 1` is the query position
  and the window arithmetic matches the existing mask exactly.
