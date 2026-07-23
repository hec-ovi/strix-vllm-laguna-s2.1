"""Correctness + timing for the SWA block-skip decode kernel override.

Runs the stock chunked_prefill_paged_decode (site-packages) and the patched
one (loaded from /tmp/patched_cppd.py, mounted by the caller) on identical
synthetic paged KV, checks both against a plain-torch reference, and times
the kernel at decode-realistic sequence lengths. Kernel-level only: no
server, milliseconds of GPU per call.

Run:
  docker compose run --rm -T \
    -v ./docker/overrides/chunked_prefill_paged_decode.py:/tmp/patched_cppd.py \
    -v ./scripts/test-swa-skip.py:/tmp/test-swa-skip.py \
    --entrypoint python3 vllm /tmp/test-swa-skip.py
"""

import importlib.util
import sys

import torch

from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as stock_fn,
)

spec = importlib.util.spec_from_file_location(
    "vllm.v1.attention.ops.patched_cppd", "/tmp/patched_cppd.py"
)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)
patched_fn = patched_mod.chunked_prefill_paged_decode

torch.manual_seed(0)
dev = "cuda"
DTYPE = torch.bfloat16
NQ, NKV, HD = 32, 8, 128  # GQA 4:1, Laguna-like head_dim
BS = 16  # physical block size
X = 16 // DTYPE.itemsize  # K-cache packing factor
WINDOW = 512
SEQ_LENS = [512, 2048, 8192, 16384]


def build_case(seq_len):
    nb = (seq_len + BS - 1) // BS
    key_cache = torch.randn(nb, NKV, HD // X, BS, X, dtype=DTYPE, device=dev)
    value_cache = torch.randn(nb, NKV, HD, BS, dtype=DTYPE, device=dev)
    block_table = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    q = torch.randn(1, NQ, HD, dtype=DTYPE, device=dev)
    return q, key_cache, value_cache, block_table


def torch_ref(q, key_cache, value_cache, seq_len):
    # gather per-token K/V from the paged layout
    nb = key_cache.shape[0]
    k = (
        key_cache.permute(0, 3, 1, 2, 4)  # [nb, BS, NKV, HD//X, X]
        .reshape(nb * BS, NKV, HD)[:seq_len]
        .float()
    )
    v = (
        value_cache.permute(0, 3, 1, 2)  # [nb, BS, NKV, HD]
        .reshape(nb * BS, NKV, HD)[:seq_len]
        .float()
    )
    scale = HD ** -0.5
    ctx = seq_len - 1
    keep = (ctx - torch.arange(seq_len, device=dev)) < WINDOW  # window mask
    out = torch.empty(1, NQ, HD, device=dev)
    for h in range(NQ):
        kv_h = h // (NQ // NKV)
        s = (q[0, h].float() @ k[:, kv_h].T) * scale
        s = torch.where(keep, s, torch.tensor(float("-inf"), device=dev))
        p = torch.softmax(s, dim=-1)
        out[0, h] = p @ v[:, kv_h]
    return out.to(DTYPE)


def run(fn, q, key_cache, value_cache, block_table, seq_len):
    out = torch.zeros_like(q)
    fn(
        query=q,
        key=torch.zeros(1, NKV, HD, dtype=DTYPE, device=dev),
        value=torch.zeros(1, NKV, HD, dtype=DTYPE, device=dev),
        output=out,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=dev),
        max_seq_len=seq_len,
        max_query_len=1,
        k_scale=torch.tensor(1.0, device=dev),
        v_scale=torch.tensor(1.0, device=dev),
        sliding_window=WINDOW,
        sm_scale=HD ** -0.5,
    )
    return out


def time_fn(fn, args, seq_len, iters=100):
    for _ in range(20):
        run(fn, *args, seq_len)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run(fn, *args, seq_len)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


ok = True
print(f"{'seq_len':>8} {'stock us':>9} {'patched us':>10} {'speedup':>8}"
      f" {'stock err':>10} {'patch err':>10}")
for seq_len in SEQ_LENS:
    args = build_case(seq_len)
    q, kc, vc, bt = args
    ref = torch_ref(q, kc, vc, seq_len)
    out_stock = run(stock_fn, *args, seq_len)
    out_patch = run(patched_fn, *args, seq_len)
    err_s = (out_stock.float() - ref.float()).abs().max().item()
    err_p = (out_patch.float() - ref.float()).abs().max().item()
    if err_p > 0.05 or err_s > 0.05:
        ok = False
    t_s = time_fn(stock_fn, args, seq_len)
    t_p = time_fn(patched_fn, args, seq_len)
    print(f"{seq_len:>8} {t_s:>9.1f} {t_p:>10.1f} {t_s / t_p:>7.2f}x"
          f" {err_s:>10.4f} {err_p:>10.4f}")

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
