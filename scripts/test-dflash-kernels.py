"""Unit-test the two GPU code paths shared by every broken DFlash config on
gfx1151: the fused input-expansion Triton kernel and the batched RMSNorm over
an expanded (stride-0) tensor used by the drafter's context-KV precompute."""
import torch

from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

torch.manual_seed(0)
dev = "cuda"
ok = True


def check(name, got, want):
    global ok
    w = torch.tensor(want, dtype=got.dtype, device=got.device)
    if torch.equal(got, w):
        print(f"pass {name}")
    else:
        ok = False
        print(f"FAIL {name}\n  got  {got.tolist()}\n  want {w.tolist()}")


# --- Test 1: copy_and_expand_dflash_inputs_kernel ---
num_reqs = 2
ctx_lens = [5, 9]
starts = [0, 5]
num_spec = 15
nq = 1 + num_spec
block_size = 16
total_ctx = sum(ctx_lens)
qsl = torch.tensor([0, 5, 14], dtype=torch.int32, device=dev)
next_tokens = torch.tensor([111, 222], dtype=torch.int32, device=dev)
tpos = torch.tensor(
    list(range(100, 105)) + list(range(200, 209)), dtype=torch.int64, device=dev
)
max_blocks = 40
bt = (
    torch.arange(num_reqs * max_blocks, dtype=torch.int32, device=dev).reshape(
        num_reqs, max_blocks
    )
    + 7
)

out_input_ids = torch.full((num_reqs * nq,), -1, dtype=torch.int32, device=dev)
out_ctx_pos = torch.full((total_ctx,), -1, dtype=torch.int64, device=dev)
out_q_pos = torch.full((num_reqs * nq,), -1, dtype=torch.int64, device=dev)
out_ctx_slot = torch.full((total_ctx,), -1, dtype=torch.int64, device=dev)
out_q_slot = torch.full((num_reqs * nq,), -1, dtype=torch.int64, device=dev)
out_tidx = torch.full((num_reqs * num_spec,), -1, dtype=torch.int32, device=dev)

max_tokens_per_req = max(ctx_lens) + nq
BLOCK = min(256, next_power_of_2(max_tokens_per_req))
nb = (max_tokens_per_req + BLOCK - 1) // BLOCK
copy_and_expand_dflash_inputs_kernel[(num_reqs, nb)](
    next_token_ids_ptr=next_tokens,
    target_positions_ptr=tpos,
    out_input_ids_ptr=out_input_ids,
    out_context_positions_ptr=out_ctx_pos,
    out_query_positions_ptr=out_q_pos,
    out_context_slot_mapping_ptr=out_ctx_slot,
    out_query_slot_mapping_ptr=out_q_slot,
    out_token_indices_ptr=out_tidx,
    block_table_ptr=bt,
    block_table_stride=bt.stride(0),
    query_start_loc_ptr=qsl,
    num_rejected_tokens_ptr=0,
    parallel_drafting_token_id=12,
    block_size=block_size,
    num_query_per_req=nq,
    num_speculative_tokens=num_spec,
    total_input_tokens=total_ctx,
    BLOCK_SIZE=BLOCK,
    HAS_NUM_REJECTED=False,
)
torch.cuda.synchronize()

ref_ids, ref_qpos, ref_qslot = [], [], []
ref_tidx = [-1] * (num_reqs * num_spec)
for r in range(num_reqs):
    last = tpos[starts[r] + ctx_lens[r] - 1].item()
    for q in range(nq):
        ref_ids.append(next_tokens[r].item() if q == 0 else 12)
        p = last + 1 + q
        ref_qpos.append(p)
        bid = bt[r, p // block_size].item()
        ref_qslot.append(bid * block_size + p % block_size)
        if q > 0:
            ref_tidx[r * num_spec + q - 1] = r * nq + q
ref_ctx_slot = []
for r in range(num_reqs):
    for j in range(ctx_lens[r]):
        p = tpos[starts[r] + j].item()
        bid = bt[r, p // block_size].item()
        ref_ctx_slot.append(bid * block_size + p % block_size)

check("input_ids", out_input_ids, ref_ids)
check("query_positions", out_q_pos, ref_qpos)
check("query_slots", out_q_slot, ref_qslot)
check("token_indices", out_tidx, ref_tidx)
check("ctx_positions", out_ctx_pos, tpos.tolist())
check("ctx_slots", out_ctx_slot, ref_ctx_slot)

# --- Test 2: batched rms_norm on stride-0 expanded input ---
from vllm import _custom_ops as ops

L, N, H = 6, 33, 3072
x = torch.randn(N, H, dtype=torch.bfloat16, device=dev)
w = torch.randn(L, H, dtype=torch.bfloat16, device=dev).abs()
out = torch.empty(L, N, H, dtype=torch.bfloat16, device=dev)
ops.rms_norm(out, x.unsqueeze(0).expand(L, -1, -1), w, 1e-6)
torch.cuda.synchronize()

xf = x.float()
rt = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
worst = 0.0
for i in range(L):
    ref = (rt * w[i].float()).to(torch.bfloat16)
    d = (out[i] - ref).abs().max().item()
    worst = max(worst, d)
    print(f"rms_norm layer {i} vs torch ref max abs diff: {d:.6f}")
if worst > 0.1:
    ok = False
    print("FAIL rms_norm batched/expanded path diverges from reference")
else:
    print("pass rms_norm batched/expanded")

print("KERNEL TESTS:", "PASS" if ok else "FAIL")
