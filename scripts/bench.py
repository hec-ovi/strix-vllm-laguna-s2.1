"""Prefill/decode throughput table across context sizes, one row per context.

Matches the llama-vulkan-strix table format for direct comparison:
prefill tok/s = prompt_tokens / time-to-first-token (prefix cache defeated
by a unique nonce at position 0), decode tok/s = streamed tokens per second
after the first token. One warmup request, then one measured round per size.
"""

import json
import sys
import time

from openai import OpenAI

BASE_URL = "http://127.0.0.1:8000/v1"
# BENCH_CONTEXTS=2048 (or comma list) for fast iteration loops
import os
CONTEXTS = [int(c) for c in os.environ.get("BENCH_CONTEXTS", "2048,8192,16384,32768").split(",")]

client = OpenAI(base_url=BASE_URL, api_key="none", timeout=1800)
model = client.models.list().data[0].id

SENTENCE = "The quick brown fox jumps over the lazy dog and the slow red crab. "  # ~16 tok


def build_prompt(nonce: str, target_tokens: int) -> str:
    reps = max(1, (target_tokens - 60) // 16)
    return (f"benchmark {nonce}. " + SENTENCE * reps
            + "\nIn one long paragraph, explain what a mixture-of-experts model is.")


def measure(nonce: str, target_tokens: int):
    t0 = time.perf_counter()
    ttft = None
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(nonce, target_tokens)}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            usage = chunk.usage
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = now - t0
    total = time.perf_counter() - t0
    p = usage.prompt_tokens
    c = usage.completion_tokens
    return p, c, p / ttft, (c / (total - ttft) if total > ttft and c > 1 else 0.0)


measure("warmup-x", 512)  # absorb first-request graph/kernel compiles

rows = []
for ctx in CONTEXTS:
    p, c, prefill, decode = measure(f"ctx{ctx}-n{ctx * 31}", ctx)
    rows.append({"context": ctx, "prompt_tokens": p,
                 "prefill_tok_s": round(prefill, 1), "decode_tok_s": round(decode, 1)})
    print(f"{ctx:>6}: prompt={p} prefill={prefill:.1f} t/s decode={decode:.1f} t/s "
          f"(completion={c} tok)", file=sys.stderr)

print(json.dumps(rows))
