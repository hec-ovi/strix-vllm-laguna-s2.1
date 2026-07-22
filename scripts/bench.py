"""Measure prefill and decode throughput of the running server.

Prefill: send a ~8k-token prompt, take prompt_tokens / time-to-first-token.
Decode: tokens per second over the rest of the stream.
Run against whatever is on :8000 (base or DFlash overlay), three rounds each.
"""

import json
import statistics
import sys
import time

from openai import OpenAI

BASE_URL = "http://127.0.0.1:8000/v1"
ROUNDS = 3

client = OpenAI(base_url=BASE_URL, api_key="none")
model = client.models.list().data[0].id

# ~8k tokens of prompt material: repeated prose, then one real question
filler = ("The quick brown fox jumps over the lazy dog. " * 12 + "\n") * 55
prompt = filler + "\nIn one long paragraph, explain what a mixture-of-experts model is."

prefill_rates, decode_rates = [], []

for r in range(ROUNDS):
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = now - t0
            n_tokens += 1
    total = time.perf_counter() - t0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else n_tokens
    prefill = prompt_tokens / ttft if ttft else 0.0
    decode = completion_tokens / (total - ttft) if ttft and total > ttft else 0.0
    prefill_rates.append(prefill)
    decode_rates.append(decode)
    print(f"round {r + 1}: prompt={prompt_tokens} tok, ttft={ttft:.2f}s, "
          f"prefill={prefill:.1f} tok/s, completion={completion_tokens} tok, "
          f"decode={decode:.1f} tok/s", file=sys.stderr)

print(json.dumps({
    "prefill_tok_s": round(statistics.median(prefill_rates), 1),
    "decode_tok_s": round(statistics.median(decode_rates), 1),
    "rounds": ROUNDS,
}))
