"""Draft acceptance + decode speed for a speculative-decoding run.

Reads vLLM's prometheus spec-decode counters around one greedy coding request
(greedy is the friendliest case for a drafter: acceptance is pure argmax match).
Prints accepted/drafted, per-position acceptance, and observed decode tok/s, so a
DFlash config can be judged in one request instead of a full bench sweep.

Usage: uv run --with openai python spec-accept.py [prompt-file]
"""

import re
import sys
import time
import urllib.request

from openai import OpenAI

BASE = "http://127.0.0.1:8000"
client = OpenAI(base_url=BASE + "/v1", api_key="none", timeout=1800)
model = client.models.list().data[0].id

PROMPT = (
    "Write a Python module that implements an LRU cache with a TTL per entry. "
    "Include the class, type hints, docstrings, and a short usage example."
)


def counters():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("spec_decode_num_accepted_tokens_total",
                 "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_drafts_total"):
        m = re.findall(rf"^vllm:{name}\{{[^}}]*}} ([0-9.e+]+)$", text, re.M)
        out[name] = sum(float(x) for x in m)
    pos = re.findall(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position=\"(\d+)\"[^}]*} ([0-9.e+]+)$",
        text, re.M)
    out["per_pos"] = {int(p): float(v) for p, v in pos}
    return out


before = counters()
t0 = time.perf_counter()
ttft = None
n_out = 0
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.0,
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.usage is not None:
        n_out = chunk.usage.completion_tokens
    if ttft is None and chunk.choices and chunk.choices[0].delta.content:
        ttft = time.perf_counter() - t0
total = time.perf_counter() - t0
after = counters()

acc = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
drafted = after["spec_decode_num_draft_tokens_total"] - before["spec_decode_num_draft_tokens_total"]
drafts = after["spec_decode_num_drafts_total"] - before["spec_decode_num_drafts_total"]
decode_ts = n_out / (total - ttft) if ttft and total > ttft else 0.0

print(f"output tokens : {n_out}")
print(f"decode        : {decode_ts:.1f} tok/s (ttft {ttft:.2f}s, total {total:.1f}s)")
if drafted:
    print(f"drafted       : {drafted:.0f} over {drafts:.0f} drafts")
    print(f"accepted      : {acc:.0f} ({100 * acc / drafted:.1f}%)")
    print(f"mean accept   : {1 + acc / drafts:.2f} tokens/step")
    per_pos = {p: after["per_pos"].get(p, 0) - before["per_pos"].get(p, 0)
               for p in after["per_pos"]}
    rates = [f"{per_pos[p] / drafts:.2f}" for p in sorted(per_pos)] if drafts else []
    print("per-position  : " + ", ".join(rates))
else:
    print("drafted       : 0 (speculative decoding not active)")
