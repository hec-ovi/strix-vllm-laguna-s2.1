"""One 16k-context decode probe: prefill t/s from TTFT, decode t/s over a
64-token window. Single request, no sweep."""
import sys
import time

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none", timeout=1800)
model = client.models.list().data[0].id

S = "The quick brown fox jumps over the lazy dog and the slow red crab. "
nonce = sys.argv[1]
prompt = (f"benchmark {nonce}. " + S * ((16384 - 60) // 16)
          + "\nIn one long paragraph, explain what a mixture-of-experts model is.")

t0 = time.perf_counter()
ttft = None
n = 0
usage = None
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=64,
    temperature=0.0,
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
        n += 1
total = time.perf_counter() - t0
p = usage.prompt_tokens
c = usage.completion_tokens
print(f"prompt={p} completion={c}")
print(f"prefill: {p / ttft:.1f} t/s (ttft {ttft:.1f}s)")
print(f"decode : {c / (total - ttft):.1f} t/s over {c} tokens")
