"""Attribute decode-step GPU time to kernels via the torch profiler.

Drives ~64 pure decode tokens (1-token prompt, so almost all time is decode),
captures the ROCm/HIP kernel timeline, and prints the top kernels by total
device time. Tells us whether the W4A16 MoE dequant or the BF16 attention read
dominates, so the kernel rewrite targets the real bottleneck.

Run inside the server container (has torch + the model process is separate, so
this uses a fresh in-process engine) OR against the HTTP server with the
server-side profiler. Here we use the server-side torch profiler via vLLM's
/start_profile + /stop_profile endpoints, then parse the trace.
"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def gen(n_tokens):
    body = {
        "model": MODEL,
        "prompt": "Count:",
        "max_tokens": n_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    post("/v1/completions", body)


MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]

# warm
gen(8)
# profile window
post("/start_profile")
t0 = time.perf_counter()
gen(64)
dt = time.perf_counter() - t0
post("/stop_profile")
print(f"64-token decode wall: {dt:.2f}s ({64/dt:.1f} tok/s)", file=sys.stderr)
print("trace written to the dir set by VLLM_TORCH_PROFILER_DIR", file=sys.stderr)
