#!/usr/bin/env bash
# End-to-end check against a running server: model listed, completion returns
# content, thinking toggle accepted. Run while 04-serve.sh is up.
set -euo pipefail
BASE="http://127.0.0.1:${PORT:-8000}/v1"

MODEL_ID=$(curl -sf "$BASE/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
echo "model: $MODEL_ID"

curl -sf "$BASE/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_ID"'",
  "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 - <<'EOF'
import sys, json
c = json.load(sys.stdin)["choices"][0]["message"]["content"]
assert c and c.strip(), "empty completion"
print("completion ok:", c.strip()[:80])
EOF

echo "smoke test passed"
