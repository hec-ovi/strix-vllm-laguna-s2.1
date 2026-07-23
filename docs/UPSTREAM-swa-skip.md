# Upstream PR: SWA block skip in chunked_prefill_paged_decode

Branch `swa-block-skip-paged-decode` is pushed to the `hec-ovi/vllm` fork,
one commit on top of upstream main. To open the PR:

```bash
cd ~/workspace/vllm-fork
gh pr create --repo vllm-project/vllm \
  --title "[Kernel] Skip out-of-window KV blocks in paged decode sliding-window path" \
  --body-file ../strix-vllm-laguna-s2.1/docs/UPSTREAM-swa-skip-body.md
```

The measured numbers in the commit and body come from
`scripts/test-swa-skip.py` in this repo.
