# Validation of the v2 timing and experiment code

Validated on Qwen3-8B BF16, A100 40GB, TP=1, async_cpu=False, CUDA graph off.
Artifacts: `log/v2-validation/` (runtime outputs excluded from Git).

- Sparse k=14,r=.25: two requests completed, 153 committed output tokens.
- Newly supported sweep point k=6,r=.25: two requests completed, 135 committed tokens with timing enabled.
- AR/full-KV path: two requests completed, 130 committed tokens.
- Long input capacity fixture: 18,022 input tokens, k=14,r=.25, 75 output tokens. This is synthetic validation, not benchmark evidence.
- Every timed run reconciles sum(step committed tokens) with final output lengths; every verify record reconciles accepted draft+1 with its linked step and request.
- New-schema traces reconcile per-request query lengths with batch query length and contain nonnegative GPU intervals.
- Greedy k=6,r=.25, seed330, same GPU: timing on/off produced identical output token IDs for both requests, 137 tokens total.
- Stochastic temperature=.65 runs are not bitwise identical: on/off totals 135/138; another no-timing execution on a second A100 gave136. Historical v1 same-GPU retries also differ. No claim of deterministic stochastic trajectories is made, and this validation does not establish the source of backend nondeterminism.
- Sample-preparer fixture: all3 oversized records excluded, screen4/confirm6 records, disjoint sample IDs, stored source/template/sample hashes.
- Python compilation and git whitespace checks pass.

The generation cap has the original engine's possible final-step overshoot; the sample preflight reserves k+1 extra tokens. Tests above request64 output tokens per request.

CUDA events are polled asynchronously and reused. Final close drains pending events. Short tests verify correctness, not stable measurement overhead or long-context throughput. Repeated warmed-up on/off measurements remain a prerequisite for performance claims. New benchmark data has not yet been downloaded; the full v2 sweep was not launched.

Commands for trace validation:

```bash
python experiments/v2/validate_trace.py log/v2-validation/k6-timing log/v2-validation/ar-timing log/v2-validation/long-timing
python experiments/v2/validate_trace.py log/v2-validation/greedy-timing log/v2-validation/greedy-no-timing --equal-outputs
```
