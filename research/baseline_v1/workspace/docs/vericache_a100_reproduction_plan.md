# VeriCache-style reproduction plan for one A100

Target date: **2026-10-31**

## Outcome

Build a minimum, correctness-first reproduction of VeriCache's long-context
decoding path on one NVIDIA A100:

- the target model drafts with a compressed GPU-resident KV view;
- the same target model verifies drafted tokens with full KV;
- full KV can live in host memory between verification rounds;
- greedy output matches an unmodified full-KV baseline token for token;
- timings distinguish draft, full-KV reload, verify, and acceptance.

This is a reproduction of the core mechanism, not a claim that the paper's
throughput result has been reproduced. The paper evaluates newer GPUs and a
custom runtime with roughly 8K lines of scheduler and KV-management changes.

## Frozen first implementation

| Component | First choice | Reason |
| --- | --- | --- |
| GPU | One A100, SM80 | Available development target |
| Control | Published vLLM 0.28.0 + LMCache 0.5.5 | Already validated without editable source trees |
| Development base | Official vLLM v0.28.0 source tag | Keeps every VeriCache change attributable and reviewable |
| LMCache source | Official LMCache v0.5.5 tag | Matches the validated control version |
| Draft KV | Minimal sink-plus-recent block selection implemented locally | Runs on A100 without H100-only attention kernels |
| Draft length | Fixed 4, then 8/16 | Reduce debugging surface before long horizons |
| Sampling | Greedy, temperature 0 | Makes exact token comparison unambiguous |
| Model | Qwen3-4B first, Qwen3-8B second | Faster bring-up before the target experiment |
| Full-KV tier | GPU-resident reference, then native pinned CPU prototype | Isolates self-spec correctness before offload |
| LMCache | Preserve as control; integrate after the CPU prototype passes | Replace the data plane without changing semantics |
| Execution | Eager first | CUDA graphs come after correctness |

`LMCache` is not part of the self-spec algorithm. It is a candidate full-KV
storage and transfer data plane. Keeping that boundary explicit prevents cache
connector bugs from being mistaken for draft/verify bugs.

## Definition of done

The October reproduction is complete only if all checks below pass:

1. A baseline full-KV run and the reproduction emit identical greedy token IDs
   for at least 20 fixed prompts.
2. Logs prove that sparse/compressed KV is used for draft and full KV is used
   for verify.
3. Full KV is not counted as permanently GPU-resident between verification
   rounds in the prototype allocator.
4. At least one verification round reloads full KV from pinned CPU memory.
5. Measurements report draft, reload, exposed reload wait, verify, accepted
   tokens, and peak GPU memory separately.
6. A clean-shell command reproduces the smoke test from documented pins.
7. The LongBench 16K-18K workload completes; the LongBench-v2 30K-40K workload
   either completes or has a documented A100 capacity limit with a smaller
   passing configuration.
8. A final report separates correctness, acceptance rate, latency, transfer
   time, throughput, and peak HBM instead of claiming paper-level parity.
9. The final report and reproduction guide are written in Chinese. Commands,
   code identifiers, raw log fields, and necessary English technical terms may
   remain unchanged, with a Chinese explanation on first use.

Stretch goals, not MVP requirements:

- overlap H2D full-KV reload with another request's draft;
- LMCache-backed storage;
- mixed draft and verify requests in one forward;
- adaptive draft length;
- FutureCapacityTracker;
- CUDA graph capture;
- PillarAttn or Vegas verification-guided selection;
- paper-level throughput reproduction.

## Schedule

### September 22-30: freeze the workspace and clean baseline

- Re-run the published-wheel vLLM 0.28.0 + LMCache 0.5.5 control.
- Create a new editable checkout from the official vLLM v0.28.0 tag; do not
  inherit commits from `vegas-vllm-pr1`.
- Create an LMCache checkout at the official v0.5.5 tag.
- Record exact source origins, tags, package hashes, driver, CUDA, and GPU.
- Keep the control environment separate from the development environment.
- Point `HF_HOME` at the copied LongBench cache under `~/vericache` and verify
  that both fixed workloads can be read without downloading data.
- Add one command each for environment checking, baseline generation, and the
  smallest GPU smoke test.

Exit criterion: both clean control and editable, unmodified source build
generate matching greedy output on A100.

### October 1-7: implement a full-KV self-spec reference

- Add a deterministic prompt fixture.
- Record full-KV baseline token IDs.
- Add the same-model draft/verify state machine while both KV views remain on
  GPU; begin with fixed draft length 4 and a single request.
- Record draft tokens, accepted lengths, corrected tokens, and final token IDs.
- Compare outputs by token ID, not decoded text.
- Handle accepted tokens, rejection, correction/bonus tokens, EOS, and a
  partially filled final block before adding compression.

Exit criterion: self-spec output equals full-KV output for all fixture prompts.

### October 8-14: add the A100 compressed draft KV path

- Document the current block table and physical KV ownership at draft and
  verify.
- Introduce separate logical handles for `compressed_kv` and `full_kv`.
- Add invariants preventing the two handles from silently referring to an
  unintended allocation.
- Implement a local sink-plus-recent-block KV selector used only by draft.
- Keep full KV on GPU and use it only for verification.
- Test draft lengths 4, 8, and 16 after the length-4 path is correct.

Exit criterion: logs prove the two KV views differ, and 20 fixed prompts still
match the baseline token for token.

### October 15-21: implement pinned-CPU full-KV offload

- Add a bounded pinned-CPU store for full-KV blocks.
- Store complete blocks after prefill/verification.
- Restore the full block table into transient GPU blocks before verification.
- Free transient verification blocks after accepted tokens are committed.
- Start with one request and fixed draft length 4.

Exit criterion: at least one full KV eviction and reload occurs with exact
output equality.

### October 22-24: add asynchronous transfer and instrumentation

- Put H2D reload on a dedicated CUDA stream.
- Add NVTX ranges: `draft_step`, `kv_reload_h2d`, `verify_pass`,
  `accept_reject`, and `kv_offload_d2h`.
- Measure total reload time, overlap, and exposed wait.
- Attempt two-request staggering only after the one-request path is stable.

Exit criterion: an Nsight Systems trace shows each phase and any real overlap.

### October 25-26: LMCache integration gate

- If the pinned-CPU data plane is stable, map its store/load interface onto
  LMCache and run a cold-store/warm-load correctness test.
- If connector integration threatens the deadline, retain the pinned-CPU
  implementation as the completed core reproduction and document LMCache as
  follow-up work. Do not weaken the self-spec correctness path to force it in.

Exit criterion: choose one tested data plane for the final reproduction.

### October 27-28: LongBench evaluation

- Run context lengths 2K, 4K, and 8K as gates, then the copied LongBench
  16K-18K workload and, if capacity permits, LongBench-v2 30K-40K.
- Run fixed draft lengths 4, 8, and 16.
- Compare baseline, self-spec without offload, and self-spec with offload.
- Save token comparisons, acceptance, latency, throughput, transfer timing,
  and peak HBM for each configuration.

Exit criterion: results are machine-readable and every reported configuration
states whether it passed correctness.

### October 29-30: clean-shell reproduction and report

- Rebuild or reactivate from a clean shell and repeat the smoke test twice.
- Run the 20-prompt correctness suite twice.
- Write the Chinese final report with exact commands, environment metadata,
  known failures, and A100-specific limitations.
- Preserve one representative Nsight Systems trace.

Exit criterion: the final command works twice without manual intervention.

### October 31: freeze and demonstrate

- No new features.
- Tag exact outer and vLLM commits.
- Run one baseline-versus-VeriCache demonstration from a clean shell.
- Archive the environment lock, logs, LongBench results, and final report.
- Mark incomplete stretch goals explicitly instead of extending the deadline.

## Architecture boundary

The control plane owns correctness and scheduling:

- request states: `DRAFTING`, `RELOAD_PENDING`, `VERIFYING`, `COMMITTING`;
- pending draft token IDs;
- accepted-prefix length and rollback;
- verify deadline and reload lifecycle;
- logical ownership of compressed and full KV.

The data plane owns bytes:

- allocate/free transient GPU blocks;
- store full KV to pinned CPU or LMCache;
- restore full KV to transient GPU blocks;
- report completion events and transfer sizes.

The attention path consumes the selected KV view but must not decide ownership.

## Daily working rule

Every implementation step must leave one small command that proves the new
property. Do not proceed from correctness to overlap, from one request to two,
or from pinned CPU to LMCache until the preceding command passes.
