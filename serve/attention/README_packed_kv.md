# Compact KV serving

Use `serve.run.run_inference(..., kv_cache="packed", batch_policy="greedy")`
to run the compact hot/cold path. Existing `full`, `streaming`, and `pillar`
entries continue to use their original schedulers. This implementation uses
the original Transformer, Q/K normalization, RoPE, projections and MLP.

## Request lifecycle

1. Chunked prefill writes a contiguous full KV lease in the shared GPU arena.
2. The first speculative round uses the full context. Verify rewrites the
   speculative KV and commits the rejection sampler's accepted prefix.
3. Per-layer/head selection splits history into hot and cold. Hot is packed
   in GPU memory. A gather kernel writes cold directly to mapped pinned CPU
   memory: there is no cold GPU staging allocation or second D2H copy.
   Only after every layer has finished
   and migration completes on every TP rank is the original full lease returned.
4. Draft appends to the hot tail. RoPE offsets use logical full context length.
5. Verify restores the cold history into a GPU workspace. One batched attention
   grid reads hot causal and cold noncausal chunks through separate pointers.
   A merge kernel combines chunk outputs using their softmax denominators.
6. Scores normalized by the merged LSE drive the next selection. Old hot,
   restored cold and CPU backing stay owned until the replacement is ready.
7. Completion returns all request leases. Returning arena intervals makes them
   available for admission; it does not reduce the arena's reserved GPU memory.

Older slots can contain different token IDs for each local layer/head. RoPE
is already embedded in their keys, and all queries see those older positions.
The recent suffix and speculative tail retain causal order. Permanent original
token IDs are unnecessary for this first version's attention/repacking path.

## Memory and TP

`packed_gpu_pool_mib` and `packed_cpu_pool_mib` specify bounded per-rank arenas.
Full KV, hot KV and restored KV share the GPU arena. CPU cold histories
share the pinned arena. Full and temporary restore leases allocate from the
high end; persistent hot leases allocate from the low end. The scheduler holds
fixed per-request hot/cold lease capacities derived from bounded output length,
k and r. Each layer's live KV is compacted into a contiguous tensor view at the
front of that lease; unused capacity is after the entire tensor, with no gaps
between its layers. Cold DMA copies live KV, excluding the unused reservation.
This avoids growing replacements losing access to the prior round's smaller
free interval. Low-level packing callers can supply `hot_capacity` and
`cold_capacity`; their defaults reserve only the current tensor size.

Admission covers initial full KV + packed replacement and the later hot KV +
restored cold + packed replacement, including double CPU backing and maximum
requested history growth. It credits the full lease retired by the first pack
when calculating future restore headroom. It does not charge retired full KV
and its later restored copy at the same time.
It may wait for an active request to release full KV before admitting another.
Allocation failures are reported; there is no automatic smaller k/r/batch limit.
Before restoring a verify group, simulation uses the same two-end GPU and
first-fit CPU allocation rules as the arenas, including fixed replacement lease
sizes. A request with one remaining output token needs no replacement after
that verify. The feasible ready prefix shares one model
forward. This conservatively reserves all replacements even though actual
adoption is sequential. A group smaller than the batch limit is recorded.
Requests not yet ready for verify continue draft/prefill in the same forward.
Ready requests excluded by the capacity check stay parked until they can verify;
their draft length is never extended beyond the configured k.

The default GPU arena is at most 4 GiB, subject to the requested total GPU memory
budget with 2 GiB kept for transient non-KV allocations. Default pinned CPU
capacity is twice the GPU arena. Set explicit budgets for larger workloads.
Contiguous-arena fragmentation can still make a configuration infeasible.

Each TP rank owns its local KV heads. A Gloo control group coordinates setup,
allocation failures, restore/migration completion and sampled token decisions.
Only rank 0 samples; token IDs and accepted lengths are shared with all ranks.
GPU device faults occurring inside model/NCCL collectives require restarting
the worker; the pre-forward coordination does not recover such device faults.

## Current execution limits

This first version runs eager synchronous scheduling. Packing/DMA follows the
whole model forward and finishes before the next forward. Prefill, draft and
verify requests can share a projection/MLP batch when arena capacity permits.
CUDA attention, KV append and score
rematerialization are batched; the independent torch reference is per request.
Query/head tiles and hot/cold chunks share one attention grid. The default
chunk planner targets two TBs per SM and bounds KV chunks to 128–2048 tokens.
This is a heuristic, not an occupancy optimum. Actual kernel occupancy is
queried from CUDA; estimated wave count/idle ratio are logged with grid size,
register/shared-memory use and workspace bytes. Unequal task durations and
causal masking mean the wave formula is not a measured utilization value.
CUDA graphs, asynchronous CPU scheduling and layerwise transfer overlap are
not implemented in this entry. Transfers still complete at whole-forward
boundaries; there is no inference/transfer overlap in this version.

One pinned CPU upload supplies the attention plan's metadata; a second supplies
RoPE offsets. CPU allocations explicitly specify their device even when the
model sets PyTorch's default device to CUDA. Plans bind their metadata and buffer
lifetimes to one compute stream once per forward. Per-layer guards reject stream
changes, repeated writes and changing query-head geometry. The GPU batch path
checks Q/K/V geometry for the whole batch; independent reference frames retain
their per-request tensor checks.

BF16 batched verification can change a greedy argmax near a numerical tie,
including in the dense self-spec control. Validation records exact equality
separately from bounded logit differences. Consult each run's `validation.json`
for its actual result; an earlier implementation's differences are not evidence
of differences in the current kernels.

Logs include `rounds.jsonl`, `kv-events.rankN.jsonl`, `requests.json`, and
`summary.json`. Events record phase, query count, logical offset, hot/cold count,
arena ownership, admission waits and actual batch sizes. Transfer byte totals
measure KV payload; they are not DMA duration or PCIe utilization measurements.

## Run the real-model validation

```bash
cd /data/users/songshiyi/vericache/external/SparseSpec
source /data/users/songshiyi/vericache/scripts/activate_reproduction.sh
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  torchrun --standalone --nproc_per_node=2 \
  -m serve.tests.validate_packed_serving \
  --output "log/packed-validation-$(date +%Y%m%d-%H%M%S)"
```

Select free physical GPU IDs. Single-GPU validation uses `python -m` with one
visible GPU. Add `--public-entry` to validate the public `run_inference` entry.
The tests load Qwen3-8B, compare greedy generation to dense AR, exercise k=14,
r=.05/.25, temperature=.65, mixed prefill/draft/verify, arena pressure, 4.8K
context and 20.4K context in an arena unable to hold two full histories.
On a 20K argmax mismatch they also run a dense self-spec control and retain
the first differing logits in `greedy-check.json`. Exact output equality and
bounded numerical differences are separate statuses with unchanged bounds.

`serve.tests.test_batched_kv_attention` checks FP16/BF16 outputs, LSE and per-head
selection scores against dense FP64 references with 20K histories, ragged query
counts, all-hot histories, poisoned unused capacity and mixed prefill/verify.
Enable it with `RUN_PACKED_KV_GPU=1`. `serve.tests.bench_aime_kv_overhead` measures
same-device native SparseSpec, dense self-spec, resident sparse and CPU offload;
the thinking default is enabled. Resident control gathers cold into a GPU mirror
and restores by D2D, preserving the same sparse math and verify trajectories.
`--modes original_sparse resident_sparse packed_offload` selects the three
corresponding timing arms. All arms still use the same model and reservations.

`test_packed_scheduler` also runs a small deterministic model through actual KV
kernels for 20K input and 512 decoded tokens to regress repeated buffer growth.
This checks arena liveness, not Qwen throughput. `test_kv_partition` poisons unused
CPU capacity to verify that it is neither written nor included in restore DMA.

Main implementation files: `full_forward.py` owns dense initial history;
`packed_forward.py` owns compact round boundaries; `packed_batch.py` adapts the
model KV calls; `packed_scheduler.py` manages requests and admission;
`packed_run.py` loads the model and arenas; `run.py` exposes the optional entry.
