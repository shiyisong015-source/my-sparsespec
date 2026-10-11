# Qwen3-8B v2: timed long-context experiments

The preserved v1 reference is tag `qwen3-8b-baseline-v1-20261009` on personal.
The complete v1 logs are in `research/baseline_v1/experiment_logs.tar.gz`.

## Environment

From the outer vericache workspace, `source scripts/activate_sparsespec.sh`.
The existing local model is `/data/models/Qwen3-8B`. V2 uses BF16, non-thinking,
TP=1, synchronous CPU scheduling, and no CUDA graph. The timing API rejects
async CPU and multi-rank runs so that traces are not silently misattributed.

## Prepare local data

Obtain GovReport, QMSum and MultiNews JSONL exports and LongBench v1's official
`dataset2prompt.json` from a pinned source revision. These files are inputs;
the preparer does not automatically download or substitute data.

```bash
python external/SparseSpec/experiments/v2/prepare_sample.py \
  --source-dir /absolute/path/to/longbench-jsonl \
  --templates /absolute/path/to/dataset2prompt.json \
  --output external/SparseSpec/log/v2/samples
```

The preparer validates with the Qwen tokenizer, records file hashes, excludes
inputs over 20K, reserves 2048 output + 15 boundary tokens, checks max r=.25
against the strict 6144 selected-KV bound, and produces disjoint screen (32)
and confirm (96) sets. Screening tries four length bins; insufficient bins
are reported in the manifest and filled from remaining valid data. Review
actual source/length coverage before launching the experiment. It preserves
natural EOS and does not select samples using a parameter-specific generated length.

## Run one case

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python external/SparseSpec/experiments/v2/run_case.py \
  --request-file external/SparseSpec/log/v2/samples/screen.jsonl \
  --batch 4 --k 6 --r .25 \
  --output external/SparseSpec/log/v2/manual/b4-k6-r025
```

`--mode ar` uses the full-KV BaseScheduler; it is not r=1 self-speculation.
`--disable-timing` provides a control for instrumentation overhead. `--seed`,
`--chunk-size`, `--output-tokens`, and `--max-mem-gib` are explicit controls.
`--async-cpu --disable-timing` enables the existing asynchronous scheduler for
either sparse or AR runs. Asynchronous cases retain engine elapsed time and
request outputs (plus acceptance rounds for sparse), without step timing.
For an async ablation, disable step timing in the synchronous controls too.
A completed output directory is never silently overwritten. The case report
records GPU inventory, visible devices, Git commit/diff and source-file hashes.

## GPU workers

Use a separate terminal/process for each GPU. Workers share the same output
root, sample, phase and seed. File locks protect complete cases, so each GPU
may independently shuffle the work order. A worker stops on a failed case;
inspect the log before resuming. Completed cases are validated then skipped.

```bash
python external/SparseSpec/experiments/v2/run_sweep.py \
  --gpu 0 --phase screen \
  --request-file external/SparseSpec/log/v2/samples/screen.jsonl \
  --output external/SparseSpec/log/v2/screen-seed330
```

Repeat with the other allocated physical GPU indices; `--gpu` accepts any
nonnegative index. Check allocation and occupancy before starting each worker.
Screen = 36 sparse cases plus 3 AR cases.
Confirm = 60 sparse cases plus 3 AR cases. Confirm is a separate explicit
launch after reviewing the screening results. A worker's `complete` status
only means its claimed jobs are done, not that other workers have finished.

## Trace semantics

- `steps.jsonl`: one scheduler step; actual batch/query tokens, request state,
  phase, KV lengths and budgets, resource snapshots, offload/restore list
  transitions, per-request committed token counts and CPU/GPU stage intervals.
- `step_start/end_perf_counter`: monotonic process timestamps, not wall-clock
  dates. `step_wall_seconds` excludes JSON serialization/draining after finish;
  the engine's summary time includes tracing overhead and final event draining.
- Request KV lengths are captured before `prepare_forward_inputs` adds this
  step's slots. Free-pool counters are recorded after updating requests.
- GPU stages: prepare, model_forward, sampling, kv_score_snapshot,
  rejection_sampling, kv_selection, update_requests. The last contains the
  rejection/selection stages: **do not add overlapping stage times**.
  CUDA-event intervals measure time on that stream, including possible host
  submission gaps, not a sum of individual kernel durations.
- GPU events are queried without waiting each step and pooled for reuse.
  Final close waits for remaining recorded events. No new per-step device
  synchronization is inserted.
- Offload/restore fields describe queue membership transitions. They are not
  isolated copy-engine transfer timings. More detailed DMA profiling is a
  separate extension.
- `rounds.jsonl`: accepted draft tokens exclude bonus; `step_id` links to the
  step where verification is consumed. Cold-start flags retain v1 meaning.
- Committed tokens include bonus/normal tokens, subtract the engine's EOS
  truncation, and must sum to final output. `eos_trimmed_by_request` records
  removed EOS/suffix tokens; raw verify acceptance remains in `rounds.jsonl`.
  Output-length overshoot inherited from the engine is retained and reported.
- Mixed-batch GPU time belongs to the whole batch; it is not a per-request
  latency. Real workload comparisons must include prefill/offload and measured
  actual batch, not only the configured maximum.

Validation artifacts are under `log/v2-validation/`. Synthetic fixtures there
are only software/capacity checks, not research benchmark data. The formal
new data has not yet been downloaded and the full sweep has not been launched.

## Trace validation

```bash
python external/SparseSpec/experiments/v2/validate_trace.py \
  external/SparseSpec/log/v2-validation/k6-timing \
  external/SparseSpec/log/v2-validation/long-timing
```

Use `--equal-outputs` for a deterministic greedy on/off control. A fixed random
seed does not guarantee identical stochastic trajectories across executions
of this backend; stochastic equality is not used as a pass criterion.
These short smoke tests do not quantify stable instrumentation overhead;
repeat warmed-up timing controls before using the new trace for performance claims.
