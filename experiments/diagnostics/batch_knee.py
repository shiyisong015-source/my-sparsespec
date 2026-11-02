"""Fixed-shape full-model forward sweep, full KV resident, no transfers.

Launch under torchrun. This measures compute-path capacity, not generated or
accepted-token throughput. Synthetic KV/tokens remove trajectory and admission
effects. Eager timings include CPU launch gaps and TP communication. Native
attention/score kernels, MLP, and vocabulary projection/gather are unchanged.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
import torch.distributed as dist

from serve.distribute.tp import setup_tp_worker
from serve.model.model import Attention, FeedForward, KVPool, Transformer
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.sampling.rejection_sampler import RejectionSampler
from serve.utils import setup_seed


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    return values[lower] + (index - lower) * (values[math.ceil(index)] - values[lower])


def metadata(pool, *, batch, context, phase, stride, ratio, min_budget, seed):
    query = 1 if phase == "draft" else stride + 1
    length = context + query
    visible = min(context, max(min_budget, math.ceil(context * ratio))) + query
    visible = visible if phase == "draft" else length
    cls, device = pool.kv_cache_class, pool.device
    make = lambda values: torch.tensor(values, dtype=torch.int32, device=device)
    flatten = cls.draft_kv_indices_tensor_buf
    if phase == "draft":
        # Independent per-layer/head random subsets; each request's shifted
        # subset is still unique. Appended query is the last physical page.
        generator = torch.Generator(device=device).manual_seed(seed + context)
        prefix = torch.stack([torch.randperm(context, generator=generator, device=device)[:visible - 1]
                              for _ in range(cls.NUM_LAYERS * cls.NUM_KV_HEADS)])
        prefix = prefix.view(cls.NUM_LAYERS, cls.NUM_KV_HEADS, visible - 1)
        for request in range(batch):
            first = request * visible
            flatten[..., first:first + visible - 1] = (
                (prefix + request * 97) % context + request * length).to(torch.int32)
            flatten[..., first + visible - 1] = request * length + context
    tokens = torch.randint(10, pool.vocab_size, (batch * query,), dtype=torch.int32, device=device)
    return dict(token_ids=tokens, token_indptr=make([i * query for i in range(batch + 1)]),
                pos_offsets=make([context] * batch),
                kv_indices=torch.arange(batch * length, dtype=torch.int32, device=device),
                all_kv_indptr=make([i * length for i in range(batch + 1)]),
                all_kv_len_arr=make([length] * batch), kv_len_arr=make([visible] * batch),
                kv_indptr=make([i * visible for i in range(batch + 1)]),
                request_type=make([1 if phase == "draft" else 2] * batch),
                verify_kv_indptr=make([i * visible for i in range(batch)]),
                flatten_indices=flatten, flatten_indices_len=batch * visible if phase == "draft" else 0,
                dump_logits_len=batch * visible if phase == "verify" else 0)


@contextmanager
def annotate(pool):
    """Profile pass only; no instrumentation in the timing pass."""
    def wrapped(function, label):
        def call(*args, **kwargs):
            with torch.profiler.record_function("batch_knee." + label):
                return function(*args, **kwargs)
        return call

    with patch.object(Attention, "forward", wrapped(Attention.forward, "attention_block_with_tp")), \
         patch.object(FeedForward, "forward", wrapped(FeedForward.forward, "mlp_block_with_tp")), \
         patch.object(pool, "batch_attn_fwd", wrapped(pool.batch_attn_fwd, "attention_and_scores")), \
         patch.object(pool.wrappers[0], "run_attention",
                      wrapped(pool.wrappers[0].run_attention, "attention_core")), \
         patch.object(pool.wrappers[0], "rematerialize_scores",
                      wrapped(pool.wrappers[0].rematerialize_scores, "selection_scores")):
        yield


def profile_once(model, pool, ids, path):
    torch.cuda.synchronize()
    with annotate(pool), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False) as profiler:
        with torch.profiler.record_function("batch_knee.full_forward"):
            output = model(ids)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    rows = []
    for event in profiler.key_averages():
        if event.key.startswith("batch_knee."):
            rows.append(dict(name=event.key, calls=event.count,
                             cpu_total_us=event.cpu_time_total,
                             device_total_us=event.device_time_total,
                             self_device_us=event.self_device_time_total))
    del output
    return dict(trace=str(path), ranges=rows,
                scope="Separate instrumented pass; nested ranges overlap. GPU kernel activity is not SM occupancy.")


def main(args):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float16)
    setup_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model = Transformer.from_pretrained(args.model, device=device, dtype=torch.float16)
    model = setup_tp_worker(model, world, device, torch.float16, args.reduce_method)
    group = dist.new_group(backend="gloo") if world > 1 else None

    def gather(value):
        if group is None:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value, group=group)
        return result

    def barrier():
        if group is not None:
            dist.barrier(group=group)

    cfg, cls = model.config, CompactCachePtr
    spec = dict(spec_stride=args.stride, budget_ratio=args.ratio, num_min_budget=args.min_budget,
                max_batch_size=max(args.batches), max_selected_kv_len=6144)
    # Fresh process: match run_inference's k-dependent probability row policy.
    RejectionSampler.MAX_SPEC_LEN = 10
    if args.stride + 1 >= RejectionSampler.MAX_SPEC_LEN:
        RejectionSampler.MAX_SPEC_LEN = math.ceil((args.stride + 1) / 4) * 4
    cls.initialize_class_metadata(dtype=torch.float16, head_dim=cfg.head_dim,
        num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads,
        num_layers=cfg.num_layers, capacity=cls._MAX_KV_CONTEXT_LEN,
        device=device, max_batch_size=max(args.batches), spec_cache_args=spec)
    assert cls.MAX_SPEC_STRIDE == args.stride + 1
    page_bytes = cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * 2
    per_request_probs = RejectionSampler.MAX_SPEC_LEN * cfg.vocab_size * 4
    per_request_selection = cfg.num_layers * cfg.num_kv_heads * cls.MAX_LEN_SELECTED_KV_CACHE * 4
    source_root = Path(__file__).resolve().parents[2]
    source_names = ["experiments/diagnostics/batch_knee.py", "serve/attention/backend.py",
                    "serve/model/model.py", "serve/request/kv_cache_ptr/compact.py",
                    "serve/request/kv_cache_ptr/pillar.py"]
    source_hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                     for name in source_names}
    manifest = dict(model=args.model, rank=rank, tp=world, gpu=torch.cuda.get_device_name(device),
                    physical_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    page_bytes_per_rank=page_bytes, probability_rows=RejectionSampler.MAX_SPEC_LEN,
                    native_max_query_rows=cls.MAX_SPEC_STRIDE,
                    packed_verify_query_rows=cls.MAX_SPEC_STRIDE * cfg.num_qo_heads // cfg.num_kv_heads,
                    per_request_draft_prob_bytes=per_request_probs,
                    per_request_selection_buffer_bytes=per_request_selection,
                    source_sha256=source_hashes, dtype="float16", graph=False, offload=False,
                    limitations=["Synthetic fixed-shape state, not generated-token throughput or an online scheduler.",
                                 "All full KV stays on GPU. No D2H/H2D/repack/restore is called.",
                                 "Timed forward includes native scores, all layers, vocabulary head and TP communication.",
                                 "No sampling, rejection, selection update, prefill or scheduler in timed forward.",
                                 "Resident request probability/selection storage is allocated at native k=8 sizes.",
                                 "Eager event intervals include submission gaps; a throughput plateau alone is not proof of SM saturation.",
                                 "No 512-query scheduler cap: query count is phase query length times actual batch.",
                                 "Memory-guard skips are conservative unmeasured points, not observed OOMs."])
    save(args.output / f"manifest.rank{rank}.json", manifest)
    rows = []
    with torch.inference_mode():
        for context in args.contexts:
            for phase in args.phases:
                query = 1 if phase == "draft" else args.stride + 1
                visible = (min(context, max(args.min_budget, math.ceil(context * args.ratio))) + query
                           if phase == "draft" else context + query)
                # Native aggregate index capacities are k-independent. Keep
                # them unchanged and report their real feasibility boundary.
                base_free = min(gather(torch.cuda.mem_get_info(device)[0]))
                plans = []
                for batch in args.batches:
                    pages = batch * (context + query)
                    bytes_needed = pages * page_bytes + batch * (per_request_probs + per_request_selection)
                    reason = None
                    if batch >= cls._MAX_BATCH_SIZE or batch * query >= cls._MAX_NNZ:
                        reason = "query_or_batch_metadata_capacity"
                    elif pages > cls._MAX_KV_CONTEXT_LEN:
                        reason = "full_index_capacity"
                    elif phase == "draft" and batch * visible >= cls.MAX_TOTAL_DRAFT_KV_LEN:
                        reason = "draft_index_capacity"
                    elif phase == "verify" and batch * visible >= cls.MAX_TOTAL_VERIFY_KV_LEN:
                        reason = "verify_index_capacity"
                    elif visible >= cls.MAX_LEN_SELECTED_KV_CACHE and phase == "draft":
                        reason = "per_request_selected_index_capacity"
                    elif bytes_needed + args.workspace_gib * 2**30 > base_free:
                        reason = "memory_guard_not_measured"
                    plans.append((batch, pages, bytes_needed, reason))
                feasible = [b for b, _, _, reason in plans if reason is None]
                trace_batches = {feasible[0], feasible[-1]} if feasible and args.trace else set()
                for batch, pages, bytes_needed, reason in plans:
                    row = dict(phase=phase, context_tokens_before_forward=context, batch=batch,
                               query_tokens_per_request=query, total_query_tokens=batch * query,
                               full_kv_tokens_per_request=context + query,
                               visible_kv_tokens_per_request=visible,
                               configured_ratio=args.ratio,
                               historical_visible_ratio=(visible - query) / context,
                               full_kv_bytes_per_rank=pages * page_bytes,
                               resident_probability_bytes_per_rank=batch * per_request_probs,
                               resident_selection_bytes_per_rank=batch * per_request_selection,
                               workspace_guard_bytes=int(args.workspace_gib * 2**30),
                               phase_start_free_bytes_per_rank=base_free,
                               state_bytes_estimate_per_rank=bytes_needed,
                               status="skipped" if reason else "running", reason=reason)
                    if reason:
                        rows.append(row)
                        save(args.output / f"results.rank{rank}.json", rows)
                        if rank == 0:
                            print("BATCH_KNEE_SKIP", phase, context, batch, reason, flush=True)
                        continue
                    print("BATCH_KNEE_START", rank, phase, context, batch, flush=True)
                    # No production scheduler is constructed: transfers are
                    # absent, rather than replaced by D2D copies or sleeps.
                    pool = KVPool(cfg, pages, device, cls, dtype=torch.float16, allocate_legacy_cpu=False)
                    pool._physical_mem_gpu.normal_(0, .1)
                    model.attach_kv_pool(pool)
                    probability_storage = torch.empty((batch, RejectionSampler.MAX_SPEC_LEN, cfg.vocab_size),
                                                      dtype=torch.float32, device=device)
                    selection_storage = torch.empty((batch, cfg.num_layers, cfg.num_kv_heads,
                                                     cls.MAX_LEN_SELECTED_KV_CACHE),
                                                    dtype=torch.int32, device=device)
                    meta = metadata(pool, batch=batch, context=context, phase=phase, stride=args.stride,
                                    ratio=args.ratio, min_budget=args.min_budget, seed=args.seed)
                    assert pool.wrappers[0].MAX_SPEC_STRIDE == args.stride + 1
                    if args.stride == 8:
                        assert not pool.wrappers[0].uses_paged_scores
                    row["kernel_buffers"] = dict(max_spec_stride=pool.wrappers[0].MAX_SPEC_STRIDE,
                        score_tile_q=pool.wrappers[0].DISPATCH_CTA_TILE_Q,
                        uses_paged_scores=pool.wrappers[0].uses_paged_scores,
                        draft_probability_shape=list(probability_storage.shape),
                        selection_shape=list(selection_storage.shape),
                        query_shape=list(meta["token_ids"].shape),
                        draft_indices_capacity=list(cls.draft_kv_indices_tensor_buf.shape),
                        score_output_capacity=list(pool.attn_metadata["dump_logits"].shape))
                    torch.cuda.synchronize()
                    prepared = time.perf_counter()
                    pool.pre_attn_fwd(**meta)
                    torch.cuda.synchronize()
                    row["initial_metadata_plan_wall_ms"] = (time.perf_counter() - prepared) * 1000
                    for _ in range(args.warmup):
                        output = model(meta["token_ids"])
                    assert bool(torch.isfinite(output).all()), (phase, context, batch)
                    del output
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    timings = []
                    for repetition in range(args.repeats):
                        barrier()
                        torch.cuda.synchronize()
                        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        started = time.perf_counter()
                        begin.record()
                        for _ in range(args.calls):
                            output = model(meta["token_ids"])
                            del output
                        end.record()
                        end.synchronize()
                        timings.append(dict(event_ms=begin.elapsed_time(end) / args.calls,
                                            wall_ms=(time.perf_counter() - started) * 1000 / args.calls))
                    row.update(status="complete", reason=None,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        allocated_before_profile_bytes=torch.cuda.memory_allocated(),
                        timing_samples=timings, kv_d2h_bytes=0, kv_h2d_bytes=0, repack_calls=0, restore_calls=0)
                    if batch in trace_batches:
                        barrier()
                        row["profile"] = profile_once(model, pool, meta["token_ids"],
                            args.output / f"trace-{phase}-l{context}-b{batch}.rank{rank}.json")
                    ranks = gather(row)
                    # All control decisions and feasible shapes must agree.
                    assert all(r["batch"] == batch and r["status"] == "complete" for r in ranks)
                    row["tp_critical_event_ms_samples"] = [
                        max(r["timing_samples"][rep]["event_ms"] for r in ranks) for rep in range(args.repeats)]
                    row["tp_critical_wall_ms_samples"] = [
                        max(r["timing_samples"][rep]["wall_ms"] for r in ranks) for rep in range(args.repeats)]
                    times = row["tp_critical_event_ms_samples"]
                    row.update(median_forward_ms=statistics.median(times), p90_forward_ms=percentile(times, .9),
                        p25_forward_ms=percentile(times, .25), p75_forward_ms=percentile(times, .75),
                        query_tokens_per_second=batch * query * 1000 / statistics.median(times),
                        request_positions_per_second=batch * 1000 / statistics.median(times))
                    rows.append(row)
                    save(args.output / f"results.rank{rank}.json", rows)
                    if rank == 0:
                        print("BATCH_KNEE_RESULT", json.dumps({key: row[key] for key in (
                            "phase", "context_tokens_before_forward", "batch", "total_query_tokens",
                            "median_forward_ms", "query_tokens_per_second", "peak_allocated_bytes")}), flush=True)
                    model.attach_kv_pool(None)
                    del meta, pool, probability_storage, selection_storage, ranks
                    gc.collect()
                    torch.cuda.empty_cache()
                    barrier()
    assert all(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items()), "Source changed during sweep"
    save(args.output / f"complete.rank{rank}.json", dict(complete=True, points=len(rows),
        measured=sum(r["status"] == "complete" for r in rows), source_sha256=source_hashes))
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=.05)
    parser.add_argument("--min-budget", type=int, default=192)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    parser.add_argument("--phases", nargs="+", choices=["draft", "verify"], default=["draft", "verify"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--workspace-gib", type=float, default=6.)
    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--reduce-method", default="customized", choices=["customized", "naive"])
    parser.add_argument("--trace", action="store_true", help="Separate CUPTI trace at smallest/largest feasible batches")
    args = parser.parse_args()
    if (min(args.contexts + args.batches) < 1 or args.stride < 1 or not 0 < args.ratio <= 1
            or min(args.warmup, args.repeats, args.calls) < 1 or args.workspace_gib <= 0):
        parser.error("Require positive sizes, iterations, workspace, and 0<ratio<=1")
    args.batches, args.contexts = sorted(set(args.batches)), sorted(set(args.contexts))
    main(args)
