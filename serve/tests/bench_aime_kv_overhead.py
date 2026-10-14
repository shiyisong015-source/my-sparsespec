"""AIME pilot: compact KV costs and a GPU-resident cold-copy control.

Run after activating scripts/activate_reproduction.sh. All modes share one
loaded model, device, requests and arena reservations. Main measurements have
no CUDA timing events or per-step file writes. A separate diagnostic samples
only a bounded number of packed verify rounds. The resident control maps CPU
arena views to an extra GPU mirror; it preserves gather, D2D reconstruction,
selection and split attention. It is a cost control, not a serving policy.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from serve.attention import full_forward, kv_partition, packed_attention, packed_forward
from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.kv_partition import PackedKVRegion
from serve.distribute.tp import setup_tp_worker
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import KVPool, Transformer
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request
from serve.scheduler.packed_scheduler import PackedSpecScheduler
from serve.scheduler.spec_scheduler import SpecScheduler
from serve.utils import MetricBucket, setup_seed, wandb_logger


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def describe(values):
    if not values:
        return dict(count=0)
    ordered = sorted(values)
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                p95=ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)])


class Events:
    """Deferred event resolution; adds no device barrier inside model steps."""

    def __init__(self, limit, minimum_history):
        self.limit = limit
        self.minimum_history = minimum_history
        self.rounds = 0
        self.active = False
        self.pairs = []
        self.host = defaultdict(list)
        self.window = []

    def call(self, name, function, *args, **kwargs):
        if not self.active:
            return function(*args, **kwargs)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        host_start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.host[name].append((time.perf_counter() - host_start) * 1000)
            end.record()
            self.pairs.append((name, start, end))

    def resolve(self):
        groups = defaultdict(list)
        for name, start, end in self.pairs:
            # Resolve after the timed run, including all transfer streams.
            end.synchronize()
            groups[name].append(start.elapsed_time(end))
        return dict(window=self.window, event_pairs=len(self.pairs),
                    gpu_ms={name: describe(rows) for name, rows in groups.items()},
                    host_ms={name: describe(rows) for name, rows in self.host.items()})


class BenchScheduler(PackedSpecScheduler):
    """Small CPU counters; no logits copies or per-step disk access."""

    def __init__(self, *args, diagnostic=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic = diagnostic
        self.phases = Counter()
        self.batch_histogram = Counter()
        self.selections = []
        self.round_rows = []

    def _forward(self, entries):
        for _, _, ids, phase in entries:
            self.phases[phase + "_query_tokens"] += len(ids)
        self.batch_histogram[len(entries)] += 1
        return super()._forward(entries)

    def _verify_many(self, states, active, completed):
        diagnostic = self.diagnostic
        eligible = [s for s in states if isinstance(s.region, PackedKVRegion)
                    and s.region.history_length >= (diagnostic.minimum_history if diagnostic else 0)]
        sampled = diagnostic is not None and eligible and diagnostic.rounds < diagnostic.limit
        if sampled:
            diagnostic.rounds += 1
            diagnostic.active = True
            diagnostic.window.append(dict(step=self.step, verify_batch=len(states),
                requests=[dict(request_id=s.request.request_id, history=s.region.history_length,
                               hot=s.region.hot_count, cold=s.region.cold_count,
                               query_tokens=len(s.drafts) + 1) for s in eligible]))
        try:
            return super()._verify_many(states, active, completed)
        finally:
            if diagnostic is not None:
                diagnostic.active = False

    def _commit_verify(self, state, active, completed, frame, restored, sampled, probabilities):
        packed = isinstance(state.region, PackedKVRegion)
        before_migrations = self.metrics["migrations"]
        before_accepted, before_proposed = self.accepted, self.proposed
        before_output = len(state.output)
        before_history = state.round_start
        result = super()._commit_verify(state, active, completed, frame, restored, sampled, probabilities)
        self.round_rows.append(dict(request_id=state.request.request_id, history=before_history,
            proposed=self.proposed - before_proposed, accepted=self.accepted - before_accepted,
            emitted=len(state.output) - before_output, packed=packed))
        if self.metrics["migrations"] != before_migrations:
            region = state.region
            self.selections.append(dict(history=region.history_length, hot=region.hot_count,
                                        cold=region.cold_count))
        return result


@contextmanager
def copy_control(cpu_pool, shadow, resident, diagnostic=None):
    original = PackedKVRegion._copy_contiguous_layers
    original_pack = kv_partition.pack_selected_kv

    def mirror(view):
        # CPU slices retain layer gaps. Preserve both strides and byte offset.
        offset = view.data_ptr() - cpu_pool._buffer.data_ptr()
        if not view.numel():
            return shadow[:0].view(view.dtype).view(view.shape)
        extent = sum((size - 1) * stride for size, stride in zip(view.shape, view.stride())) + 1
        if offset < 0 or offset % view.element_size() or offset + extent * view.element_size() > shadow.numel():
            raise ValueError("Cold view is outside the mirrored CPU arena")
        return shadow.view(view.dtype).as_strided(view.shape, view.stride(),
                                                 offset // view.element_size())

    def copy(destination, source):
        direction = "D2H" if destination.device.type == "cpu" else "H2D"
        if resident:
            if destination.device.type == "cpu":
                destination = mirror(destination)
            if source.device.type == "cpu":
                source = mirror(source)
            direction = "D2D_" + direction + "_control"
        if diagnostic is None:
            return original(destination, source)
        return diagnostic.call(direction, original, destination, source)

    def gather(source, selected, destination, count, **kwargs):
        if resident and destination.device.type == "cpu":
            destination = mirror(destination)
        return original_pack(source, selected, destination, count, **kwargs)

    with patch.object(PackedKVRegion, "_copy_contiguous_layers", staticmethod(copy)), \
            patch.object(kv_partition, "pack_selected_kv", gather):
        yield


@contextmanager
def phases(diagnostic, model):
    if diagnostic is None:
        yield
        return
    from contextlib import ExitStack
    with ExitStack() as stack:
        def wrap(module, attribute, label):
            original = getattr(module, attribute)
            def wrapped(*args, **kwargs):
                name = label(*args, **kwargs) if callable(label) else label
                return diagnostic.call(name, original, *args, **kwargs)
            stack.enter_context(patch.object(module, attribute, wrapped))

        wrap(model, "forward", "model_forward_parent")
        wrap(kv_partition, "pack_selected_kv", lambda source, selected, destination, *a, **kw:
             "gather_cold_destination" if destination.device.type == "cpu" else "gather_hot_GPU")
        wrap(full_forward, "partition_pages", "partition_indices")
        wrap(packed_forward, "partition_pages", "partition_indices")
        for module in (full_forward, packed_attention):
            wrap(module, "attention_segment", lambda *a, **kw:
                 "attention_causal" if kw["causal"] else "attention_cold")
            wrap(module, "selection_scores", "selection_score")
        wrap(packed_attention, "merge_attention_states", "attention_merge")
        wrap(packed_forward.PackedForward, "selection_plan", "selection_plan_parent")
        from serve.attention import batched_kv_attention
        class TimedLaunch:
            def __init__(self, kernel, label):
                self.kernel, self.label = kernel, label
            def __getitem__(self, grid):
                launch = self.kernel[grid]
                return lambda *a, **kw: diagnostic.call(self.label, launch, *a, **kw)
        for attribute, label in (("_append_batch", "batch_append"), ("_split_attention", "batch_attention"),
                                 ("_merge_batch", "batch_merge"), ("_batch_scores", "batch_scores")):
            stack.enter_context(patch.object(batched_kv_attention, attribute,
                TimedLaunch(getattr(batched_kv_attention, attribute), label)))
        original_finish = PackedKVRegion.finish_offload
        def finish(region, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original_finish(region, *args, **kwargs)
            finally:
                if diagnostic.active:
                    diagnostic.host["finish_offload_wait"].append((time.perf_counter() - started) * 1000)
        stack.enter_context(patch.object(PackedKVRegion, "finish_offload", finish))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--data", default="eval/benchmarks/data/aime24/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output-length", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--r", type=float, default=.1)
    parser.add_argument("--minimum", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("original_sparse", "full_self_spec", "resident_sparse", "packed_offload"),
                        default=["original_sparse", "full_self_spec", "resident_sparse", "packed_offload"])
    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--gpu-mib", type=int, default=2048)
    parser.add_argument("--cpu-mib", type=int, default=2048)
    parser.add_argument("--diagnostic-rounds", type=int, default=8)
    parser.add_argument("--diagnostic-min-history", type=int, default=256)
    args = parser.parse_args()
    if args.repetitions < 1 or len(args.modes) != len(set(args.modes)):
        raise ValueError("Use positive repetitions and distinct benchmark modes")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This cost pilot requires TP1 for a same-device comparison")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    selected = random.Random(args.seed).sample(rows, args.samples)
    inputs = [tokenizer.apply_chat_template([dict(role="user", content=row["problem"]
              + "\nPlease reason step by step, and put your final answer within \\boxed{}.")],
              tokenize=True, add_generation_prompt=True, enable_thinking=not args.no_thinking)
              for row in selected]
    eos = tuple({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
    manifest = dict(config=vars(args), device=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, dtype="bfloat16", temperature=0,
        cuda_graph=False, async_cpu=False, tp=1, eos_token_ids=eos,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=[dict(id=row["id"], input_tokens=len(ids), input_ids=ids)
                  for row, ids in zip(selected, inputs)],
        timing="Only scheduler.run and terminal GPU completion; weights/arenas/warmup excluded",
        resident_control="Cold CPU views mapped to a reserved GPU shadow; both transfers become D2D",
        comparison_limit="Original scheduler/attention differs; full self-spec uses full-context drafting. "
                         "Only resident vs offload isolates CPU transfers with identical sparse math.")
    save(output / "manifest.json", manifest)
    print("CONFIG", json.dumps({**vars(args), "input_lengths": [len(ids) for ids in inputs]}), flush=True)
    started = time.perf_counter()
    model = Transformer.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    model = setup_tp_worker(model, 1, str(device), torch.bfloat16, "naive")
    print(f"MODEL_LOADED seconds={time.perf_counter() - started:.3f}", flush=True)
    gpu = KVBufferPool(args.gpu_mib * 1024**2, device=device)
    cpu = KVBufferPool(args.cpu_mib * 1024**2)
    shadow = torch.empty(cpu.capacity_bytes, dtype=torch.uint8, device=device)

    # Reuse the same weights for the pre-existing pillar path. Its reservation
    # also stays present in every arm, so free GPU memory is comparable.
    cfg = model.config
    token_bytes = cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * 2
    capacity = gpu.capacity_bytes // token_bytes
    additional = dict(spec_stride=args.k, budget_ratio=args.r, num_min_budget=args.minimum,
                      num_sink_tokens=4, max_batch_size=args.batch)
    PillarCachePtr.initialize_class_metadata(dtype=torch.bfloat16, head_dim=cfg.head_dim,
        num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, num_layers=cfg.num_layers,
        capacity=capacity, device=str(device), max_batch_size=args.batch, spec_cache_args=additional)
    original_pool = KVPool(cfg, capacity, str(device), PillarCachePtr, dtype=torch.bfloat16)
    original_args = dict(eos_token_id=eos, admit_policy="offloading", kv_cache="pillar",
        async_cpu=False, tp_ranks=1, temperature=0., enable_torch_profiler=False,
        profiler_result_dir=None, batch_policy="greedy", enable_cuda_graph=False,
        chunk_size=args.chunk_size, trace_name="aime-overhead", **additional)
    manifest["reserved_gpu_bytes_after_setup"] = torch.cuda.memory_allocated()
    manifest["experimental_shadow_bytes"] = shadow.numel()
    manifest["token_kv_bytes"] = token_bytes
    save(output / "manifest.json", manifest)
    results, warmups = [], []
    reference = {}

    def trial(mode, stage, repeat, limit, diagnostic=None):
        setup_seed(args.seed)
        requests = [Request(ids, limit) for ids in inputs]
        if mode == "original_sparse":
            model.attach_kv_pool(original_pool)
            wandb_logger.log_data = MetricBucket()
            worker = SpecScheduler(original_pool, CUDAGraphModelWrapper(model), **original_args)
        else:
            gpu._peak_bytes = cpu._peak_bytes = 0  # Both arenas are idle here.
            worker = BenchScheduler(model, gpu, cpu, max_batch_size=args.batch,
                chunk_size=args.chunk_size, spec_stride=args.k, budget_ratio=args.r,
                num_min_budget=args.minimum, temperature=0., eos_token_id=eos,
                retain_full_reference=mode == "full_self_spec", diagnostic=diagnostic)
        print(f"START stage={stage} repeat={repeat} mode={mode} output_limit={limit}", flush=True)
        with copy_control(cpu, shadow, mode == "resident_sparse", diagnostic), phases(diagnostic, model):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                completed = worker.run(requests)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        completed = sorted(completed, key=lambda req: req.request_id)
        generated = [list(req.token_ids[req.len_input:]) for req in completed]
        if len(generated) != len(inputs) or any(not tokens for tokens in generated):
            raise AssertionError("Every selected AIME request must produce an output")
        if mode == "original_sparse":
            worker.close_acceptance_log()
            worker.step_trace.close(True)
            metrics = dict(forwards=worker.prof._step,
                accepted_draft_tokens=wandb_logger.log_data.get("spec/accepted_speculative_tokens", 0),
                proposed_draft_tokens=wandb_logger.log_data.get("spec/total_speculative_tokens", 0))
            assert len(original_pool._free_gpu) == original_pool.capacity
        else:
            metrics = dict(worker.metrics, query_phases=dict(worker.phases),
                actual_batch_histogram=dict(worker.batch_histogram),
                selection_history=describe([row["history"] for row in worker.selections]),
                selection_hot=describe([row["hot"] for row in worker.selections]),
                effective_hot_ratio=describe([row["hot"] / row["history"] for row in worker.selections]))
            assert not gpu.stats()["in_use_bytes"] and not cpu.stats()["in_use_bytes"]
        record = dict(mode=mode, stage=stage, repeat=repeat, seconds=elapsed,
            output_lengths=[len(tokens) for tokens in generated], output_tokens=sum(map(len, generated)),
            tokens_per_second=sum(map(len, generated)) / elapsed,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), metrics=metrics,
            output_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest())
        if stage != "smoke" and mode in ("resident_sparse", "packed_offload"):
            signature = (generated, worker.round_rows, metrics["forwards"], metrics["migrations"])
            if limit not in reference:
                reference[limit] = signature
            elif reference[limit] != signature:
                raise AssertionError("Resident/offload outputs or verify trajectories differ; timings not paired")
            record["resident_offload_exact_match"] = True
        save(output / f"{stage}-{repeat}-{mode}.json", dict(record=record, outputs=generated,
            rounds=getattr(worker, "round_rows", []), selections=getattr(worker, "selections", [])))
        print("DONE", json.dumps(record), flush=True)
        if diagnostic is not None:
            save(output / f"phases-{mode}.json", diagnostic.resolve())
        return record

    modes = args.modes
    try:
        for mode in modes:
            trial(mode, "smoke", 0, min(16, args.output_length))
        # Full-workload warmup also compiles shape-specialized selection/gather
        # kernels. Warmups are retained separately, never mixed into statistics.
        for mode in modes:
            warmups.append(trial(mode, "warmup", 0, args.output_length))
        rng = random.Random(args.seed + 1)
        for repeat in range(args.repetitions):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                results.append(trial(mode, "measured", repeat, args.output_length))
                save(output / "runs.json", dict(warmups=warmups, measured=results))
        summary = {mode: dict(seconds=describe([row["seconds"] for row in results if row["mode"] == mode]),
            tokens_per_second=describe([row["tokens_per_second"] for row in results if row["mode"] == mode]))
            for mode in modes}
        for mode in ("resident_sparse", "packed_offload"):
            if mode not in modes:
                continue
            diagnostic = Events(args.diagnostic_rounds, args.diagnostic_min_history)
            measured = trial(mode, "diagnostic", 0, args.output_length, diagnostic)
            measured["vs_uninstrumented_median_percent"] = 100 * (
                measured["seconds"] / summary[mode]["seconds"]["median"] - 1)
            save(output / f"diagnostic-overhead-{mode}.json", measured)
        medians = {mode: summary[mode]["seconds"]["median"] for mode in modes}
        comparison = {}
        if "packed_offload" in medians:
            for baseline, label in (("resident_sparse", "resident"), ("original_sparse", "original"),
                                    ("full_self_spec", "full_self_spec")):
                if baseline in medians:
                    comparison[f"offload_vs_{label}_time_percent"] = 100 * (medians["packed_offload"] / medians[baseline] - 1)
            if "resident_sparse" in medians:
                comparison["offload_minus_resident_seconds"] = medians["packed_offload"] - medians["resident_sparse"]
        summary["comparison"] = comparison
        save(output / "summary.json", summary)
        print("BENCHMARK_PASSED", json.dumps(summary), flush=True)
    finally:
        cpu.close(wait=True)
        gpu.close(wait=True)


if __name__ == "__main__":
    main()
