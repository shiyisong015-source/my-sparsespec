"""Same-shape native microbench + quiet AIME pilot for native compact KV.

Graphs are used ONLY to amortize microbenchmark submissions. Serving arms all
use eager synchronous execution. A test-only resident-cold control retains the
identical compact schedule/math while replacing mapped CPU transfers by D2D.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import torch
import tqdm as tqdm_module
from transformers import AutoTokenizer

from serve.attention.kv_pack import _pool_copy, gather_to_pool, pack_selected_kv, restore_to_pool
from serve.attention.kv_partition import partition_pages
from serve.attention.backend import BatchTopKWrapper
from serve.distribute.tp import setup_tp_worker
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import KVPool, Transformer
from serve.request.kv_cache_ptr import compact as compact_module
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request
from serve.scheduler.compact_scheduler import CompactSpecScheduler
from serve.scheduler.spec_scheduler import SpecScheduler
from serve.tests.validate_compact_native import error_metrics
from serve.utils import MetricBucket, setup_seed, wandb_logger


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def diagnostic(model, pool, manifest, output, limit):
    """Heavy independent correctness run. Never included in performance data."""
    reference = None
    comparisons = []
    for label, resident in (("resident_a", True), ("resident_b", True), ("offload", False)):
        setup_seed(330)
        assert pool.num_free_pages == pool.capacity
        pool._free_gpu, pool.phase_idx = set(range(pool.capacity)), 0
        model.attach_kv_pool(pool)
        trace, repeats, moves = dict(forward=[], selection=[]), [], []
        original_topk = BatchTopKWrapper.run

        def topk(wrapper, *args, **kwargs):
            result = original_topk(wrapper, *args, **kwargs)
            starts = kwargs["in_val_start_pos"].tolist()
            lengths, ks = kwargs["in_lens"].tolist(), kwargs["ks"].tolist()
            values = torch.cat([kwargs["in_val"][..., s:s+n].to(torch.float16).cpu().flatten()
                                for s, n in zip(starts, lengths)])
            selected = torch.cat([result[1][i, ..., :k].cpu().flatten() for i, k in enumerate(ks)])
            ordered_sets = torch.cat([result[1][i, ..., :k].sort(-1).values.cpu().flatten()
                                      for i, k in enumerate(ks)])
            trace["selection"].append(dict(lengths=lengths, ks=ks, scores=tensor_digest(values),
                selected=tensor_digest(selected), selected_sets=tensor_digest(ordered_sets)))
            if label == "resident_a" and len(repeats) < 2 and any(n > k > 0 for n, k in zip(lengths, ks)):
                ordered, sets = [], []
                for _ in range(16):
                    again = original_topk(wrapper, *args, **kwargs)[1]
                    ordered.append(tensor_digest(torch.cat([again[i, ..., :k].cpu().flatten() for i,k in enumerate(ks)])))
                    sets.append(tensor_digest(torch.cat([again[i, ..., :k].sort(-1).values.cpu().flatten() for i,k in enumerate(ks)])))
                repeats.append(dict(selection_call=len(trace["selection"])-1, lengths=lengths, ks=ks,
                    repeated_calls=16, distinct_ordered_outputs=len(set(ordered)), distinct_selected_sets=len(set(sets))))
                save(output / "frozen-topk.json", repeats)
            return result

        class DiagnosticModel(CUDAGraphModelWrapper):
            def __call__(self, **kwargs):
                logits = super().__call__(**kwargs)
                cpu = logits.cpu()
                trace["forward"].append(dict(input=kwargs["token_ids"].tolist(),
                    request_types=kwargs["request_type"].tolist(), logits=tensor_digest(cpu),
                    argmax=cpu.argmax(-1).tolist(), top2=cpu.float().topk(2).values.tolist()))
                return logits

        def gather_expected(source, selection, count):
            index = selection[..., :count, None, None, None].long().expand(
                source.shape[0], source.shape[4], count, 2, 1, source.shape[5])
            return torch.gather(source.permute(0,4,1,2,3,5), 2, index).permute(0,2,3,4,1,5).contiguous()

        with resident_control(resident):
            original_gather = compact_module.gather_to_pool
            original_pack = compact_module.pack_selected_kv
            original_restore = compact_module.restore_to_pool
            counters = dict(hot=0, cold=0, restore=0)

            def gather(source, selection, pages, **kwargs):
                expected = gather_expected(source, selection, len(pages)) if counters["hot"] < 8 else None
                result = original_gather(source, selection, pages, **kwargs)
                if expected is not None:
                    torch.testing.assert_close(source[:, list(pages)], expected, rtol=0, atol=0)
                    moves.append(dict(kind="hot", count=len(pages), bitwise=True))
                counters["hot"] += 1
                return result

            def pack(source, selection, destination, count):
                expected = gather_expected(source, selection, count).cpu() if counters["cold"] < 8 else None
                result = original_pack(source, selection, destination, count)
                if expected is not None:
                    torch.cuda.synchronize()
                    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
                    moves.append(dict(kind="cold", count=count, bitwise=True))
                counters["cold"] += 1
                return result

            def restore(source, destination, pages):
                expected = source.cpu().clone() if counters["restore"] < 8 else None
                result = original_restore(source, destination, pages)
                if expected is not None:
                    torch.cuda.synchronize()
                    torch.testing.assert_close(destination[:, list(pages)].cpu(), expected, rtol=0, atol=0)
                    moves.append(dict(kind="restore", count=len(pages), bitwise=True))
                counters["restore"] += 1
                return result

            with patch.object(BatchTopKWrapper, "run", topk), \
                 patch.object(compact_module, "gather_to_pool", gather), \
                 patch.object(compact_module, "pack_selected_kv", pack), \
                 patch.object(compact_module, "restore_to_pool", restore):
                worker = CompactSpecScheduler(pool, DiagnosticModel(model), spec_stride=4, budget_ratio=.1,
                    num_min_budget=128, max_batch_size=4, eos_token_id=manifest["eos_token_ids"],
                    admit_policy="offloading", async_cpu=False, tp_ranks=1, temperature=0.,
                    enable_torch_profiler=False, profiler_result_dir=None, batch_policy="greedy",
                    enable_cuda_graph=False, chunk_size=256, trace_name="diagnostic", kv_cache="compact",
                    compact_cpu_pages=2*pool.capacity)
                completed = worker.run([Request(row["input_ids"], limit) for row in manifest["requests"]])
                worker.close_acceptance_log()
        trace["moves"] = moves
        trace["outputs"] = [list(r.token_ids[r.len_input:]) for r in completed]
        save(output / f"diagnostic-{label}.json", trace)
        if reference is None:
            reference = trace
        else:
            first = {}
            for group, fields in (("forward", ("input", "logits", "argmax")),
                                  ("selection", ("scores", "selected", "selected_sets"))):
                for field in fields:
                    first[f"{group}_{field}"] = next((i for i,(a,b) in enumerate(zip(reference[group],trace[group]))
                                                       if a[field] != b[field]), None)
            comparisons.append(dict(label=label, first_difference=first,
                outputs_equal=reference["outputs"] == trace["outputs"], all_audited_moves_bitwise=True))
            save(output / "diagnostic-summary.json", comparisons)
        print("DIAGNOSTIC", label, len(trace["forward"]), len(trace["selection"]), flush=True)


class SelectionTrace:
    """TEST ONLY: retain valid native selections, while still running TopK.

    RAFT's unsorted atomic output is not deterministic. Preloaded GPU tensors
    fix the choices/order in paired storage trials. The production path is
    unchanged. Host metadata signatures are checked without per-step readback.
    """
    def __init__(self):
        self.records = []

    @contextmanager
    def use(self, *, capture):
        cursor, signature = 0, None
        original_run = BatchTopKWrapper.run
        original_dispatch = PillarCachePtr.dispatch_update_selected_indices.__func__

        def dispatch(cls, wrappers, selected_indices_metadatas, spec_cache_args, input_metadata, device):
            nonlocal signature
            tasks = []
            if input_metadata is not None:
                for metadata in selected_indices_metadatas:
                    prev = metadata["prev_idx"]
                    if prev >= 0 and input_metadata["request_type"][prev] == Request.ReqExecType.VERIFY.value:
                        tasks.append((int(metadata["idx"]), int(prev), int(metadata["num_effective_kv_len"]),
                                      int(metadata["num_selected_tokens"])))
            signature = (tuple(tasks), tuple(spec_cache_args["all_kv_indptr_cpu"]))
            return original_dispatch(cls, wrappers, selected_indices_metadatas, spec_cache_args, input_metadata, device)

        def run(wrapper, *args, **kwargs):
            nonlocal cursor
            if not capture:
                if cursor >= len(self.records) or signature != self.records[cursor]["signature"]:
                    raise RuntimeError(f"Selection trace metadata diverged at call {cursor}")
            result = original_run(wrapper, *args, **kwargs)
            if capture:
                counts = [task[-1] for task in signature[0]]
                self.records.append(dict(signature=signature,
                    selections=[result[1][i, ..., :k].clone() for i,k in enumerate(counts)]))
            else:
                for i, value in enumerate(self.records[cursor]["selections"]):
                    result[1][i, ..., :value.shape[-1]].copy_(value)
            cursor += 1
            return result

        with patch.object(PillarCachePtr, "dispatch_update_selected_indices", classmethod(dispatch)), \
             patch.object(BatchTopKWrapper, "run", run):
            yield
        if not capture and cursor != len(self.records):
            raise RuntimeError(f"Selection trace consumed {cursor}/{len(self.records)} calls")

    def save_and_measure(self, output):
        torch.save([dict(signature=r["signature"], selections=[x.cpu() for x in r["selections"]])
                    for r in self.records], output / "native-selection-trace.pt")
        record = next(r for r in self.records if r["selections"] and r["selections"][0].shape[-1] > 0)
        destination = [torch.empty((*x.shape[:-1], 6144), dtype=x.dtype, device=x.device)
                       for x in record["selections"]]
        def copy():
            for dst, src in zip(destination, record["selections"]):
                dst[..., :src.shape[-1]].copy_(src)
        save(output / "selection-replay-overhead.json", dict(
            calls=len(self.records), stored_bytes=sum(x.numel()*x.element_size() for r in self.records for x in r["selections"]),
            sample_counts=[x.shape[-1] for x in record["selections"]], copy=measure(copy, unroll=32),
            scope="Representative D2D trace-copy API only; metadata signature host checks also remain in measured trials. Do not subtract this as an exact end-to-end correction."))


@contextmanager
def resident_control(enabled):
    if not enabled:
        yield
        return
    original_empty = torch.empty
    def empty(*args, **kwargs):
        shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        if len(shape) == 6 and kwargs.get("device") == "cpu" and kwargs.get("pin_memory"):
            kwargs = dict(kwargs, device="cuda:0", pin_memory=False)
        return original_empty(*args, **kwargs)
    def restore(source, pool, pages):
        selection = torch.arange(len(pages), device=pool.device, dtype=torch.int32)
        selection = selection.expand(source.shape[0], source.shape[4], len(pages))
        _pool_copy(source, pool, pages, selection)
    with patch.object(compact_module.torch, "empty", empty), patch.object(compact_module, "restore_to_pool", restore):
        yield


def measure(call, *, unroll, repeats=8, capture=True):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    started = time.perf_counter()
    for _ in range(8):
        call()
    cpu_ms = (time.perf_counter() - started) * 1000 / 8
    end.record()
    end.synchronize()
    eager_ms = begin.elapsed_time(end) / 8
    result = dict(host_call_wall_ms=cpu_ms, eager_event_ms_including_submission_gaps=eager_ms)
    if not capture:
        return result
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            call()
    graph.replay()
    torch.cuda.synchronize()
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    replay_ms = begin.elapsed_time(end) / (repeats * unroll)
    return dict(**result, graph_gpu_ms=replay_ms, graph_unroll=unroll, graph_replays=repeats)


def transfer_microbench(pool, output):
    """Actual public copy APIs, including their metadata setup and allocation.

    Eager events include CPU submission gaps. Mapped CPU buffers stay alive
    until the final event. These are separate diagnostics, outside AIME timing.
    """
    source = pool._physical_mem_gpu
    layers, _, _, _, heads, _ = source.shape
    length, hot, recent = 1024, 128, 4
    full = torch.arange(length, device=source.device, dtype=torch.int32)
    selected = torch.stack([torch.stack([
        torch.randperm(length-recent, device=source.device)[:hot-recent]
        for _ in range(heads)]) for _ in range(layers)]).to(torch.int32)
    partition = lambda: partition_pages(full, selected, hot-recent, pool_capacity=pool.capacity,
                                        recent_count=recent)
    plan = partition()
    hot_pages, restored_pages = tuple(range(length, length+hot)), tuple(range(length+hot, 2*length))
    cold = torch.empty((layers, length-hot, *source.shape[2:]), dtype=source.dtype,
                       device="cpu", pin_memory=True)
    gather = lambda: gather_to_pool(source, plan.hot_pages, hot_pages, source_page_ids=tuple(range(length)))
    offload = lambda: pack_selected_kv(source, plan.cold_pages, cold, length-hot)
    offload()
    torch.cuda.synchronize()
    restore = lambda: restore_to_pool(cold, source, restored_pages)

    def pack_boundary():
        current = partition()
        gather_to_pool(source, current.hot_pages, hot_pages, source_page_ids=tuple(range(length)))
        pack_selected_kv(source, current.cold_pages, cold, length-hot)

    rows = dict(logical_length=length, hot_count=hot, cold_count=length-hot,
                layers=layers, local_heads=heads, pool_capacity=pool.capacity,
                cold_bytes=cold.numel()*cold.element_size(),
                timing="host call wall includes allocation/implicit waits; eager API events include submission gaps; timings overlap and must not be added")
    for name, call in (("partition_complement", partition), ("hot_gather", gather),
                       ("cold_d2h", offload), ("cold_h2d", restore), ("pack_boundary", pack_boundary)):
        rows[name] = measure(call, unroll=1, capture=False)
    save(output / "transfer-microbench.json", rows)


def microbench(model, pools, output):
    cfg, device = model.config, model.device
    batch, hot = 4, 128
    cases = (("prefill", 256, 32, 0), ("draft", 1024, 1, 1), ("verify", 1024, 5, 2))
    rows = []
    for phase, length, query, request_type in cases:
        original, compact = pools["native"], pools["compact"]
        original._physical_mem_gpu.normal_(0, .1)
        visible = hot if phase == "draft" else length
        selections = []
        for r in range(batch):
            # Each layer/head uses a different prefix permutation; suffix order
            # and the visible KV multiset are identical in the two arms.
            selection = torch.stack([torch.stack([
                torch.cat([torch.randperm(length-query, device=device)[:visible-query],
                           torch.arange(length-query, length, device=device)]) + r * length
                for _ in range(cfg.num_kv_heads)]) for _ in range(cfg.num_layers)])
            if phase == "prefill":
                selection = torch.arange(r*length, (r+1)*length, device=device).expand(cfg.num_layers, cfg.num_kv_heads, length)
            selections.append(selection)
            dest = list(range(r * visible, (r + 1) * visible))
            for layer in range(cfg.num_layers):
                for head in range(cfg.num_kv_heads):
                    compact._physical_mem_gpu[layer, dest, :, :, head] = original._physical_mem_gpu[layer, selection[layer, head], :, :, head]
                    if phase == "verify":
                        # The performance pair uses identical reduction order.
                        # Prefix permutation sensitivity is a separate stress test.
                        original._physical_mem_gpu[layer, list(range(r*length, (r+1)*length)), :, :, head] = compact._physical_mem_gpu[layer, dest, :, :, head]
        tokens = torch.randint(10, cfg.vocab_size, (batch * query,), device=device, dtype=torch.int32)
        q = torch.randn(batch * query, cfg.num_qo_heads, cfg.head_dim, device=device, dtype=torch.bfloat16)
        logits = {}
        for arm, pool in pools.items():
            cls = pool.kv_cache_class
            n = length if arm == "native" else visible
            pages = torch.arange(batch*n, device=device, dtype=torch.int32)
            flatten = cls.draft_kv_indices_tensor_buf
            if phase == "draft":
                for r in range(batch):
                    flatten[..., r*visible:(r+1)*visible] = (selections[r] if arm == "native" else
                        torch.arange(r*visible, (r+1)*visible, device=device, dtype=torch.int32))
            make = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
            metadata = dict(token_ids=tokens, token_indptr=make([i*query for i in range(batch+1)]),
                pos_offsets=make([length-query]*batch), kv_indices=pages,
                all_kv_indptr=make([i*n for i in range(batch+1)]), all_kv_len_arr=make([n]*batch),
                kv_len_arr=make([visible]*batch), kv_indptr=make([i*visible for i in range(batch+1)]),
                request_type=make([request_type]*batch), verify_kv_indptr=make([i*visible for i in range(batch)]),
                flatten_indices=flatten, flatten_indices_len=batch*visible if phase=="draft" else 0,
                dump_logits_len=batch*visible if phase=="verify" else 0)
            model.attach_kv_pool(pool)
            wrapped = CUDAGraphModelWrapper(model)
            wrapped.register_input(**metadata)
            logits[arm] = wrapped(**metadata).clone()
            entry = dict(arm=arm, phase=phase, batch=batch, query=query, logical_length=length,
                         visible_length=visible, pool_capacity=pool.capacity,
                         forward=measure(lambda: wrapped(**metadata), unroll=2))
            backend, meta = pool.wrappers[0], pool.attn_metadata
            def attention():
                return backend.run_attention(q, pool._physical_mem_gpu[0], meta["kv_indices"],
                                             meta["flatten_indices"][0], meta["request_type"])
            attn_out, lse = attention()
            # Explicit float32 causal oracle, independent of native split/reduce.
            expected = torch.empty_like(attn_out, dtype=torch.float32)
            gqa = cfg.num_qo_heads // cfg.num_kv_heads
            for r in range(batch):
                for h in range(cfg.num_kv_heads):
                    ids = (flatten[0, h, r*visible:(r+1)*visible].long() if phase=="draft"
                           else pages[r*n:(r+1)*n].long())
                    key = pool._physical_mem_gpu[0, ids, 0, 0, h].float()
                    value = pool._physical_mem_gpu[0, ids, 1, 0, h].float()
                    queries = q[r*query:(r+1)*query, h*gqa:(h+1)*gqa].float().permute(1, 0, 2)
                    scores = queries @ key.T / cfg.head_dim**.5
                    mask = torch.arange(visible, device=device)[None, :] > visible-query + torch.arange(query, device=device)[:, None]
                    scores.masked_fill_(mask, -float("inf"))
                    expected[r*query:(r+1)*query, h*gqa:(h+1)*gqa] = (scores.softmax(-1) @ value).permute(1, 0, 2)
            torch.testing.assert_close(attn_out.float(), expected, atol=.0002, rtol=.015)
            entry["causal_oracle_max_abs"] = float((attn_out.float()-expected).abs().max())
            entry["attention_including_internal_reduction"] = measure(attention, unroll=32)
            entry["scores"] = measure(lambda: backend.rematerialize_scores(q, attn_out, lse,
                pool._physical_mem_gpu[0], meta["kv_indices"], meta["request_type"],
                meta["verify_kv_indptr"], meta["dump_logits"][0]), unroll=32)
            if phase == "verify":
                # Strong causal oracle: zero Q/K give exactly uniform visible
                # probabilities; large suffix values make future leakage obvious.
                pool._physical_mem_gpu[0].zero_()
                sentinel = (1000 * torch.arange(1, query+1, device=device)).to(torch.bfloat16)
                for r in range(batch):
                    pool._physical_mem_gpu[0, r*n+n-query:r*n+n, 1] = sentinel[:, None, None, None]
                sentinel_out, _ = backend.run_attention(torch.zeros_like(q), pool._physical_mem_gpu[0],
                    meta["kv_indices"], meta["flatten_indices"][0], meta["request_type"])
                analytic = sentinel.float().cumsum(0) / torch.arange(n-query+1, n+1, device=device)
                analytic = analytic.repeat(batch)[:, None, None].expand_as(sentinel_out)
                torch.testing.assert_close(sentinel_out.float(), analytic, atol=.0002, rtol=.007)
                unmasked = torch.full_like(analytic, sentinel.float().sum()/n)
                assert not torch.allclose(sentinel_out.float(), unmasked, atol=.0002, rtol=.007)
                entry["causal_sentinel"] = dict(max_abs=float((sentinel_out.float()-analytic).abs().max()),
                    unmasked_negative_control_error=float((sentinel_out.float()-unmasked).abs().max()),
                    tolerance=dict(atol=.0002, rtol=.007), negative_control_rejected=True)
            rows.append(entry)
            save(output / "microbench.json", rows)
        comparison = error_metrics(logits["compact"].float(), logits["native"].float())
        # The tolerance is reported for every row, not silently changed.
        rows[-1]["same_shape_logits"] = comparison
        assert comparison["passed"], (phase, comparison)
        save(output / "microbench.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="log/kv-offload-improvement-20261014/aime-final/manifest.json")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--diagnose", action="store_true", help="Heavy repeatability/copy audits; no timing")
    parser.add_argument("--fixed-selection", action="store_true",
                        help="Test-only valid native selection trace for identical compact resident/offload work")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    setup_seed(330)
    model = Transformer.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    model = setup_tp_worker(model, 1, "cuda:0", torch.bfloat16, "naive")
    cfg, capacity = model.config, 4096
    spec = dict(spec_stride=4, budget_ratio=.1, num_min_budget=128, max_batch_size=4)
    pools = {}
    for arm, cls in (("native", PillarCachePtr), ("compact", CompactCachePtr)):
        # A subclass inherits _INITIALIZED from Pillar; set its own counter.
        cls._INITIALIZED = 0
        cls.initialize_class_metadata(dtype=torch.bfloat16, head_dim=cfg.head_dim,
            num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, num_layers=cfg.num_layers,
            capacity=capacity, device="cuda:0", max_batch_size=4, spec_cache_args=spec)
        pools[arm] = KVPool(cfg, capacity, "cuda:0", cls, dtype=torch.bfloat16, allocate_legacy_cpu=False)
    with torch.inference_mode():
        if args.diagnose:
            diagnostic(model, pools["compact"], json.loads(Path(args.manifest).read_text()), output, args.output_tokens)
            return
        if not args.skip_micro:
            microbench(model, pools, output)
            transfer_microbench(pools["compact"], output)
        if args.micro_only:
            return
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            rows = [json.loads(row) for row in Path("eval/benchmarks/data/aime24/test.jsonl").read_text().splitlines() if row]
            selected = random.Random(330).sample(rows, 8)
            manifest = dict(eos_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                requests=[dict(id=row["id"], input_ids=tokenizer.apply_chat_template([dict(role="user", content=
                    row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.")],
                    tokenize=True, add_generation_prompt=True, enable_thinking=True)) for row in selected])
        inputs = [row["input_ids"] for row in manifest["requests"]]
        manifest.update(native_compact_config=vars(args), temperature=0, thinking=True, graph=False,
            async_cpu=False, gpu_pool_pages=capacity, cpu_budget_pages=2*capacity,
            timing="quiet scheduler.run + final synchronize, excluding setup/warmup; micrographs are separate",
            resident_control="Test-only GPU cold backing + D2D scatter; extra GPU storage is not production serving",
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        manifest["selection_protocol"] = ("Native baseline free-running; compact pair executes native TopK then copies the same preloaded valid-selection trace; extra copy/check overhead is included" if args.fixed_selection else "Unmodified free-running native TopK; outputs may vary")
        save(output / "manifest.json", manifest)
        (output / "bench_source.py").write_bytes(Path(__file__).read_bytes())
        results = []
        selection_trace = SelectionTrace()
        common = dict(**spec, eos_token_id=manifest["eos_token_ids"], admit_policy="offloading", async_cpu=False,
            tp_ranks=1, temperature=0., enable_torch_profiler=False, profiler_result_dir=None,
            batch_policy="greedy", enable_cuda_graph=False, chunk_size=256, trace_name="aime-native-compact")
        def trial(mode, limit, stage, repetition, trace_mode=None):
            setup_seed(330)
            pool = pools["native" if mode == "native" else "compact"]
            assert pool.num_free_pages == pool.capacity
            pool._free_gpu = set(range(pool.capacity))  # reproducible test-only free-set cursor
            pool.phase_idx = 0
            model.attach_kv_pool(pool)
            wandb_logger.log_data = MetricBucket()
            cls = SpecScheduler if mode == "native" else CompactSpecScheduler
            worker = cls(pool, CUDAGraphModelWrapper(model), kv_cache="pillar" if mode=="native" else "compact",
                         compact_cpu_pages=2*capacity, **common)
            print(f"START {stage} {repetition} {mode}", flush=True)
            requests = [Request(ids, limit) for ids in inputs]
            original_tqdm = tqdm_module.tqdm
            quiet_tqdm = lambda *a, **kw: original_tqdm(*a, **dict(kw, disable=True))
            trace_context = selection_trace.use(capture=trace_mode == "capture") if trace_mode else nullcontext()
            with resident_control(mode == "compact_resident"), trace_context, \
                 patch.object(tqdm_module, "tqdm", quiet_tqdm):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                completed = worker.run(requests)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            generated = [list(r.token_ids[r.len_input:]) for r in sorted(completed, key=lambda r:r.request_id)]
            tokens = sum(map(len, generated))
            result = dict(mode=mode, stage=stage, repetition=repetition, seconds=elapsed, actual_output_tokens=tokens,
                tokens_per_second=tokens/elapsed, output_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), metrics=getattr(worker, "metrics", None),
                accepted=int(wandb_logger.log_data.get("spec/accepted_speculative_tokens", 0)),
                proposed=int(wandb_logger.log_data.get("spec/total_speculative_tokens", 0)))
            result["selection_trace"] = trace_mode
            worker.close_acceptance_log()
            results.append(result)
            save(output / "trials.json", results)
            print(json.dumps(result), flush=True)
            return result
        def workload(row):
            return (row["output_sha256"], row["accepted"], row["proposed"], row["actual_output_tokens"],
                    tuple(row["metrics"][key] for key in ("forwards", "request_verify_rounds", "migrations",
                          "restores", "batch_sum", "max_actual_batch", "mixed_forwards")))
        logging.getLogger().setLevel(logging.ERROR)
        modes = ["native", "compact_resident", "compact_offload"]
        for mode in modes:
            trial(mode, 32, "warmup", -1)
        if args.fixed_selection:
            reference = trial("compact_resident", args.output_tokens, "trace_capture", -1, "capture")
            selection_trace.save_and_measure(output)
            for mode in modes[1:]:
                preflight = trial(mode, args.output_tokens, "trace_preflight", -1, "replay")
                assert workload(preflight) == workload(reference), (mode, "Fixed selection preflight diverged")
            save(output / "trace-preflight.json", dict(passed=True,
                fields="output hash, accepted/proposed/actual output counts, forwards, verify/migration/restore counts, batch sum/max/mixed forwards"))
        for repetition in range(args.repetitions):
            order = modes[:]
            random.Random(330+repetition).shuffle(order)
            for mode in order:
                result = trial(mode, args.output_tokens, "measured", repetition,
                               "replay" if args.fixed_selection and mode != "native" else None)
                if args.fixed_selection and mode != "native":
                    assert workload(result) == workload(reference), (mode, "Fixed selection measured workload diverged")
        measured = [row for row in results if row["stage"] == "measured"]
        compact_rows = [r for r in measured if r["mode"] != "native"]
        if args.fixed_selection:
            assert len({workload(r) for r in compact_rows}) == 1
        summary = {mode: dict(mean_seconds=statistics.mean(r["seconds"] for r in measured if r["mode"]==mode),
                             mean_tokens_per_second=statistics.mean(r["tokens_per_second"] for r in measured if r["mode"]==mode)) for mode in modes}
        save(output / "summary.json", summary)


if __name__ == "__main__":
    main()
