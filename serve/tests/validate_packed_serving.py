"""Qwen3 integration validation; run directly or through torchrun for TP2.

Uses real model weights, compares greedy outputs against dense AR, verifies
actual cold DMA and shared-arena reuse, then exercises stochastic sampling.
Artifacts stay below SparseSpec/log; this is not a throughput benchmark.
"""

import argparse
import hashlib
import json
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from serve.attention.kv_buffer_pool import KVBufferPool
from serve.distribute.tp import end_tp_worker, setup_tp_worker
from serve.model.model import Transformer
from serve.request.request import Request
from serve.scheduler.packed_scheduler import PackedSpecScheduler, coordinated_call
from serve.utils import get_local_rank, setup_seed


class _LogitAuditScheduler(PackedSpecScheduler):
    """Keep only target prediction rows for numerical diagnostics, on rank 0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_logits = {}
        self._audit_entries = ()

    def _forward(self, entries):
        self._audit_entries = entries
        return super()._forward(entries)

    def _sample(self, logits):
        result = super()._sample(logits)
        if self.rank == 0 and self.metadata.is_greedy_sampling:
            # Model TP logits are physically [rank, token, vocab_shard].
            physical = logits.view(self.tp_ranks, logits.shape[0], -1)
            offset = 0
            for state, _, ids, phase in self._audit_entries:
                rows = []
                if phase == "verify":
                    rows = [(offset + j, len(state.output) + j) for j in range(len(ids))]
                elif phase == "ar":
                    rows = [(offset, len(state.output))]
                elif phase == "prefill" and state.region.length == state.request.len_input:
                    rows = [(offset + len(ids) - 1, 0)]
                for row, position in rows:
                    self.target_logits[state.request.request_id, position] = (
                        physical[:, row].contiguous().view(-1).float().cpu())
                offset += len(ids)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-entry", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"cuda:{get_local_rank()}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def prompt(text):
        return tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False)
    normal = [prompt("Write a detailed numbered explanation of binary search, including its loop invariant and edge cases."),
              prompt("Explain how a CPU cache works in ten paragraphs. " + "Discuss locality and memory layout. " * 35),
              prompt("Write Python code for a thread-safe bounded queue, then explain its correctness. "
                     + "Consider producer and consumer behavior, waits and notifications. " * 60)]
    if args.public_entry:
        from serve.run import run_inference

        result = run_inference(
            [Request(ids, 24) for ids in normal[:2]]
            + [Request(prompt("Reply with exactly OK and nothing else."), 64), Request(normal[0], 1)],
            args.model, kv_cache="packed",
            max_batch_size=2, tp_ranks=world, reduce_method="naive", chunk_size=32,
            spec_stride=4, budget_ratio=.1, num_min_budget=16, temperature=.65,
            packed_gpu_pool_mib=512, packed_cpu_pool_mib=1024,
            result_dir=args.output, seed=330,
        )
        assert len(result) == 4
        assert result[-1]["len_output"] == 1
        assert result[-2]["len_output"] < 64 and result[-2]["output_token_ids"][-1] == tokenizer.eos_token_id
        if rank == 0:
            summary = json.load(open(os.path.join(args.output, "summary.json")))
            assert summary["migrations"] > 0 and summary["restores"] > 0
            assert summary["gpu"]["in_use_bytes"] == summary["cpu"]["in_use_bytes"] == 0
            print("PUBLIC_ENTRY_PASSED", flush=True)
        end_tp_worker(world)
        return

    started = time.perf_counter()
    model = Transformer.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    model = setup_tp_worker(model, world, str(device), torch.bfloat16, "naive")
    control = dist.new_group(backend="gloo") if world > 1 else None
    print(f"rank={rank} loaded real Qwen3-8B TP={world} in {time.perf_counter()-started:.1f}s", flush=True)
    results = []
    audits = {}
    generations = {}

    def trial(name, inputs, output_length=32, *, dense=False, k=4, r=.1, temp=0., gpu_bytes=None,
              cpu_bytes=1024 * 1024**2, chunk_size=32, full_spec=False):
        setup_seed(330)
        gpu = coordinated_call(control, "test GPU arena", lambda: KVBufferPool(
            gpu_bytes or 512 * 1024**2, device=device), lambda pool: pool.close())
        cpu = coordinated_call(control, "test CPU arena", lambda: KVBufferPool(cpu_bytes),
                               lambda pool: pool.close())
        try:
            scheduler = _LogitAuditScheduler(
                model, gpu, cpu, max_batch_size=4, chunk_size=chunk_size, spec_stride=k,
                budget_ratio=r, num_min_budget=16, temperature=temp, eos_token_id=(),
                tp_ranks=world, control_group=control, result_dir=os.path.join(args.output, name),
                dense_reference=dense, retain_full_reference=full_spec,
            )
            start = time.perf_counter()
            with torch.inference_mode():
                completed = scheduler.run([Request(ids, output_length) for ids in inputs])
            outputs = [list(req.token_ids[req.len_input:]) for req in completed]
            assert all(len(ids) == output_length for ids in outputs)
            stats = dict(name=name, k=k, r=r, temperature=temp,
                         seconds=time.perf_counter() - start,
                         input_lengths=[len(ids) for ids in inputs],
                         output_lengths=[len(ids) for ids in outputs], **scheduler.metrics)
            if not dense and not full_spec:
                assert stats["migrations"] > 0 and stats["restores"] > 0
                assert stats["d2h_bytes"] > 0 and stats["h2d_bytes"] > 0
                assert stats["full_bytes_returned"] > 0
            if control is not None:
                digest = hashlib.sha256(json.dumps(outputs).encode()).hexdigest()
                digests = [None] * world
                dist.all_gather_object(digests, digest, group=control)
                assert len(set(digests)) == 1, digests
            if rank == 0:
                json.dump(dict(stats=stats, outputs=outputs),
                          open(os.path.join(args.output, name, "result.json"), "w"), indent=2)
            print(f"rank={rank} completed_inference {name}: {stats}", flush=True)
            results.append(stats)
            audits[name] = scheduler.target_logits
            generations[name] = outputs
            return outputs, stats
        finally:
            cpu.close(wait=True)
            gpu.close(wait=True)

    def check_greedy(name, outputs, expected, stats, *, dense_name, full_name=None):
        diagnostics = None
        if outputs != expected:
            differences = [(i, next(j for j, (a, b) in enumerate(zip(got, ref)) if a != b))
                           for i, (got, ref) in enumerate(zip(outputs, expected)) if got != ref]
            if rank == 0:
                diagnostics = []
                for request_id, position in differences:
                    dense = audits[dense_name][request_id, position]
                    packed = audits[name][request_id, position]
                    delta = packed - dense
                    top_values, top_ids = dense.topk(2)
                    row = dict(request_id=request_id, position=position,
                        dense_token=int(dense.argmax()), packed_token=int(packed.argmax()),
                        dense_top2_ids=top_ids.tolist(), dense_top2_logits=top_values.tolist(),
                        packed_logits_for_dense_top2=packed[top_ids].tolist(),
                        max_abs_logit_error=float(delta.abs().max()),
                        rmse=float(delta.square().mean().sqrt()),
                        relative_rmse=float(delta.square().mean().sqrt() / dense.square().mean().sqrt()),
                        cosine=float(torch.nn.functional.cosine_similarity(dense, packed, dim=0)))
                    if full_name is not None:
                        same_prefix = generations[full_name][request_id][:position] == expected[request_id][:position]
                        row["full_spec_same_prefix"] = same_prefix
                        if same_prefix:
                            full = audits[full_name][request_id, position]
                            row.update(full_spec_token=int(full.argmax()),
                                full_spec_logits_for_dense_top2=full[top_ids].tolist(),
                                full_spec_vs_ar_max_error=float((full - dense).abs().max()),
                                packed_vs_full_spec_max_error=float((packed - full).abs().max()))
                    diagnostics.append(row)
            if control is not None:
                holder = [diagnostics]
                dist.broadcast_object_list(holder, src=0, group=control)
                diagnostics = holder[0]
            print(f"rank={rank} GREEDY_NUMERICAL_DIAGNOSTICS {name}: {diagnostics}", flush=True)
            # Same bounds for every case. A changed argmax is recorded as
            # nonexact; no tolerance is enlarged to make a failing case pass.
            for row in diagnostics:
                assert row["max_abs_logit_error"] <= .5 and row["relative_rmse"] <= .015
                assert row["cosine"] >= .9999
                gap = row["dense_top2_logits"][0] - row["dense_top2_logits"][1]
                assert gap <= 2 * row["max_abs_logit_error"]
            stats["greedy_numerical_differences"] = diagnostics
        stats["greedy_matches_dense_ar"] = outputs == expected
        if rank == 0:
            json.dump(dict(exact=outputs == expected, diagnostics=diagnostics),
                      open(os.path.join(args.output, name, "greedy-check.json"), "w"), indent=2)

    try:
        expected, _ = trial("dense-ar", normal, dense=True)
        full_outputs, full_stats = trial("full-self-spec-k14", normal, k=14, full_spec=True)
        full_stats["greedy_matches_dense_ar"] = full_outputs == expected
        print(f"rank={rank} FULL_SELF_SPEC_MATCHES_AR={full_outputs == expected}", flush=True)
        for k, r in ((4, .05), (4, .25), (14, .1)):
            name = f"packed-k{k}-r{r}"
            outputs, stats = trial(name, normal, k=k, r=r)
            if outputs != expected:
                differences = [(i, next(j for j, (a, b) in enumerate(zip(got, ref)) if a != b))
                               for i, (got, ref) in enumerate(zip(outputs, expected)) if got != ref]
                diagnostics = None
                if rank == 0:
                    diagnostics = []
                    for request_id, position in differences:
                        dense = audits["dense-ar"][request_id, position]
                        packed = audits[name][request_id, position]
                        delta = packed - dense
                        full = audits["full-self-spec-k14"][request_id, position]
                        top_values, top_ids = dense.topk(2)
                        diagnostics.append(dict(request_id=request_id, position=position,
                            dense_token=int(dense.argmax()), packed_token=int(packed.argmax()),
                            dense_top2_ids=top_ids.tolist(), dense_top2_logits=top_values.tolist(),
                            packed_logits_for_dense_top2=packed[top_ids].tolist(),
                            max_abs_logit_error=float(delta.abs().max()),
                            rmse=float(delta.square().mean().sqrt()),
                            relative_rmse=float(delta.square().mean().sqrt() / dense.square().mean().sqrt()),
                            cosine=float(torch.nn.functional.cosine_similarity(dense, packed, dim=0)),
                            full_spec_token=int(full.argmax()),
                            full_spec_logits_for_dense_top2=full[top_ids].tolist(),
                            full_spec_vs_ar_max_error=float((full - dense).abs().max()),
                            packed_vs_full_spec_max_error=float((packed - full).abs().max())))
                if control is not None:
                    holder = [diagnostics]
                    dist.broadcast_object_list(holder, src=0, group=control)
                    diagnostics = holder[0]
                print(f"rank={rank} GREEDY_NUMERICAL_DIAGNOSTICS {name}: {diagnostics}", flush=True)
                # Keep a visible distinction between exact generation equality
                # and bounded floating-point equivalence. Never label a changed
                # greedy output as exact or dismiss it as sampling randomness.
                for row in diagnostics:
                    # The high-scoring BF16 logits here have ULP=.125; bound
                    # absolute errors and preserve the explicit nonexact status.
                    assert row["max_abs_logit_error"] <= .5 and row["relative_rmse"] <= .015
                    assert row["cosine"] >= .9999
                    gap = row["dense_top2_logits"][0] - row["dense_top2_logits"][1]
                    assert gap <= 2 * row["max_abs_logit_error"]
                stats["greedy_numerical_differences"] = diagnostics
            stats["greedy_matches_dense_ar"] = outputs == expected
            assert stats["mixed_verify_draft_forwards"] > 0, "No mixed verify/draft forward exercised"
            assert stats["mixed_verify_prefill_forwards"] > 0, "No mixed verify/prefill forward exercised"
        trial("stochastic", normal[:2], temp=.65)
        # Identical 250-token inputs isolate resource reuse from semantics.
        pressure = [normal[-1][:230] + normal[-1][-20:] for _ in range(4)]
        token_bytes = model.config.num_layers * 2 * model.config.num_kv_heads * model.config.head_dim * 2
        outputs, stats = trial("pressure", pressure, output_length=24,
                               gpu_bytes=1000 * token_bytes)
        assert stats["admission_waits"] > 0, "Workload did not reach arena admission pressure"
        rows = [json.loads(line) for line in open(os.path.join(args.output, "pressure",
                                                              f"kv-events.rank{rank}.jsonl"))]
        first_pack = next(i for i, row in enumerate(rows) if row["event"] == "pack")
        assert any(row["event"] == "admit" for row in rows[first_pack + 1:]), "No admission after full KV release"
        long_ids = prompt("Summarize the following technical notes in detail. "
                          + "A cache retains recently used data to reduce memory access latency. " * 400)
        long_expected, _ = trial("long-dense-ar", [long_ids], output_length=12, dense=True,
                                 gpu_bytes=2 * 1024**3, cpu_bytes=2 * 1024**3, chunk_size=512)
        long_outputs, long_stats = trial("long-packed", [long_ids], output_length=12,
                                        gpu_bytes=2 * 1024**3, cpu_bytes=2 * 1024**3, chunk_size=512)
        assert len(long_ids) > 4000
        assert long_outputs == long_expected, "Long-context greedy output differs from dense AR"
        long_stats["greedy_matches_dense_ar"] = True
        constrained_ids = prompt("Summarize the following technical notes in detail. "
                                 + "A cache retains recently used data to reduce memory access latency. " * 1700)
        assert 20000 < len(constrained_ids) < 22000
        # The arena fits the lifecycle peak but cannot hold two full histories.
        # Per-rank token budgets keep this boundary identical under TP1/TP2.
        constrained_gpu, constrained_cpu = 24000 * token_bytes, 40000 * token_bytes
        assert 2 * len(constrained_ids) * token_bytes > constrained_gpu
        constrained_expected, _ = trial("long-constrained-dense-ar", [constrained_ids],
            output_length=12, dense=True, gpu_bytes=constrained_gpu, cpu_bytes=constrained_cpu, chunk_size=512)
        constrained_outputs, constrained_stats = trial("long-constrained-packed", [constrained_ids],
            output_length=12, gpu_bytes=constrained_gpu, cpu_bytes=constrained_cpu, chunk_size=512)
        full_name = None
        if constrained_outputs != constrained_expected:
            full_name = "long-constrained-full-self-spec"
            _, full_stats = trial(full_name, [constrained_ids], output_length=12, full_spec=True,
                gpu_bytes=constrained_gpu, cpu_bytes=constrained_cpu, chunk_size=512)
            full_stats["greedy_matches_dense_ar"] = generations[full_name] == constrained_expected
        check_greedy("long-constrained-packed", constrained_outputs, constrained_expected, constrained_stats,
                     dense_name="long-constrained-dense-ar", full_name=full_name)
        # All-rank failure propagation before any model collective.
        def fail_one_rank():
            if rank == world - 1:
                raise MemoryError("injected arena exhaustion")
            return True
        try:
            coordinated_call(control, "injected pre-forward capacity failure", fail_one_rank)
        except RuntimeError as exc:
            assert "injected arena exhaustion" in str(exc)
        else:
            raise AssertionError("Capacity error was not propagated")
        if rank == 0:
            status = ("passed_with_numerical_difference" if any(
                row.get("greedy_numerical_differences") for row in results) else "passed")
            json.dump(dict(status=status, tp=world, tests=results,
                           coordinated_failure_passed=True),
                      open(os.path.join(args.output, "validation.json"), "w"), indent=2)
        print(f"END_TO_END_PASSED TP={world}", flush=True)
    finally:
        if control is not None:
            dist.destroy_process_group(control)
        end_tp_worker(world)


if __name__ == "__main__":
    main()
