"""Native compact integration and same-prefix dense verification (not timing).

The test-only teacher owns a separate full pool and recomputes the exact token
prefix. Production compact retains just the single original GPU page pool.
Run with torchrun for TP2; use --public for the actual run_inference entry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from serve.distribute.tp import setup_tp_worker
from serve.model.config import ModelArgs
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import KVPool, Transformer
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request
from serve.scheduler.compact_scheduler import CompactSpecScheduler
from serve.utils import setup_seed


def error_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    maximum = delta.abs().amax(dim=-1)
    relative = delta.square().mean(dim=-1).sqrt() / expected.square().mean(dim=-1).sqrt()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    top2 = expected.topk(2).values
    changed = actual.argmax(dim=-1) != expected.argmax(dim=-1)
    gaps = (top2[:, 0] - top2[:, 1])[changed]
    return dict(max_abs=float(maximum.max()), max_relative_rmse=float(relative.max()),
                min_cosine=float(cosine.min()), changed_argmax=int(changed.sum()),
                changed_gaps=gaps.tolist(), changed_within_bound=bool(torch.all(gaps <= 2 * maximum[changed])),
                passed=bool(torch.all(maximum <= .5) and torch.all(relative <= .015)
                            and torch.all(cosine >= .9999) and torch.all(gaps <= 2 * maximum[changed])))


class AuditModel(CUDAGraphModelWrapper):
    def __init__(self, model, teacher_pool, native_pool, world, limit=8, permute_prefix=False):
        super().__init__(model)
        self.teacher_pool, self.native_pool = teacher_pool, native_pool
        self.world, self.limit = world, limit
        self.rows, self.teacher_rows, self.audits = [], [], 0
        self.scheduler = None
        self.permute_prefix = permute_prefix

    def _rows(self, logits):
        return logits.view(self.world, logits.shape[0], -1).permute(1, 0, 2).reshape(logits.shape[0], -1)

    def _native_control(self, original, active, kwargs):
        # Test-only copy of the exact visible per-layer/head KV. Same query
        # matrix, positions, model weights, native append/attention/score math.
        # Prefix permutation is allowed only before the causal query suffix.
        indices, indptr, lengths, draft_offsets = [], [0], [], [0]
        kv_indptr = []
        buffer = PillarCachePtr.draft_kv_indices_tensor_buf
        for i, request in enumerate(active):
            ptr = request.kv_cache_ptr
            source = list(ptr.indices)
            qlen = int(kwargs["token_indptr"][i + 1] - kwargs["token_indptr"][i])
            if self.permute_prefix and isinstance(ptr, CompactCachePtr) and ptr.epoch > 0:
                source = source[:-qlen][::-1] + source[-qlen:]
            destination = list(range(indptr[-1], indptr[-1] + len(source)))
            self.native_pool._physical_mem_gpu[:, destination] = original._physical_mem_gpu[:, source]
            if request.status == Request.ReqExecType.DRAFT:
                kv_indptr.append(draft_offsets[-1])
                buffer[..., draft_offsets[-1]:draft_offsets[-1] + len(source)] = torch.tensor(
                    destination, device=original.device, dtype=torch.int32)
                draft_offsets.append(draft_offsets[-1] + len(source))
            else:
                kv_indptr.append(indptr[-1])
            indices.extend(destination)
            indptr.append(indptr[-1] + len(source))
            lengths.append(len(source))
        kv_indptr.append(kv_indptr[-1] + lengths[-1])
        metadata = dict(kwargs)
        make = lambda x: torch.tensor(x, device=original.device, dtype=torch.int32)
        metadata.update(kv_indices=make(indices), all_kv_indptr=make(indptr), all_kv_len_arr=make(lengths),
                        kv_len_arr=make(lengths), kv_indptr=make(kv_indptr), flatten_indices=buffer,
                        flatten_indices_len=draft_offsets[-1])
        self.model.attach_kv_pool(self.native_pool)
        self.native_pool.pre_attn_fwd(**metadata)
        return self.model(metadata["token_ids"])

    def _dense_teacher(self, ids, qlen):
        self.model.attach_kv_pool(self.teacher_pool)
        ptr = FullCachePtr(self.teacher_pool)
        try:
            split = len(ids) - qlen
            chunks = [ids[i:min(i + 256, split)] for i in range(0, split, 256)] + [ids[split:]]
            device = self.teacher_pool.device
            make = lambda x: torch.tensor(x, device=device, dtype=torch.int32)
            for chunk in chunks:
                start = ptr.len_kv_cache
                ptr.allocate_tokens(len(chunk))
                metadata = dict(token_ids=make(chunk), token_indptr=make([0, len(chunk)]), pos_offsets=make([start]),
                                kv_indptr=make([0, ptr.len_kv_cache]), kv_indices=make(ptr.indices),
                                kv_len_arr=make([ptr.len_kv_cache]))
                self.teacher_pool.pre_attn_fwd(**metadata)
                reference = self.model(metadata["token_ids"])
            return reference
        finally:
            torch.cuda.current_stream().synchronize()
            del ptr

    def __call__(self, **kwargs):
        output = super().__call__(**kwargs)
        active = self.scheduler.active_requests
        if self.audits >= self.limit or not any(r.status == Request.ReqExecType.VERIFY for r in active):
            return output
        original = self.model.kv_cache
        try:
            control = self._native_control(original, active, kwargs)
            actual_rows, native_rows = self._rows(output), self._rows(control)
            for i, request in enumerate(active):
                if request.status != Request.ReqExecType.VERIFY or self.audits >= self.limit:
                    continue
                start, end = (int(kwargs["token_indptr"][i]), int(kwargs["token_indptr"][i + 1]))
                actual, expected = actual_rows[start:end], native_rows[start:end]
                row = dict(request_id=int(request.request_id), epoch=int(request.kv_cache_ptr.epoch),
                           logical_len=int(request.len_kv_cache), qlen=end-start, **error_metrics(actual, expected))
                self.rows.append(row)
                assert row["passed"], row
                # Diagnose the initial full path against a recomputed teacher.
                # A strict teacher failure remains reported, never relabeled pass.
                if request.kv_cache_ptr.epoch == 0:
                    dense = self._rows(self._dense_teacher(list(request.token_ids[:request.len_kv_cache]), end-start))
                    self.teacher_rows.append(dict(request_id=int(request.request_id), epoch=0,
                        compact=error_metrics(actual, dense), native_control=error_metrics(expected, dense),
                        compact_vs_native_bitwise=bool(torch.equal(actual, expected))))
                self.audits += 1
        finally:
            self.model.attach_kv_pool(original)
        return output


class ReadyPressureScheduler(CompactSpecScheduler):
    """Test-only barrier presents three real hot verifies simultaneously.

    Requests keep their tokens, probabilities, CPU backing and GPU ownership.
    The unchanged production admission logic must park work that cannot fit,
    then complete it later. This is a lifecycle stress, not a throughput run.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held, self.aligned = [], False
        self.parked_verifies = 0

    def _current_cpu(self):
        return super()._current_cpu() + sum(r.kv_cache_ptr.cold_count for r in self.held)

    def admit_requests(self, pending):
        ready = self.active_requests + self.offload_requests
        if not self.aligned and not pending and len(ready) + len(self.held) == 3:
            waiting = []
            for request in ready:
                if (request.status == Request.ReqExecType.VERIFY
                        and isinstance(request.kv_cache_ptr, CompactCachePtr)
                        and request.kv_cache_ptr.phase == "hot"):
                    self.held.append(request)
                else:
                    waiting.append(request)
            self.active_requests, self.offload_requests = waiting, []
            if len(self.held) == 3:
                self.active_requests += self.held
                self.held, self.aligned = [], True
        super().admit_requests(pending)
        self.parked_verifies += sum(r.status == Request.ReqExecType.VERIFY for r in self.offload_requests)

    def run(self, pending):
        try:
            return super().run(pending)
        finally:
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            for request in self.held:
                request.retire()
            self.held = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--ready-pressure", action="store_true",
                        help="One tiny GPU test: barrier-align real hot verifies and require parked work")
    parser.add_argument("--output-tokens", type=int, default=24)
    parser.add_argument("--pool-pages", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--permute-prefix", action="store_true",
                        help="Separate numerical stress: reverse the native control's old prefix")
    args = parser.parse_args()
    if args.ready_pressure and (not args.tiny or not args.pressure or args.public):
        parser.error("--ready-pressure requires --tiny --pressure, without --public")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    setup_seed(330)
    if args.tiny:
        config = ModelArgs(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           hidden_size=256, head_dim=64, intermediate_size=512, vocab_size=5000)
        model = Transformer(config, device=device, dtype=torch.bfloat16)
        prompts = [[i % 127 for i in range(n)] for n in (193, 311, 129)]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = [tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                    add_generation_prompt=True, enable_thinking=True) for text in (
                    "Explain Euclid's algorithm step by step and prove that it terminates.",
                    "Discuss cache locality. " + "Explain memory hierarchy and locality. " * 48,
                    "Solve for positive integers x and y: x squared minus y squared equals 2024.")]
        model = None
    if args.long:
        prompts = [(prompts[1] * (20000 // len(prompts[1]) + 1))[:20000]]
    if args.pressure:
        prompts = [(ids * (700 // len(ids) + 1))[:700] for ids in prompts]
    capacity = args.pool_pages or (22500 if args.long else 1850 if args.pressure else 1600)
    if args.public:
        from serve.run import run_inference
        outputs = run_inference([Request(ids, args.output_tokens) for ids in prompts], args.model,
            kv_cache="compact", spec_stride=4, budget_ratio=.1, num_min_budget=128,
            max_batch_size=3, chunk_size=256, tp_ranks=world, reduce_method="naive", temperature=args.temperature,
            compact_pool_pages=capacity, compact_cpu_pages=capacity * 2, result_dir=str(out),
            acceptance_log_path=str(out / f"acceptance.rank{rank}.jsonl"), seed=330)
        if rank == 0:
            (out / "public.json").write_text(json.dumps(outputs, indent=2))
        return
    if model is None:
        model = Transformer.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    model = setup_tp_worker(model, tp_ranks=world, device=device, dtype=torch.bfloat16, reduce_method="naive")
    control = dist.new_group(backend="gloo") if world > 1 else None
    cfg = model.config
    spec = dict(spec_stride=4, budget_ratio=.1, num_min_budget=16 if args.ready_pressure else 128,
                max_batch_size=3)
    init = dict(dtype=torch.bfloat16, head_dim=cfg.head_dim, num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads, num_layers=cfg.num_layers, capacity=capacity,
                device=device, max_batch_size=3, spec_cache_args=spec)
    CompactCachePtr.initialize_class_metadata(**init)
    PillarCachePtr.initialize_class_metadata(**init)
    FullCachePtr.initialize_class_metadata(**init)
    pool = KVPool(cfg, capacity, device, CompactCachePtr, dtype=torch.bfloat16, allocate_legacy_cpu=False)
    teacher = KVPool(cfg, max(map(len, prompts)) + args.output_tokens + 16, device,
                     FullCachePtr, dtype=torch.bfloat16, allocate_legacy_cpu=False)
    native = KVPool(cfg, capacity, device, PillarCachePtr, dtype=torch.bfloat16, allocate_legacy_cpu=False)
    model.attach_kv_pool(pool)
    wrapper = AuditModel(model, teacher, native, world, limit=4 if args.long else 12,
                         permute_prefix=args.permute_prefix)
    scheduler_class = ReadyPressureScheduler if args.ready_pressure else CompactSpecScheduler
    worker = scheduler_class(pool, wrapper, **spec, compact_control_group=control,
        compact_cpu_pages=capacity * 2, compact_validate_moves=True, eos_token_id=(),
        admit_policy="offloading", kv_cache="compact", async_cpu=False, tp_ranks=world,
        temperature=args.temperature, enable_torch_profiler=False, profiler_result_dir=str(out),
        batch_policy="naive" if args.ready_pressure else "greedy", enable_cuda_graph=False, chunk_size=256,
        trace_name="native-compact-validation",
        acceptance_log_path=str(out / f"acceptance.rank{rank}.jsonl"))
    wrapper.scheduler = worker
    with torch.inference_mode():
        outputs = worker.run([Request(ids, args.output_tokens) for ids in prompts])
    worker.close_acceptance_log()
    generated = [list(req.token_ids[req.len_input:]) for req in outputs]
    digest = hashlib.sha256(json.dumps(generated).encode()).hexdigest()
    digests = [digest]
    if control is not None:
        digests = [None] * world
        dist.all_gather_object(digests, digest, group=control)
    assert len(set(digests)) == 1, digests
    assert worker.metrics["migrations"] and worker.metrics["restores"]
    assert worker.metrics["d2h_bytes"] and worker.metrics["h2d_bytes"]
    assert any(row["epoch"] > 0 for row in wrapper.rows)
    if args.ready_pressure:
        assert worker.aligned and worker.parked_verifies > 0 and worker.metrics["capacity_waits"] > 0, worker.metrics
    result = dict(status="passed_native_control_with_dense_teacher_difference" if any(not r["compact"]["passed"] for r in wrapper.teacher_rows) else
                  ("passed_with_numerical_difference" if any(r["changed_argmax"] for r in wrapper.rows) else "passed"),
                  world=world, tiny=args.tiny, temperature=args.temperature, permute_prefix=args.permute_prefix,
                  strict_teacher_passed=all(r["compact"]["passed"] for r in wrapper.teacher_rows),
                  native_control_passed=all(r["passed"] for r in wrapper.rows),
                  outputs=generated, audits=wrapper.rows, dense_teacher_audits=wrapper.teacher_rows, metrics=worker.metrics, rank_digests=digests)
    if args.ready_pressure:
        result["ready_pressure"] = dict(test_only_barrier=True, aligned=worker.aligned,
                                        parked_verify_observations=worker.parked_verifies)
    (out / f"validation.rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result["status"], world=world, audit_count=len(wrapper.rows), metrics=worker.metrics)), flush=True)


if __name__ == "__main__":
    main()
