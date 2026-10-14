"""Synchronous native compact KV scheduling at whole-forward boundaries.

SpecScheduler remains the authority for input tokens, logical RoPE positions,
sampling/rejection, and request transitions. Waiting requests keep their page
ownership. Capacity only changes which ready requests join a mixed batch.
"""

import json
import math
import os
import time

import torch
import torch.distributed as dist

from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.request import Request
from serve.scheduler.spec_scheduler import SpecScheduler


def coordinated(group, label, operation, rollback=None):
    """Coordinate setup/allocation errors before the next model collective.

    CUDA faults during a model collective are fatal; this is not recovery.
    """
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if group is not None:
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        if result is not None and rollback is not None:
            rollback(result)
        raise RuntimeError(f"Native compact {label}: {errors}")
    return result


class _RankZeroRejection:
    def __init__(self, scheduler, original):
        self.scheduler, self.original = scheduler, original

    def __call__(self, **kwargs):
        scheduler = self.scheduler
        result = coordinated(scheduler.group, "rejection", lambda:
                             self.original(**kwargs) if scheduler.rank == 0 else
                             torch.empty((len(kwargs["num_draft_tokens"]), kwargs["max_spec_len"] + 1),
                                         dtype=torch.int32, device=scheduler.kv_pool.device))
        if scheduler.group is not None:
            dist.broadcast(result, src=0)
        return result


class CompactSpecScheduler(SpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        if kwargs.get("async_cpu") or kwargs.get("enable_cuda_graph"):
            raise ValueError("Native compact currently uses synchronous eager execution")
        k, ratio, minimum = kwargs["spec_stride"], kwargs["budget_ratio"], kwargs["num_min_budget"]
        if (k < 1 or not math.isfinite(ratio) or not 0 < ratio <= 1
                or not k + 2 <= minimum < CompactCachePtr.MAX_LEN_SELECTED_KV_CACHE):
            raise ValueError("Require k>=1, 0<ratio<=1, and k+2<=min_budget<6144")
        if (k + 1) * kv_pool.num_qo_heads // kv_pool.num_kv_heads > 64:
            raise ValueError("Native score kernel supports at most 64 packed QO rows")
        super().__init__(kv_pool, model, **kwargs)
        self.group = kwargs.get("compact_control_group")
        self.rank = dist.get_rank() if self.group is not None else 0
        if kwargs["tp_ranks"] > 1 and self.group is None:
            raise ValueError("TP compact execution requires a Gloo control group")
        self.cpu_capacity = kwargs["compact_cpu_pages"]
        if self.cpu_capacity <= 0:
            raise ValueError("Positive compact pinned CPU page budget required")
        self.validate_moves = kwargs.get("compact_validate_moves", False)
        self.rejection_sampler = _RankZeroRejection(self, self.rejection_sampler)
        self.metrics = dict(forwards=0, request_verify_rounds=0, verify_forwards=0,
                            migrations=0, restores=0, admission_waits=0, capacity_waits=0,
                            max_actual_batch=0, batch_sum=0, mixed_forwards=0,
                            gpu_peak_inuse_pages=0, cpu_peak_pages=0, d2h_bytes=0, h2d_bytes=0,
                            freed_source_pages=0, max_live_requests=0)
        self.page_bytes = (kv_pool.num_layers * 2 * kv_pool.num_kv_heads * kv_pool.head_dim
                           * kv_pool._physical_mem_gpu.element_size())
        self._last_served = {}
        self._result_dir = kwargs.get("profiler_result_dir")
        if self._result_dir:
            os.makedirs(self._result_dir, exist_ok=True)

    def _estimate(self, request):
        # The original scheduler checks limits after verify and can overshoot
        # desired output. Reserve its final speculative tail as well.
        length = request.len_oracle_ctx + self.spec_stride + 1
        hot = min(length, max(self.spec_cache_args["num_min_budget"],
                              math.ceil(length * self.spec_cache_args["budget_ratio"])))
        cold = length - hot
        packed_base = hot + self.spec_stride + 1
        ptr = request.kv_cache_ptr
        packed = isinstance(ptr, CompactCachePtr) and ptr.phase != "full"
        if packed:
            base, extra = packed_base, cold + packed_base
        else:
            base = max(request.len_input + self.spec_stride + 1, packed_base)
            extra = max(packed_base, cold + 2 * packed_base - base)
        return base, extra, cold

    def _fits(self, requests):
        estimates = [self._estimate(request) for request in requests]
        return (sum(x[0] for x in estimates) + max((x[1] for x in estimates), default=0)
                <= self.kv_pool.capacity
                and sum(x[2] for x in estimates) + max((x[2] for x in estimates), default=0)
                <= self.cpu_capacity)

    def _broadcast(self, value):
        if self.group is None:
            return value
        items = [value if self.rank == 0 else None]
        dist.broadcast_object_list(items, src=0, group=self.group)
        return items[0]

    def _current_cpu(self):
        return sum(r.kv_cache_ptr.cold_count for r in self.active_requests + self.offload_requests
                   if isinstance(r.kv_cache_ptr, CompactCachePtr))

    def _record_peak(self, extra_cpu=0):
        self.metrics["gpu_peak_inuse_pages"] = max(self.metrics["gpu_peak_inuse_pages"],
                                                   self.kv_pool.capacity - self.kv_pool.num_free_pages)
        self.metrics["cpu_peak_pages"] = max(self.metrics["cpu_peak_pages"], self._current_cpu() + extra_cpu)

    def _work_cost(self, request):
        ptr = request.kv_cache_ptr
        if request.status == Request.ReqExecType.NORMAL:
            append = min(request.len_token_ids - request.len_kv_cache, self._args["chunk_size"])
            return append, 0, 0, 0, append
        if request.status == Request.ReqExecType.DRAFT:
            return 1, 0, 0, ptr.len_loading_indices + 1, 1
        logical = ptr.len_kv_cache + 1
        hot = min(logical, max(self.spec_cache_args["num_min_budget"],
                               math.ceil(logical * self.spec_cache_args["budget_ratio"])))
        restore = ptr.cold_count if ptr.phase == "hot" else 0
        return 1 + restore, hot, logical - hot, 0, self.spec_stride + 1 - request.initial_idx

    def admit_requests(self, pending_requests):
        # Reconsider parked work every iteration. Oldest unserved first provides
        # a finite wait while preserving the same speculative k/r for all work.
        self.active_requests += self.offload_requests
        self.offload_requests = []
        while pending_requests and len(self.active_requests) < self.max_batch_size:
            if any(r.status == Request.ReqExecType.NORMAL for r in self.active_requests):
                break
            request = pending_requests[0]
            fits = self._broadcast(self._fits(self.active_requests + [request]))
            if not fits:
                self.metrics["admission_waits"] += 1
                if not self.active_requests:
                    raise MemoryError("Request cannot fit original pool + restore/repack peak or CPU double backing")
                break
            coordinated(self.group, "admit buffers", lambda: request.attach_kv_pool(self.kv_pool))
            self.add_request(request)
            self.prefilling_request = request
            pending_requests.pop(0)
            break  # Preserve the original one active chunked-prefill policy.
        self.metrics["max_live_requests"] = max(self.metrics["max_live_requests"], len(self.active_requests))
        chosen, waiting = [], []
        gpu, replacement, cpu_new, drafts, verifies, queries = 0, 0, 0, 0, 0, 0
        cpu_current = self._current_cpu()
        # Replacement packs run one at a time after the batched model forward;
        # sum restore/append + MAX replacement is sufficient for that ordering.
        for request in sorted(self.active_requests, key=lambda r: self._last_served.get(r.request_id, -1)):
            alloc, hot, cold, draft, qo = self._work_cost(request)
            verify = request.len_kv_cache + 1 if request.status == Request.ReqExecType.VERIFY else 0
            fits = (gpu + alloc + max(replacement, hot) <= self.kv_pool.num_free_pages
                    and cpu_current + cpu_new + cold <= self.cpu_capacity
                    and drafts + draft < CompactCachePtr.MAX_TOTAL_DRAFT_KV_LEN
                    and verifies + verify < CompactCachePtr.MAX_TOTAL_VERIFY_KV_LEN
                    and queries + qo < CompactCachePtr._MAX_NNZ)
            if fits:
                chosen.append(request)
                gpu += alloc
                replacement, cpu_new = max(replacement, hot), cpu_new + cold
                drafts, verifies, queries = drafts + draft, verifies + verify, queries + qo
            else:
                waiting.append(request)
        if not chosen:
            raise MemoryError("No ready request fits the reserved native metadata/transfer capacity")
        self.active_requests, self.offload_requests = chosen, waiting
        self.metrics["capacity_waits"] += len(waiting)
        if self.group is not None:
            ids = [r.request_id for r in chosen]
            expected = self._broadcast(ids)
            if ids != expected:
                raise RuntimeError("TP scheduler states diverged before forward")

    def prepare_forward_inputs(self):
        for request in self.active_requests:
            ptr = request.kv_cache_ptr
            if request.status == Request.ReqExecType.VERIFY and ptr.phase == "hot":
                def restore():
                    transfer = ptr.begin_restore()
                    transfer.wait()
                    return transfer
                transfer = coordinated(self.group, "restore", restore, ptr.abort_transfer)
                ptr.commit_restore(transfer)
                self.metrics["restores"] += 1
                self.metrics["h2d_bytes"] += ptr.cold_count * self.page_bytes
                self._record_peak()
        return super().prepare_forward_inputs()

    def _sample(self, logits, count):
        def sample():
            if self.rank == 0:
                return self.sampler(logits, self.sampling_metadata, count)
            return (torch.empty(count, dtype=torch.int32, device=logits.device),
                    torch.zeros((count, logits.shape[1]), dtype=torch.float32, device=logits.device))
        tokens, probs = coordinated(self.group, "sampling", sample)
        # Greedy returns int64 and the original stochastic kernel returns
        # int32. NCCL requires every rank to agree before the broadcast.
        tokens = tokens.to(torch.int32)
        if self.group is not None:
            dist.broadcast(tokens, src=0)
            if self.temperature == 0:
                if self.rank != 0:
                    probs.scatter_(1, tokens.long()[:, None], 1.)
            else:
                dist.broadcast(probs, src=0)
        return tokens, probs

    def update_requests(self, *args, **kwargs):
        completed = super().update_requests(*args, **kwargs)
        for request in self.active_requests:
            ptr = request.kv_cache_ptr
            if isinstance(ptr, CompactCachePtr) and ptr.needs_pack:
                old_count = len(ptr.indices)
                def pack():
                    transfer = ptr.begin_pack(validate=self.validate_moves)
                    self._record_peak(extra_cpu=transfer.cold.shape[1])
                    transfer.wait()
                    return transfer
                transfer = coordinated(self.group, "repack", pack, ptr.abort_transfer)
                ptr.commit_pack(transfer)
                self.metrics["migrations"] += 1
                self.metrics["d2h_bytes"] += ptr.cold_count * self.page_bytes
                self.metrics["freed_source_pages"] += old_count
        return completed

    def run(self, pending_requests):
        pending_requests = list(pending_requests)
        for i, request in enumerate(pending_requests):
            request.request_id = i
            limit = request.len_oracle_ctx
            if limit is None or request.len_output < 1 or request.len_input < 1:
                raise ValueError("Compact requests require a nonempty prompt and positive finite output limit")
            if max(self.spec_cache_args["num_min_budget"], math.ceil(
                    (limit + self.spec_stride + 1) * self.spec_cache_args["budget_ratio"])) >= 6144:
                raise ValueError("Request's maximum hot budget must be strictly below 6144")
        completed = []
        start = time.perf_counter()
        try:
            while pending_requests or self.active_requests or self.offload_requests:
                self.step_id += 1
                coordinated(self.group, "admission", lambda: self.admit_requests(pending_requests))
                phases = [request.status for request in self.active_requests]
                def setup():
                    host, tensors = self.prepare_forward_inputs()
                    self.model.register_input(**tensors)
                    return host, tensors
                host, tensors = coordinated(self.group, "forward metadata", setup)
                self._record_peak()
                logits = self.model(**tensors)
                tokens, probs = self._sample(logits, len(host["token_ids"]))
                del logits
                host.update(self.kv_pool.output_snapshot(host))
                done = torch.cuda.current_stream(self.kv_pool.device).record_event()
                coordinated(self.group, "whole-forward completion", done.synchronize)
                for request in self.active_requests:
                    self._last_served[request.request_id] = self.step_id
                newly_completed = self.update_requests(host["token_indptr"], tokens, probs, host)
                completed.extend(newly_completed)
                self.kv_pool.step()
                self.metrics["forwards"] += 1
                self.metrics["request_verify_rounds"] += phases.count(Request.ReqExecType.VERIFY)
                self.metrics["verify_forwards"] += int(Request.ReqExecType.VERIFY in phases)
                self.metrics["batch_sum"] += len(phases)
                self.metrics["max_actual_batch"] = max(self.metrics["max_actual_batch"], len(phases))
                self.metrics["mixed_forwards"] += int(len(set(phases)) > 1)
                self.prof.step()
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            self.metrics["elapsed_seconds"] = time.perf_counter() - start
            self.metrics["gpu_final_free_pages"] = self.kv_pool.num_free_pages
            self.metrics["gpu_capacity_pages"] = self.kv_pool.capacity
            if self.kv_pool.num_free_pages != self.kv_pool.capacity:
                raise RuntimeError("Compact scheduler leaked original pool pages")
            if self._result_dir:
                with open(os.path.join(self._result_dir, f"compact-metrics.rank{self.rank}.json"), "w") as out:
                    json.dump(self.metrics, out, indent=2)
            return sorted(completed, key=lambda r: r.request_id)
        finally:
            # All ordinary exits wait before freeing mapped CPU and pool pages.
            # A CUDA fault is fatal and is deliberately not advertised as recoverable.
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            for request in self.active_requests + self.offload_requests:
                if request.kv_cache_ptr is not None:
                    request.retire()
            self.active_requests, self.offload_requests = [], []
