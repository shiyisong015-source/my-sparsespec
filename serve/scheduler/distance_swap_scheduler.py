"""Next-Verify history replacement, chunked swaps and a TP1 transfer worker."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

import torch

from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.kv_cache_ptr.distance_swap import CUDASwapTransport, SwapStorage
from serve.request.request import Request
from serve.scheduler.async_cpu_compact_scheduler import AsyncCPUCompactSpecScheduler
from serve.scheduler.distance_policy import Candidate, DISTANCE_SCOPE, distance_to_verify
from serve.scheduler.distance_swap_engine import DistanceSwapEngine
from serve.scheduler.dynamic_compact_scheduler import DynamicCompactSpecScheduler
from serve.scheduler.spec_scheduler import SpecScheduler


class DistanceSwapSpecScheduler(DynamicCompactSpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        if kwargs.get("tp_ranks", 1) != 1 or kwargs.get("compact_control_group") is not None:
            raise ValueError("distance_swap currently supports TP1 only; background TP coordination is not implemented")
        if kwargs.get("compact_reservation") or kwargs.get("compact_dynamic_admission"):
            raise ValueError("distance_swap is an independent policy, not reservation/dynamic admission")
        if kwargs.get("compact_early_verify_length") is not None:
            raise ValueError("distance_swap preserves the native k boundary")
        if kwargs.get("enable_cuda_graph"):
            raise ValueError("distance_swap currently requires eager execution")
        chunk = kwargs.get("distance_swap_chunk_pages", 256)
        if type(chunk) is not int or chunk < 1:
            raise ValueError("distance_swap_chunk_pages must be positive")
        super().__init__(kv_pool, model, **kwargs)
        self._residency_lock = threading.RLock()
        self._swap = DistanceSwapEngine(kv_pool, CUDASwapTransport(kv_pool), self._residency_lock,
                                       self.cpu_capacity, chunk,
                                       device_context=self._transfer_context, trace=self._swap_event)
        self._service_order = {}
        self._service_sequence = 0
        self._swap_generation = 0
        self.metrics.update(distance_swap_enabled=True, distance_swap_chunk_pages=chunk)

    def _swap_event(self, row):
        if self._result_dir:
            with (Path(self._result_dir) / f"distance-swap.rank{self.rank}.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")

    @contextmanager
    def _transfer_context(self):
        with torch.cuda.device(self.kv_pool.device), torch.inference_mode():
            yield

    def _run_stage(self, label, operation):
        # Metadata/selection/append and worker residency changes have a single
        # host owner at a time. Model submission is outside this lock.
        with self._residency_lock:
            self._swap.check()
            return super()._run_stage(label, operation)

    def run(self, pending_requests):
        pending_requests = list(pending_requests)
        for request in pending_requests:
            length = request.len_oracle_ctx
            if length is not None:
                hot = min(length, max(self.spec_cache_args["num_min_budget"],
                                     math.ceil(length * self.spec_cache_args["budget_ratio"])))
                if length + hot + self.spec_stride + 2 > self.kv_pool.capacity:
                    raise ValueError("distance_swap requires one full Verify plus its stable Draft view and append space")
        self._swap.start()
        try:
            return super().run(pending_requests)
        finally:
            self._swap.stop()  # Also covers validation failure before pipeline.run.

    def _drain_pipeline_transfers(self):
        self._swap.stop()
        self._swap.check()

    def _before_pipeline_cleanup(self):
        self._swap.stop()

    def _after_pipeline_streams_drained(self):
        with self._residency_lock:
            # Normal completion has already retired these owners. An early
            # profile exit can leave live or deferred-retired requests here.
            for storage in self._swap.storages.values():
                storage.ptr.close()
            self._swap.storages.clear()
            self._swap.garbage.clear()
            self._swap.compute.clear()

    def _current_cpu(self):
        return self._swap.cpu_pages if hasattr(self, "_swap") else 0

    def _publish_candidates(self):
        # Only history used by Verify is replaceable. Draft's immediate hot view
        # stays resident; queue position and query capacity do not change how
        # many of THIS request's Draft steps remain before Verify.
        candidates = []
        for request in self._requests.values():
            if request.status not in (Request.ReqExecType.DRAFT, Request.ReqExecType.VERIFY):
                continue
            verify = request.status == Request.ReqExecType.VERIFY
            distance = distance_to_verify(self.spec_stride, request.cur_spec_idx, verify=verify)
            candidates.append(Candidate(request.request_id, distance, verify,
                                        self._service_order.setdefault(request.request_id, -1),
                                        self._last_served.get(request.request_id, -1)))
        self._swap.publish(candidates, max(1, len(self._requests)))

    def _on_compute_submitted(self, requests, phases, model_done):
        with self._residency_lock:
            for request, phase in zip(requests, phases):
                identity = request.request_id
                self._swap.sealed.discard(identity)
                self._swap.compute[identity] = model_done, phase == Request.ReqExecType.VERIFY
                self._service_sequence += 1
                self._service_order[identity] = self._service_sequence
            self._swap.wake.set()

    def _defer_retire(self, request):
        self._swap.retire(request.request_id)
        super()._defer_retire(request)

    def _poll_retired(self):
        with self._residency_lock:
            self._poll_retired_locked()

    def _poll_retired_locked(self):
        # Parent retirement fences cover compute/control; include all transfer
        # and deferred-hot ownership before allowing ptr.close().
        self._swap.poll()
        blocked, ready = [], []
        for item in self._retired:
            storage = next((s for s in self._swap.storages.values() if s.ptr is item.ptr), None)
            extra = any(ptr is item.ptr for ptr, _, _ in self._swap.garbage)
            if storage is not None and (storage.busy or not storage.source_ready.query() or extra):
                blocked.append(item)
            else:
                ready.append(item)
        self._retired = ready
        super()._poll_retired()
        self._retired.extend(blocked)
        for identity, storage in list(self._swap.storages.items()):
            if storage.ptr._closed:
                self._swap.forget(identity)

    def _ready(self, request):
        if request.status == Request.ReqExecType.NORMAL:
            return True
        storage = self._swap.storages.get(request.request_id)
        if storage is None or not storage.hot_ready:
            return False
        if request.status == Request.ReqExecType.DRAFT:
            return True
        return request.status == Request.ReqExecType.VERIFY and storage.complete

    def admit_requests(self, pending_requests):
        self._poll_retired()
        self._output_keepalive = [(output, event) for output, event in self._output_keepalive if not event.query()]
        self._publish_candidates()
        self._swap.tick()
        self._swap.pressure = 0
        chosen, pages, queries, drafts, verifies, indices = [], 0, 0, 0, 0, 0

        def choose(request):
            nonlocal pages, queries, drafts, verifies, indices
            if not self._ready(request):
                return False
            query, append = self._cost(request, self.token_budget - queries)
            if query <= 0 or queries + query > self.token_budget:
                self.metrics["token_budget_waits"] += 1
                return False
            ptr = request.kv_cache_ptr
            draft = ptr.len_loading_indices + append if request.status == Request.ReqExecType.DRAFT else 0
            verify = ptr.len_kv_cache + append if request.status == Request.ReqExecType.VERIFY else 0
            n = len(ptr.indices) + append
            if (drafts + draft >= CompactCachePtr.MAX_TOTAL_DRAFT_KV_LEN
                    or verifies + verify >= CompactCachePtr.MAX_TOTAL_VERIFY_KV_LEN
                    or indices + n > FullCachePtr._MAX_KV_CONTEXT_LEN):
                self.metrics["metadata_waits"] += 1
                return False
            if pages + append > self.kv_pool.num_free_pages:
                self._swap.pressure = max(self._swap.pressure, pages + append)
                self.metrics["kv_capacity_waits"] += 1
                return False
            request.scheduled_tokens = query
            chosen.append(request)
            self._swap.sealed.add(request.request_id)
            pages, queries = pages + append, queries + query
            drafts, verifies, indices = drafts + draft, verifies + verify, indices + n
            return True

        ordered = sorted(self._requests.values(), key=lambda r: (
            self._service_order.get(r.request_id, -1), r.request_id))
        for request in ordered:
            request.scheduled_tokens = None
            choose(request)
        while pending_requests and len(self._requests) < self.max_batch_size and queries < self.token_budget:
            request = pending_requests[0]
            query = min(request.len_input, self._args["chunk_size"], self.token_budget - queries)
            if self._current_cpu() >= self.cpu_admit_limit:
                self.metrics["cpu_watermark_waits"] += 1
                break
            # Admission consumes the current prefill chunk only. Do not
            # reserve every request's final hot budget or a future full Verify.
            # Later growth/restore uses actual pages and chunked replacement.
            if pages + query > self.kv_pool.num_free_pages:
                self._swap.pressure = max(self._swap.pressure, pages + query)
                self.metrics["admission_waits"] += 1
                break
            request.attach_kv_pool(self.kv_pool)
            self.active_requests = list(self._requests.values())
            self.add_request(request)
            self._requests[request.request_id] = request
            self._service_order[request.request_id] = -1
            pending_requests.pop(0)
            if not choose(request):
                break
        stalls = sorted((r for r in self._requests.values() if r.status == Request.ReqExecType.STALL),
                        key=lambda r: r.prev_idx)
        self.active_requests = chosen + stalls
        active = {r.request_id for r in self.active_requests}
        self.offload_requests = [r for r in self._requests.values() if r.request_id not in active]
        self._swap.reserve = max(1, len(self._requests), pages)
        self._swap.wake.set()
        self.metrics["capacity_waits"] += len(self.offload_requests)
        self.metrics["max_live_requests"] = max(self.metrics["max_live_requests"], len(self._requests))
        self.metrics["scheduled_token_peak"] = max(self.metrics["scheduled_token_peak"], queries)
        self.metrics["prefill_request_peak"] = max(self.metrics["prefill_request_peak"], sum(
            r.status == Request.ReqExecType.NORMAL for r in chosen))
        self.metrics["prefill_query_tokens"] += sum(r.scheduled_tokens for r in chosen
                                                   if r.status == Request.ReqExecType.NORMAL)

    def prepare_forward_inputs(self):
        if any(s.busy for s in self._swap.storages.values()):
            self.metrics["forwards_with_transfer_inflight"] += 1
        return AsyncCPUCompactSpecScheduler.prepare_forward_inputs(self)

    def _sample(self, logits, count):
        return AsyncCPUCompactSpecScheduler._sample(self, logits, count)

    def update_requests(self, *args, **kwargs):
        completed = SpecScheduler.update_requests(self, *args, **kwargs)
        for request in completed:
            self._requests.pop(request.request_id, None)
        for request in self.active_requests:
            ptr = request.kv_cache_ptr
            if not isinstance(ptr, CompactCachePtr):
                continue
            old = self._swap.storages.get(request.request_id)
            if ptr.needs_pack:
                if old is not None and (old.busy or not old.complete):
                    raise RuntimeError("New selection overlaps unfinished epoch migration")
                ptr.keep_full_resident()
            elif old is not None:
                continue
            elif ptr.phase == "full":
                # Native first cycle has no sparse selection: all history is
                # the forced recent view until the first Verify/selection.
                ptr.phase = "resident"
                ptr.recent_count = ptr.len_loading_indices
                ptr._resident_selected_at = ptr.len_kv_cache
            else:
                continue
            self._swap_generation += 1
            storage = SwapStorage(ptr, self._swap.transport, self._swap.chunk_pages, self._swap_generation)
            if old is not None and old.hot_pages:
                self._swap.garbage.append((ptr, old.hot_pages, storage.source_ready))
            if self._swap.target == request.request_id:
                self._swap._cancel_plan()
            self._swap.storages[request.request_id] = storage
            self._swap.compute.pop(request.request_id, None)
            storage.start_hot()
        self._publish_candidates()
        return completed

    def wait_for_ready_requests(self):
        self._swap.check()
        if not self._requests and not self._retired:
            raise RuntimeError("Empty distance-swap queue")
        # Release the residency lock while waiting; the transfer worker must
        # be able to fulfill the dependency that wakes the compute loop.
        self._swap.wake.set()
        self._swap.stopping.wait(.001)
        with self._residency_lock:
            self._swap.check()
            self._poll_retired()
            self._swap.tick()
            active = any(s.busy for s in self._swap.storages.values())
            if not active and not self._swap.producers and self._swap.target is None:
                if any((self._ready(r) and 0 < self._cost(r, self.token_budget)[1]
                        <= self.kv_pool.num_free_pages) or r.status == Request.ReqExecType.STALL
                       for r in self._requests.values()):
                    return
                raise MemoryError("distance_swap cannot progress under request priority, CPU/GPU capacity and lease constraints")

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        with self._residency_lock:
            self.metrics.update(self._swap.metrics)
            report = super().transfer_timing_snapshot(elapsed_seconds)
            report.update(mode="distance_swap", transfer_execution="independent TP1 worker; chunked H2D/D2H",
                          distance_scope=DISTANCE_SCOPE, cpu_backing_scope="chronological prefix including selected KV",
                          cpu_swap_pages=self._swap.cpu_pages)
            return report
