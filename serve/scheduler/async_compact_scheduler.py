"""On-demand KV transfers with request readiness and deferred page release.

The native model/sampling/rejection path remains synchronous on its compute
stream. A separate stream carries one bounded KV transaction at a time;
ready requests continue computing while that transaction is in flight.
"""
from dataclasses import dataclass
import math
import time

import torch
import torch.distributed as dist

from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.request import Request
from serve.scheduler.compact_scheduler import CompactSpecScheduler, coordinated
from serve.scheduler.spec_scheduler import SpecScheduler


@dataclass
class TransferJob:
    request: object
    direction: str
    transfer: object
    source_pages: int
    cold_pages: int


class AsyncCompactSpecScheduler(CompactSpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        super().__init__(kv_pool, model, **kwargs)
        self.transfer_stream = torch.cuda.Stream(device=kv_pool.device)
        self._requests = {}
        self._transfers = {}
        self.metrics.update(resident_updates=0, transfers_submitted=0,
                            forwards_with_transfer_inflight=0, transfer_idle_waits=0,
                            pressure_offloads=0, max_transfers_inflight=0,
                            transfer_poll_tp_wall_seconds=0.)

    def _current_cpu(self):
        total = 0
        for request in self._requests.values():
            ptr = request.kv_cache_ptr
            if isinstance(ptr, CompactCachePtr):
                total += ptr.cold_count
                pending = ptr._pending
                if pending is not None and pending.cold is not None:
                    total += pending.cold.shape[1]
        return total

    def _reserve_pages(self, requests=None):
        # Keep enough free space to compact one resident cache even after
        # appending this round. This is a pool-page reserve, not a CUDA tensor.
        requests = self._requests.values() if requests is None else requests
        return max((min(r.len_oracle_ctx + self.spec_stride + 1, max(
            self.spec_cache_args["num_min_budget"], math.ceil(
                (r.len_oracle_ctx + self.spec_stride + 1) * self.spec_cache_args["budget_ratio"])))
                    + self.spec_stride + 1 for r in requests), default=0)

    def _submit_transfer(self, request, direction):
        if self._transfers:
            raise RuntimeError("Only one KV transaction may be in flight")
        ptr = request.kv_cache_ptr
        source_pages = len(ptr.indices)
        timing = {}

        def begin():
            started = time.perf_counter()
            producer = torch.cuda.current_stream(self.kv_pool.device).record_event()
            self.transfer_stream.wait_event(producer)
            compute_done = getattr(self, "_compute_fences", {}).get(request.request_id)
            if compute_done is not None:
                self.transfer_stream.wait_event(compute_done)
            with torch.cuda.stream(self.transfer_stream):
                if direction == "repack":
                    ptr.prepare_resident_offload()
                    transfer = ptr.begin_pack(validate=self.validate_moves)
                else:
                    transfer = ptr.begin_restore()
            timing["submit_wall_seconds"] = time.perf_counter() - started
            return transfer

        transfer = coordinated(self.group, f"async {direction} submit", begin,
                               ptr.abort_transfer, timing=timing)
        cold_pages = transfer.cold.shape[1] if direction == "repack" else ptr.cold_count
        self._transfers[request.request_id] = TransferJob(
            request, direction, transfer, source_pages, cold_pages)
        for stage, seconds in timing.items():
            key = f"{direction}_{stage}"
            self.metrics[key] = self.metrics.get(key, 0.) + seconds
        self.metrics["transfers_submitted"] += 1
        self.metrics["max_transfers_inflight"] = max(
            self.metrics["max_transfers_inflight"], len(self._transfers))
        if direction == "repack":
            self.metrics["pressure_offloads"] += 1
        self._record_peak()

    def _poll_transfers(self):
        if not self._transfers:
            return
        jobs = sorted(self._transfers.items())
        local = [(identity, job.direction, job.transfer.ready.query()) for identity, job in jobs]
        observations = [local]
        if self.group is not None:
            observations = [None] * dist.get_world_size(self.group)
            started = time.perf_counter()
            dist.all_gather_object(observations, local, group=self.group)
            duration = time.perf_counter() - started
            self.metrics["transfer_poll_tp_wall_seconds"] += duration
            key = f"{jobs[0][1].direction}_tp_coordination_wall_seconds"
            self.metrics[key] = self.metrics.get(key, 0.) + duration
        signature = [(identity, direction) for identity, direction, _ in local]
        if any([(identity, direction) for identity, direction, _ in other] != signature
               for other in observations):
            raise RuntimeError("TP in-flight KV jobs diverged")
        for index, (identity, job) in enumerate(jobs):
            if not all(other[index][2] for other in observations):
                continue
            ptr = job.request.kv_cache_ptr
            started = time.perf_counter()
            if job.direction == "restore":
                ptr.commit_restore(job.transfer, nonblocking=True)
                self.metrics["restores"] += 1
                self.metrics["h2d_bytes"] += job.cold_pages * self.page_bytes
            else:
                ptr.commit_pack(job.transfer, nonblocking=True)
                self.metrics["migrations"] += 1
                self.metrics["d2h_bytes"] += job.cold_pages * self.page_bytes
                self.metrics["freed_source_pages"] += job.source_pages
            key = f"{job.direction}_commit_wall_seconds"
            self.metrics[key] = self.metrics.get(key, 0.) + time.perf_counter() - started
            del self._transfers[identity]

    def _pressure_victim(self):
        free = self.kv_pool.num_free_pages
        cpu = self._current_cpu()
        candidates = []
        for request in self._requests.values():
            ptr = request.kv_cache_ptr
            if (not isinstance(ptr, CompactCachePtr) or ptr.phase != "resident"
                    or request.request_id in self._transfers
                    or request.status == Request.ReqExecType.STALL):
                continue
            hot = ptr.len_loading_indices
            cold = len(ptr.indices) - hot
            if cold <= 0 or hot > free or cpu + cold > self.cpu_capacity:
                continue
            # Prefer draft requests with time before their next verify. Full
            # resident verify can also be spilled to make progress under pressure.
            distance = (self.spec_stride - request.cur_spec_idx
                        if request.status == Request.ReqExecType.DRAFT else 0)
            candidates.append((distance > 0, distance, cold, -request.request_id, request))
        return max(candidates, key=lambda entry: entry[:-1])[-1] if candidates else None

    def _append_cost(self, request):
        if request.status == Request.ReqExecType.NORMAL:
            if request.kv_cache_ptr is None:
                return min(request.len_input, self._args["chunk_size"])
            return min(request.len_token_ids - request.len_kv_cache, self._args["chunk_size"])
        return 1

    def admit_requests(self, pending_requests):
        self._poll_transfers()
        self.active_requests = list(self._requests.values())
        self.offload_requests = []
        reserve = self._reserve_pages()
        pending_needs_space = False
        if (pending_requests and len(self._requests) < self.max_batch_size
                and not any(r.status == Request.ReqExecType.NORMAL for r in self._requests.values())):
            request = pending_requests[0]
            candidates = list(self._requests.values()) + [request]
            if self._broadcast(self._fits(candidates)):
                new_reserve = self._reserve_pages(candidates)
                if self.kv_pool.num_free_pages >= new_reserve + self._append_cost(request):
                    coordinated(self.group, "admit buffers", lambda: request.attach_kv_pool(self.kv_pool))
                    self.add_request(request)
                    self._requests[request.request_id] = request
                    self.prefilling_request = request
                    pending_requests.pop(0)
                    reserve = new_reserve
                else:
                    pending_needs_space = True
                    self.metrics["admission_waits"] += 1
            elif not self._requests:
                raise MemoryError("Request cannot fit original compact peak reservation")
            else:
                self.metrics["admission_waits"] += 1

        ordered = sorted(self._requests.values(), key=lambda r: (
            self._last_served.get(r.request_id, -1), r.request_id))
        pressure = pending_needs_space
        if not self._transfers:
            for request in ordered:
                ptr = request.kv_cache_ptr
                if (request.status == Request.ReqExecType.VERIFY
                        and isinstance(ptr, CompactCachePtr) and ptr.phase == "hot"):
                    if ptr.cold_count + reserve + 1 <= self.kv_pool.num_free_pages:
                        self._submit_transfer(request, "restore")
                    else:
                        pressure = True
                    break
        ready = [r for r in ordered if r.status != Request.ReqExecType.STALL
                 and r.request_id not in self._transfers
                 and not (r.status == Request.ReqExecType.VERIFY
                          and isinstance(r.kv_cache_ptr, CompactCachePtr)
                          and r.kv_cache_ptr.phase == "hot")]
        pressure |= reserve + sum(self._append_cost(r) for r in ready) > self.kv_pool.num_free_pages
        if pressure and not self._transfers:
            victim = self._pressure_victim()
            if victim is not None:
                self._submit_transfer(victim, "repack")

        # A submitted offload already owns its replacement hot pages. Its
        # locked source pages will be returned at commit, so ready work may
        # temporarily use part of the reserve without losing future workspace.
        effective_reserve = reserve
        if self._transfers:
            job = next(iter(self._transfers.values()))
            if job.direction == "repack":
                effective_reserve = max(0, reserve - job.source_pages)

        chosen, alloc, drafts, verifies, queries = [], 0, 0, 0, 0
        for request in ordered:
            ptr = request.kv_cache_ptr
            if (request.status == Request.ReqExecType.STALL
                    or request.request_id in self._transfers
                    or (request.status == Request.ReqExecType.VERIFY
                        and isinstance(ptr, CompactCachePtr) and ptr.phase == "hot")):
                continue
            append = self._append_cost(request)
            draft = ptr.len_loading_indices + 1 if request.status == Request.ReqExecType.DRAFT else 0
            verify = ptr.len_kv_cache + 1 if request.status == Request.ReqExecType.VERIFY else 0
            query = self.spec_stride + 1 - request.initial_idx if verify else append
            if (alloc + append + effective_reserve <= self.kv_pool.num_free_pages
                    and drafts + draft < CompactCachePtr.MAX_TOTAL_DRAFT_KV_LEN
                    and verifies + verify < CompactCachePtr.MAX_TOTAL_VERIFY_KV_LEN
                    and queries + query < CompactCachePtr._MAX_NNZ):
                chosen.append(request)
                alloc, drafts, verifies, queries = alloc + append, drafts + draft, verifies + verify, queries + query
        ids = [r.request_id for r in chosen]
        self._check_ready_batch(ids)
        self.active_requests = chosen
        chosen_ids = set(ids)
        self.offload_requests = [r for r in self._requests.values() if r.request_id not in chosen_ids]
        self.metrics["capacity_waits"] += len(self.offload_requests)
        self.metrics["max_live_requests"] = max(self.metrics["max_live_requests"], len(self._requests))
        has_stall = getattr(self, "async_on", False) and any(
            r.status == Request.ReqExecType.STALL for r in self._requests.values())
        has_retirement = bool(getattr(self, "_retired", []))
        if not chosen and not self._transfers and not has_stall and not has_retirement:
            raise MemoryError("No ready request fits native metadata and compaction reserve")

    def wait_for_ready_requests(self):
        if not self._transfers:
            raise RuntimeError("Empty ready batch without an in-flight KV job")
        job = next(iter(self._transfers.values()))
        started = time.perf_counter()
        timing = {}

        def wait():
            wait_started = time.perf_counter()
            job.transfer.wait()
            timing["event_wait_wall_seconds"] = time.perf_counter() - wait_started

        coordinated(self.group, "idle KV wait", wait, timing=timing)
        for stage, seconds in timing.items():
            key = f"{job.direction}_{stage}"
            self.metrics[key] = self.metrics.get(key, 0.) + seconds
        # This includes rank coordination and is explicitly reported as an
        # idle blocking section, rather than a pure GPU transfer duration.
        key = f"{job.direction}_idle_wait_wall_seconds"
        self.metrics[key] = self.metrics.get(key, 0.) + time.perf_counter() - started
        self.metrics["transfer_idle_waits"] += 1
        self._poll_transfers()

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        report = super().transfer_timing_snapshot(elapsed_seconds)
        report.update(mode="on_demand_async_kv", cpu_scheduling="synchronous",
                      transfers_inflight=len(self._transfers),
                      transfer_idle_waits=self.metrics["transfer_idle_waits"],
                      commit_scope="nonblocking commit after all TP ranks report event completion")
        return report

    def prepare_forward_inputs(self):
        if self._transfers:
            self.metrics["forwards_with_transfer_inflight"] += 1
        # Restore readiness is enforced before metadata preparation.
        return SpecScheduler.prepare_forward_inputs(self)

    def update_requests(self, *args, **kwargs):
        completed = SpecScheduler.update_requests(self, *args, **kwargs)
        for request in completed:
            del self._requests[request.request_id]
        for request in self.active_requests:
            ptr = request.kv_cache_ptr
            if isinstance(ptr, CompactCachePtr) and ptr.needs_pack:
                ptr.keep_full_resident()
                self.metrics["resident_updates"] += 1
        return completed
