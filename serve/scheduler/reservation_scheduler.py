"""VeriCache-style BW/HBM reservations on the native compact KV layout.

The calendar predicts future reload/verify windows; real page availability,
query/metadata bounds and TP-wide CUDA completion still gate execution.
Unlike the on-demand policy, every selected resident cold cache is repacked
before its next sparse cycle. CPU eviction/replay and pressure early-verify
are deliberately not part of this baseline. Ready transfers are queued on one
CUDA stream per rank so same-slot bookings progress during model submission.
The configured iteration duration remains the approximate forward-cycle clock;
only observed TP-wide events authorize layout publication and page release.
"""
import json
import math
from pathlib import Path

from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.request import Request
from serve.scheduler.compact_scheduler import coordinated
from serve.scheduler.dynamic_compact_scheduler import DynamicCompactSpecScheduler
from serve.scheduler.reservation_table import ReservationTable


class ReservationCompactSpecScheduler(DynamicCompactSpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        bandwidth = kwargs.get("reservation_bandwidth_gbps")
        iteration_ms = kwargs.get("reservation_iteration_ms")
        if bandwidth is None or iteration_ms is None:
            raise ValueError("Reservations require measured per-rank bandwidth and iteration time")
        if kwargs.get("compact_early_verify_length") is not None:
            raise ValueError("Reservation deadlines replace compact_early_verify_length")
        horizon = kwargs.get("reservation_horizon", 64)
        target = kwargs.get("reservation_target_length")
        target = kwargs["spec_stride"] if target is None else target
        if type(target) is not int or not 1 <= target <= kwargs["spec_stride"]:
            raise ValueError("Require 1 <= reservation_target_length <= spec_stride")
        # Validate before constructing any CUDA objects.
        table = ReservationTable(horizon, bandwidth * 1e9, iteration_ms / 1000.)
        super().__init__(kv_pool, model, **kwargs)
        self.reservation_transfer_queue = kwargs.get("reservation_transfer_queue", True)
        self.reservation_table, self.reservation_target = table, target
        self._reservation_wait_since = {}
        self._reservation_selection = {}
        self._reservation_log = (Path(self._result_dir) / f"reservations.rank{self.rank}.jsonl"
                                 if self._result_dir else None)
        if self._reservation_log:
            self._reservation_log.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.update(reservation_enabled=True, reservation_admissions=0,
                            reservation_waits=0, reservation_deadline_misses=0,
                            reservation_replans=0, reservation_reload_starts=0,
                            reservation_cpu_waits=0, reservation_memory_waits=0,
                            reservation_empty_windows=0, reservation_repack_waits=0,
                            reservation_drafts_during_own_reload=0,
                            reservation_release_plans=0, reservation_release_commits=0,
                            reservation_release_delays=0, reservation_dependency_waits=0,
                            reservation_dependent_admissions=0, reservation_dependency_replans=0,
                            reservation_capacity_replans=0,
                            reservation_target_length=target, reservation_horizon=horizon,
                            reservation_bandwidth_gbps=bandwidth,
                            reservation_iteration_ms=iteration_ms)

    def _reservation_event(self, event, request_id, **fields):
        if self._reservation_log:
            if request_id < 0:
                request_id, fields = -request_id - 1, dict(fields, reservation_kind="repack")
            with self._reservation_log.open("a") as out:
                out.write(json.dumps(dict(iteration=self.step_id, event=event,
                                          request_id=request_id, **fields)) + "\n")

    def _cancel_reservation(self, identity, reason):
        if identity in self.reservation_table.reservations:
            invalidated = self.reservation_table.cancel(identity)
            for key in invalidated:
                self._reservation_event("cancel", key, reason=reason if key == identity
                                        else "release_dependency_cancelled", source_key=identity)
                if key != identity:
                    self.metrics["reservation_dependency_replans"] += 1
                    self._reservation_wait_since.setdefault(key, self.step_id)
        self._reservation_wait_since.pop(identity, None)

    def _poll_transfers(self):
        repacks = {key: job for key, job in self._transfers.items() if job.direction == "repack"}
        super()._poll_transfers()
        for identity, job in repacks.items():
            if identity in self._transfers:
                continue
            booking = self.reservation_table.reservations.get(-identity - 1)
            if booking is not None and booking.release_event is not None:
                event = self.reservation_table.releases[booking.release_event]
                if self.reservation_table.confirm_release(event.event_id):
                    self.metrics["reservation_release_commits"] += 1
                    self._reservation_event("release_commit", identity, event_id=event.event_id,
                                            forecast_release_pages=event.pages,
                                            net_release_pages=job.cold_pages,
                                            actual_source_pages_freed=job.source_pages)

    def _growth_pages(self):
        total = 0
        for request in self._requests.values():
            if request.status == Request.ReqExecType.NORMAL:
                total += max(0, request.len_input - request.len_kv_cache) + self.spec_stride + 1
            elif request.status == Request.ReqExecType.VERIFY:
                total += 1
            else:
                total += max(0, self.spec_stride - request.len_draft_probs) + 1
        return total

    def _pack_workspace(self):
        workspace = 0
        for request in self._requests.values():
            job = self._transfers.get(request.request_id)
            if job is not None and job.direction == "repack":
                continue  # The new hot destinations are already allocated.
            ptr = request.kv_cache_ptr
            length = max(request.len_input, ptr.len_kv_cache) + self.spec_stride + 1
            hot = min(length, max(self.spec_cache_args["num_min_budget"], math.ceil(
                length * self.spec_cache_args["budget_ratio"])))
            if isinstance(ptr, CompactCachePtr) and ptr.phase == "resident":
                hot = max(hot, ptr.len_loading_indices)
            workspace = max(workspace, hot)
        return workspace

    def _reload_headroom(self):
        # Reserve room for at least one unmaterialized cold cache even if its
        # calendar admission failed. Otherwise hot-only admission can deadlock.
        cold = max((r.kv_cache_ptr.cold_count for r in self._requests.values()
                    if isinstance(r.kv_cache_ptr, CompactCachePtr)
                    and r.kv_cache_ptr.phase == "hot"
                    and r.request_id not in self.reservation_table.reservations
                    and r.request_id not in self._transfers), default=0)
        return cold

    def _projected_resident(self):
        return (self.kv_pool.capacity - self.kv_pool.num_free_pages
                + self._growth_pages() + self._pack_workspace())

    def _revalidate_future_capacity(self):
        # Materialized targets remain resident past their nominal deadline.
        # Correct that prediction by rebooking younger unstarted reloads;
        # never revoke pages used by an in-flight transfer or model kernel.
        table = self.reservation_table
        while True:
            bad = {w["iteration"] for w in table.memory_profile(
                self._projected_resident(), self.kv_pool.capacity) if w["projected_free_pages"] < 0}
            candidates = [r for r in table.reservations.values()
                          if r.request_id >= 0 and r.memory_pages and not r.materialized
                          and any(r.start <= tick <= r.deadline for tick in bad)]
            if not bad or not candidates:
                return
            victim = max(candidates, key=lambda r: (r.deadline, r.request_id))
            self._cancel_reservation(victim.request_id, "capacity_forecast_changed")
            self.metrics["reservation_capacity_replans"] += 1

    def _ensure_cpu_space(self, pages, protected=()):
        # Preserve CPU sources of booked reloads. No implicit LRU/recompute.
        return self._current_cpu() + pages <= self.cpu_capacity

    def _early_verify(self, request, pressure=False):
        reservation = self.reservation_table.reservations.get(request.request_id)
        if (reservation is not None and self.step_id >= reservation.deadline
                and request.status == Request.ReqExecType.DRAFT and request.len_draft_probs):
            request.initial_idx = self.spec_stride - request.len_draft_probs
            request.cur_spec_idx = self.spec_stride
            request.status = Request.ReqExecType.VERIFY

    def _eligible(self, request):
        if request.status == Request.ReqExecType.STALL:
            return False
        if request.status == Request.ReqExecType.NORMAL:
            return True
        ptr = request.kv_cache_ptr
        reservation = self.reservation_table.reservations.get(request.request_id)
        if reservation is None:
            return False
        job = self._transfers.get(request.request_id)
        if job is not None:
            # Restore writes fresh cold pages; Draft reads the unchanged hot
            # view. Repack relocates hot pages, so its producer must park.
            return job.direction == "restore" and request.status == Request.ReqExecType.DRAFT
        if request.status == Request.ReqExecType.VERIFY:
            return (self.step_id >= reservation.deadline
                    and ptr.phase != "hot" and ptr._pending is None)
        return True

    def _reconcile_reservations(self):
        table = self.reservation_table
        for identity in list(table.reservations):
            if identity < 0:
                # Negative calendar keys are link-only repack reservations.
                owner = -identity - 1
                request = self._requests.get(owner)
                reservation = table.reservations[identity]
                if request is None or request.kv_cache_ptr.phase != "resident":
                    self._cancel_reservation(identity, "repack_committed")
                elif self.step_id > reservation.deadline and owner not in self._transfers:
                    self._cancel_reservation(identity, "repack_unstarted_overdue")
                    self.metrics["reservation_replans"] += 1
                elif reservation.release_event is not None:
                    event = table.releases[reservation.release_event]
                    if not event.committed and event.available_at <= self.step_id:
                        old = event.available_at
                        invalidated = table.defer_release(event.event_id, self.step_id + 1)
                        self.metrics["reservation_release_delays"] += 1
                        for key in invalidated:
                            self.metrics["reservation_dependency_replans"] += 1
                            self._reservation_wait_since.setdefault(key, self.step_id)
                            self._reservation_event("cancel", key, reason="release_dependency_delayed",
                                                    event_id=event.event_id)
                        self._reservation_event("release_delay", owner, event_id=event.event_id,
                                                old_available_at=old, available_at=event.available_at)
                continue
            if identity not in table.reservations:
                continue  # An earlier source cancellation invalidated it.
            request = self._requests.get(identity)
            if request is None:
                self._cancel_reservation(identity, "completed")
                self._reservation_selection.pop(identity, None)
                continue
            reservation = table.reservations[identity]
            ptr = request.kv_cache_ptr
            selection = (id(ptr), ptr._resident_selected_at)
            if (ptr.phase == "resident" and request.status == Request.ReqExecType.DRAFT
                    and self._reservation_selection.get(identity) != selection):
                self._reservation_selection[identity] = selection
                self._cancel_reservation(identity, "new_selection")
                continue
            if self.step_id > reservation.deadline and request.status != Request.ReqExecType.STALL:
                if not reservation.missed:
                    reservation.missed = True
                    self.metrics["reservation_deadline_misses"] += 1
                    self._reservation_event("deadline_miss", identity, deadline=reservation.deadline)
                if not reservation.materialized:
                    self._cancel_reservation(identity, "unstarted_overdue")
                    self.metrics["reservation_replans"] += 1

    def _place_reservations(self):
        table = self.reservation_table
        self._reconcile_reservations()
        self._plan_repacks()
        self._revalidate_future_capacity()
        candidates = [r for r in self._requests.values()
                      if isinstance(r.kv_cache_ptr, CompactCachePtr)
                      and r.status in (Request.ReqExecType.DRAFT, Request.ReqExecType.VERIFY)
                      and r.request_id not in table.reservations
                      and r.request_id not in self._transfers
                      and (r.kv_cache_ptr.phase in ("full", "hot")
                           or (r.kv_cache_ptr.phase == "resident"
                               and len(r.kv_cache_ptr.indices) == r.kv_cache_ptr.len_loading_indices))]
        for request in candidates:
            self._reservation_wait_since.setdefault(request.request_id, self.step_id)
        candidates.sort(key=lambda r: (self._reservation_wait_since[r.request_id], r.request_id))
        baseline = self.kv_pool.capacity - self.kv_pool.num_free_pages
        baseline += self._growth_pages() + self._pack_workspace()
        for request in candidates:
            ptr = request.kv_cache_ptr
            cold = ptr.cold_count if ptr.phase == "hot" else 0
            reservation = table.reserve(
                request.request_id, transfer_bytes=cold * self.page_bytes, memory_pages=cold,
                anchor=max(1, self.reservation_target - request.len_draft_probs),
                resident_pages=baseline, capacity_pages=self.kv_pool.capacity)
            if reservation is None:
                self.metrics["reservation_waits"] += 1
                continue
            self._reservation_wait_since.pop(request.request_id, None)
            self.metrics["reservation_admissions"] += 1
            self.metrics["reservation_dependent_admissions"] += bool(reservation.dependencies)
            self._reservation_event("reserve", request.request_id, start=reservation.start,
                                    deadline=reservation.deadline, cold_pages=cold,
                                    transfer_bytes=reservation.transfer_bytes,
                                    dependencies=list(reservation.dependencies))

    def _plan_repacks(self):
        table = self.reservation_table
        planned_cpu = sum(r.transfer_bytes // self.page_bytes for r in table.reservations.values()
                          if r.request_id < 0 and not r.materialized)
        for request in sorted(self._requests.values(), key=lambda r: r.request_id):
            ptr = request.kv_cache_ptr
            if (not isinstance(ptr, CompactCachePtr) or ptr.phase != "resident"
                    or request.status != Request.ReqExecType.DRAFT
                    or request.request_id in table.reservations
                    or request.request_id in self._transfers
                    or -request.request_id - 1 in table.reservations):
                continue
            cold = len(ptr.indices) - ptr.len_loading_indices
            if cold <= 0:
                continue
            if not self._ensure_cpu_space(planned_cpu + cold):
                self.metrics["reservation_cpu_waits"] += 1
                continue
            key = -request.request_id - 1
            reservation = table.reserve(key, transfer_bytes=cold * self.page_bytes, memory_pages=0,
                                        anchor=1, resident_pages=0, capacity_pages=self.kv_pool.capacity)
            if reservation is None:
                self.metrics["reservation_waits"] += 1
                continue
            event = table.plan_release(key, pages=cold, available_at=reservation.deadline + 1)
            planned_cpu += cold
            self.metrics["reservation_release_plans"] += 1
            self._reservation_event("repack_reserve", key, start=reservation.start,
                                    deadline=reservation.deadline, cold_pages=cold,
                                    release_event=event.event_id, release_at=event.available_at)

    def _start_repack(self):
        if self._transfers and not getattr(self, "reservation_transfer_queue", True):
            return
        # Also supports direct callers; planning itself needs no free link.
        self._plan_repacks()
        bookings = sorted((r for r in self.reservation_table.reservations.values()
                           if r.request_id < 0), key=lambda r: (r.start, r.deadline, r.request_id))
        for reservation in bookings:
            if (reservation.materialized or reservation.start > self.step_id
                    or -reservation.request_id - 1 in self._transfers):
                continue
            request = self._requests[-reservation.request_id - 1]
            ptr = request.kv_cache_ptr
            cold = len(ptr.indices) - ptr.len_loading_indices
            if not self._ensure_cpu_space(cold):
                self.metrics["reservation_cpu_waits"] += 1
                continue
            if ptr.len_loading_indices > self.kv_pool.num_free_pages:
                self.metrics["reservation_repack_waits"] += 1
                continue
            self._submit_transfer(request, "repack")
            self.reservation_table.materialize(reservation.request_id, source_pages=len(ptr.indices))
            self._reservation_event("repack_start", request.request_id, cold_pages=cold,
                                    release_event=reservation.release_event,
                                    source_pages=len(ptr.indices))
            if not getattr(self, "reservation_transfer_queue", True):
                return

    def _start_reload(self):
        if self._transfers and not getattr(self, "reservation_transfer_queue", True):
            return
        table = self.reservation_table
        for reservation in sorted(table.reservations.values(), key=lambda r: (r.start, r.deadline, r.request_id)):
            # Allocations can invalidate later bookings in this snapshot.
            if (reservation.request_id not in table.reservations
                    or reservation.request_id < 0 or reservation.materialized
                    or reservation.start > self.step_id
                    or reservation.request_id in self._transfers):
                continue
            request = self._requests[reservation.request_id]
            ptr = request.kv_cache_ptr
            if not table.dependencies_ready(request.request_id):
                self.metrics["reservation_dependency_waits"] += 1
                continue
            if not reservation.memory_pages:
                table.materialize(request.request_id)
                continue
            # Allocation is whole-cache, not progressive DMA. Predictive
            # reservations never authorize allocating unavailable pages.
            required = ptr.cold_count + self._growth_pages() + self._pack_workspace()
            if required > self.kv_pool.num_free_pages:
                self.metrics["reservation_memory_waits"] += 1
                continue
            self._submit_transfer(request, "restore")
            table.materialize(request.request_id)
            self._revalidate_future_capacity()
            self.metrics["reservation_reload_starts"] += 1
            self._reservation_event("reload_start", request.request_id, deadline=reservation.deadline,
                                    cold_pages=ptr.cold_count, dependencies=list(reservation.dependencies))
            if not getattr(self, "reservation_transfer_queue", True):
                return

    def admit_requests(self, pending_requests):
        self._poll_retired()
        self._output_keepalive = [(output, event) for output, event in self._output_keepalive
                                  if not event.query()]
        self._poll_transfers()
        self.reservation_table.advance(self.step_id)
        for request in self._requests.values():
            request.scheduled_tokens = None
        self._place_reservations()
        for request in self._requests.values():
            self._early_verify(request)
        self._start_reload()
        self._start_repack()

        chosen, pages, queries, drafts, verifies, indices = [], 0, 0, 0, 0, 0

        def choose(request):
            nonlocal pages, queries, drafts, verifies, indices
            if not self._eligible(request):
                return False
            ptr = request.kv_cache_ptr
            query, append = self._cost(request, self.token_budget - queries)
            if query <= 0 or queries + query > self.token_budget:
                self.metrics["token_budget_waits"] += 1
                return False
            draft = ptr.len_loading_indices + append if request.status == Request.ReqExecType.DRAFT else 0
            verify = ptr.len_kv_cache + append if request.status == Request.ReqExecType.VERIFY else 0
            full = len(ptr.indices) + append
            if (drafts + draft >= CompactCachePtr.MAX_TOTAL_DRAFT_KV_LEN
                    or verifies + verify >= CompactCachePtr.MAX_TOTAL_VERIFY_KV_LEN
                    or indices + full > FullCachePtr._MAX_KV_CONTEXT_LEN):
                self.metrics["metadata_waits"] += 1
                return False
            reserve = self._pack_workspace()
            if pages + append + reserve > self.kv_pool.num_free_pages:
                self.metrics["kv_capacity_waits"] += 1
                return False
            request.scheduled_tokens = query
            chosen.append(request)
            if (request.status == Request.ReqExecType.DRAFT
                    and request.request_id in self._transfers
                    and self._transfers[request.request_id].direction == "restore"):
                self.metrics["reservation_drafts_during_own_reload"] += 1
            pages, queries = pages + append, queries + query
            drafts, verifies, indices = drafts + draft, verifies + verify, indices + full
            return True

        def order(request):
            reservation = self.reservation_table.reservations.get(request.request_id)
            due = (request.status == Request.ReqExecType.VERIFY and reservation is not None
                   and self.step_id >= reservation.deadline)
            return (not due, reservation.deadline if due else request.request_id, request.request_id)

        for request in sorted(self._requests.values(), key=order):
            choose(request)
        while (pending_requests and len(self._requests) < self.max_batch_size
               and queries < self.token_budget):
            if self._current_cpu() >= self.cpu_admit_limit:
                self.metrics["cpu_watermark_waits"] += 1
                break
            request = pending_requests[0]
            # Protect known input, one bounded speculation window and one
            # pack workspace. Never reserve the entire future output cap.
            hot = min(request.len_input + self.spec_stride + 1, max(
                self.spec_cache_args["num_min_budget"], math.ceil(
                    (request.len_input + self.spec_stride + 1) * self.spec_cache_args["budget_ratio"])))
            needed = request.len_input + self.spec_stride + 1 + self._growth_pages()
            needed += max(hot, self._pack_workspace())
            projected = self.kv_pool.capacity - self.kv_pool.num_free_pages + needed + self._reload_headroom()
            if (needed > self.kv_pool.num_free_pages
                    or not self.reservation_table.memory_feasible(projected, self.kv_pool.capacity)):
                self.metrics["admission_waits"] += 1
                break
            coordinated(self.group, "reservation admit buffers", lambda: request.attach_kv_pool(self.kv_pool))
            self.active_requests = list(self._requests.values())
            self.add_request(request)
            self._requests[request.request_id] = request
            pending_requests.pop(0)
            if not choose(request):
                break
        stalls = sorted((r for r in self._requests.values()
                         if r.status == Request.ReqExecType.STALL), key=lambda r: r.prev_idx)
        self.active_requests = chosen + stalls
        active_ids = {r.request_id for r in self.active_requests}
        self.offload_requests = [r for r in self._requests.values() if r.request_id not in active_ids]
        self._check_ready_batch([r.request_id for r in chosen])
        self.metrics["capacity_waits"] += len(self.offload_requests)
        self.metrics["max_live_requests"] = max(self.metrics["max_live_requests"], len(self._requests))
        self.metrics["scheduled_token_peak"] = max(self.metrics["scheduled_token_peak"], queries)
        self.metrics["prefill_request_peak"] = max(self.metrics["prefill_request_peak"], sum(
            r.status == Request.ReqExecType.NORMAL for r in chosen))
        self.metrics["prefill_query_tokens"] += sum(r.scheduled_tokens for r in chosen
                                                   if r.status == Request.ReqExecType.NORMAL)

    def wait_for_ready_requests(self):
        if self._transfers or self._retired:
            return super().wait_for_ready_requests()
        future = [tick for reservation in self.reservation_table.reservations.values()
                  for tick in (reservation.start, reservation.deadline) if tick > self.step_id]
        if future:
            next_tick = min(future)
            self.metrics["reservation_empty_windows"] += next_tick - self.step_id
            self.step_id = next_tick - 1
            return
        raise MemoryError("Reservation policy cannot progress: no feasible BW/HBM window, "
                          "CPU repack space, or native query/metadata capacity; no fallback applied")

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        report = super().transfer_timing_snapshot(elapsed_seconds)
        report.update(mode="vericache_reservation_compact", reservations=self.reservation_table.snapshot(
            self._projected_resident(), self.kv_pool.capacity),
                      reservation_scope="cold KV per rank; future repack releases with completion dependencies",
                      transfer_queue_enabled=self.reservation_transfer_queue,
                      deadline_scope="predicted iteration, not a guaranteed CUDA completion time")
        return report

    def update_requests(self, *args, **kwargs):
        completed = super().update_requests(*args, **kwargs)
        for request in completed:
            self._cancel_reservation(request.request_id, "completed")
            self._cancel_reservation(-request.request_id - 1, "completed")
            self._reservation_selection.pop(request.request_id, None)
        return completed
