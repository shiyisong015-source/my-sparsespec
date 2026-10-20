"""Token-budgeted compact serving with eviction and draft-preserving replay.

Reuse native attention, sampling/rejection and the one-round CPU pipeline.
Only the new explicit compact_dynamic_admission route uses this policy.
"""
from collections import OrderedDict
import hashlib
import math
from pathlib import Path

import torch

from serve.request.request import Request
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.scheduler.async_cpu_compact_scheduler import (
    AsyncCPUCompactSpecScheduler, RetiredStorage,
)
from serve.scheduler.compact_scheduler import coordinated


class DynamicCompactSpecScheduler(AsyncCPUCompactSpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        budget = kwargs.get("max_num_batched_tokens")
        if type(budget) is not int or not kwargs["spec_stride"] + 1 <= budget < FullCachePtr._MAX_NNZ:
            raise ValueError("Dynamic compact requires k+1 <= max_num_batched_tokens < 32768")
        early = kwargs.get("compact_early_verify_length")
        if early is not None and (type(early) is not int or not 1 <= early <= kwargs["spec_stride"]):
            raise ValueError("early verify length must be between 1 and k")
        fraction = kwargs.get("compact_cpu_admit_fraction", 1.)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("CPU admission fraction must be in (0,1]")
        super().__init__(kv_pool, model, **kwargs)
        self.token_budget = budget
        self.early_verify_length = early
        self.cpu_admit_limit = self.cpu_capacity * fraction
        self._cold_lru = OrderedDict()
        self._preempt_requested = OrderedDict()
        self._waiting_replays = OrderedDict()
        self._sample_rows = None
        self.metrics.update(dynamic_admission=True, scheduled_token_peak=0,
                            prefill_request_peak=0, prefill_query_tokens=0,
                            replay_query_tokens=0, replay_starts=0, replay_completions=0,
                            preemptions=0, cpu_evictions=0, cpu_evicted_pages=0,
                            early_verifies=0, cpu_watermark_waits=0,
                            token_budget_waits=0, metadata_waits=0,
                            kv_capacity_waits=0, pure_replay_forwards=0,
                            recovery_queue_peak=0, replay_admissions=0,
                            recovery_memory_waits=0, recovery_preemption_waits=0,
                            recovery_idle_reclaims=0, known_growth_reserve_peak=0)
        self.metrics["dynamic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def run(self, pending_requests):
        pending_requests = list(pending_requests)
        # A solo full-context verification must be possible even after all
        # other requests have been preempted. This is not aggregate reservation.
        if any(r.len_oracle_ctx is None or
               r.len_oracle_ctx + self.spec_stride + 1 > self.kv_pool.capacity
               for r in pending_requests):
            raise ValueError("A request's full verification context exceeds the GPU KV pool")
        torch.cuda.reset_peak_memory_stats(self.kv_pool.device)
        return super().run(pending_requests)

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        self.metrics["runtime_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.kv_pool.device)
        self.metrics["runtime_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.kv_pool.device)
        self.metrics["max_num_batched_tokens"] = self.token_budget
        return super().transfer_timing_snapshot(elapsed_seconds)

    def _runtime_reserve(self):
        # Actual current sparse views, not every request's final output cap.
        reserve = max((r.kv_cache_ptr.len_loading_indices + 1
                       for r in self._requests.values()
                       if isinstance(r.kv_cache_ptr, CompactCachePtr)
                       and r.kv_cache_ptr.phase == "resident"), default=0)
        if self._transfers:
            job = next(iter(self._transfers.values()))
            if job.direction == "repack":
                reserve = max(0, reserve - job.source_pages)
        return reserve

    def _poll_transfers(self):
        before = {key: job.direction for key, job in self._transfers.items()}
        super()._poll_transfers()
        for identity in before.keys() - self._transfers.keys():
            request = self._requests.get(identity)
            if request is not None and request.kv_cache_ptr.cold_count:
                self._cold_lru.pop(identity, None)
                self._cold_lru[identity] = None

    def _ensure_cpu_space(self, pages, protected=()):
        if pages > self.cpu_capacity:
            return False
        for identity in list(self._cold_lru):
            if self._current_cpu() + pages <= self.cpu_capacity:
                break
            request = self._requests.get(identity)
            if request is None or not getattr(request.kv_cache_ptr, "cold_count", 0):
                self._cold_lru.pop(identity, None)
                continue
            if ((identity in protected and request.status == Request.ReqExecType.VERIFY)
                    or identity in self._transfers
                    or request.status == Request.ReqExecType.STALL):
                continue
            count = request.kv_cache_ptr.evict_cold()
            self._cold_lru.pop(identity, None)
            self.metrics["cpu_evictions"] += 1
            self.metrics["cpu_evicted_pages"] += count
        return self._current_cpu() + pages <= self.cpu_capacity

    def _try_spill(self, protected):
        if self._transfers:
            return False
        candidates = []
        for request in self._requests.values():
            ptr = request.kv_cache_ptr
            if (request.request_id in protected or request.request_id in self._preempt_requested
                    or request.status == Request.ReqExecType.STALL
                    or not isinstance(ptr, CompactCachePtr) or ptr.phase != "resident"):
                continue
            hot, cold = ptr.len_loading_indices, len(ptr.indices) - ptr.len_loading_indices
            if cold > 0 and hot <= self.kv_pool.num_free_pages and cold <= self.cpu_capacity:
                distance = self.spec_stride - request.cur_spec_idx
                candidates.append((request.status == Request.ReqExecType.DRAFT,
                                   distance, cold, -request.request_id, request))
        for entry in sorted(candidates, key=lambda x: x[:-1], reverse=True):
            request, cold = entry[-1], entry[2]
            if self._ensure_cpu_space(cold, protected):
                self._submit_transfer(request, "repack")
                return True
        return False

    def _queue_preemption(self, protected):
        if self._preempt_requested:
            return True
        for request in sorted(self._requests.values(), key=lambda r: r.request_id, reverse=True):
            if (request.request_id in protected or request.request_id in self._transfers
                    or request.status == Request.ReqExecType.STALL
                    or not request.kv_cache_ptr.indices):
                continue
            self._preempt_requested[request.request_id] = None
            return True
        return False

    def _make_room(self, pages, protected):
        if pages <= self.kv_pool.num_free_pages:
            return True
        retiring = sum(len(item.ptr.owned_pages) if isinstance(item.ptr, CompactCachePtr)
                       else len(item.ptr.indices) for item in self._retired if item.ptr is not None)
        if self._transfers or self.kv_pool.num_free_pages + retiring >= pages:
            self.metrics["kv_capacity_waits"] += 1
            return False
        protected = set(protected)
        if not self._try_spill(protected):
            self._queue_preemption(protected)
        self.metrics["kv_capacity_waits"] += 1
        return False

    def _early_verify(self, request, pressure=False):
        if request.status != Request.ReqExecType.DRAFT or not request.len_draft_probs:
            return
        if not pressure and (self.early_verify_length is None
                             or request.len_draft_probs < self.early_verify_length):
            return
        # Keep native window math and rejection assertions consistent with
        # the actual retained draft count, including the pending async row.
        request.initial_idx = self.spec_stride - request.len_draft_probs
        request.cur_spec_idx = self.spec_stride
        request.status = Request.ReqExecType.VERIFY
        self.metrics["early_verifies"] += 1

    def _cost(self, request, remaining):
        ptr = request.kv_cache_ptr
        if request.status == Request.ReqExecType.VERIFY:
            query = request.len_draft_probs + 1
            return query, query if request.rebuild_verify else 1
        if request.status == Request.ReqExecType.DRAFT:
            return 1, 1
        target = request.replay_target if request.replaying else request.len_token_ids
        query = min(target - request.len_kv_cache, self._args["chunk_size"], remaining)
        return query, query

    def _eligible(self, request):
        return (request.status != Request.ReqExecType.STALL
                and request.request_id not in self._transfers
                and request.request_id not in self._preempt_requested)

    @staticmethod
    def _needs_restore(request):
        return (request.status == Request.ReqExecType.VERIFY
                and isinstance(request.kv_cache_ptr, CompactCachePtr)
                and request.kv_cache_ptr.phase == "hot")

    @staticmethod
    def _known_growth(request):
        """Pages to finish known input/replay, excluding future generated output.

        Recovery admission protects the entire known prefix and retained verify
        window, even though only a chunk is allocated this round. Otherwise a
        resumed prefix can repeatedly consume the space needed by older work.
        """
        if request.replaying:
            target = (request.len_input if request.replay_resume_status == Request.ReqExecType.NORMAL
                      else request.replay_target + request.len_draft_probs + 1)
            return max(0, target - request.len_kv_cache)
        if request.status == Request.ReqExecType.NORMAL:
            return max(0, request.len_input - request.len_kv_cache)
        if request.rebuild_verify:
            return request.len_draft_probs + 1
        return 0

    def admit_requests(self, pending_requests):
        self._poll_retired()
        self._output_keepalive = [(output, event) for output, event in self._output_keepalive
                                  if not event.query()]
        self._poll_transfers()
        for request in self._requests.values():
            request.scheduled_tokens = None
            if not request.replaying:
                self._early_verify(request)
            if self._needs_restore(request) and request.kv_cache_ptr.cold_missing:
                self._preempt_requested[request.request_id] = None
        for identity in list(self._waiting_replays):
            if identity not in self._requests or not self._requests[identity].replaying:
                self._waiting_replays.pop(identity)
        ordered = sorted(self._requests.values(), key=lambda r: r.request_id)
        waiting = [r for r in ordered if r.request_id in self._waiting_replays
                   or self._needs_restore(r)]
        waiting_ids = {r.request_id for r in waiting}
        running = [r for r in ordered if r.request_id not in waiting_ids]
        self.metrics["recovery_queue_peak"] = max(self.metrics["recovery_queue_peak"], len(waiting))
        # Select a spill before compute-batch membership protects every ready
        # candidate and consumes the last replacement workspace.
        growing = [r for r in running if self._eligible(r)]
        growth = min(self.token_budget, sum(self._cost(r, self.token_budget)[1] for r in growing))
        if not self._transfers and self._runtime_reserve() + growth > self.kv_pool.num_free_pages:
            protected = {growing[0].request_id} if growing else set()
            self._try_spill(protected)
        chosen, pages, queries, drafts, verifies, indices = [], 0, 0, 0, 0, 0

        def choose(request, *, reclaim=True):
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
            required = pages + append + self._runtime_reserve()
            # A younger running/replaying request must never evict older work
            # to advance its own next chunk. Newly resumed work only uses free
            # pages; it cannot trigger reclamation during waiting admission.
            protected = {r.request_id for r in ordered if r.request_id <= request.request_id}
            protected.update(r.request_id for r in chosen)
            fits = (self._make_room(required, protected) if reclaim
                    else required <= self.kv_pool.num_free_pages)
            if not fits:
                # Workspace is advisory when no spill/preemption can use it.
                # A solo resident request must not stall below its valid cap.
                if (not reclaim or pages + append > self.kv_pool.num_free_pages or self._transfers
                        or self._preempt_requested or self._retired):
                    self._early_verify(request, pressure=True)
                    return False
            request.scheduled_tokens = query
            chosen.append(request)
            pages, queries = pages + append, queries + query
            drafts, verifies, indices = drafts + draft, verifies + verify, indices + full
            return True

        for request in running:
            choose(request)
        # Running work gets first use of the round, as in vLLM V1. Waiting
        # recovery consumes residual capacity without evicting that work. A
        # request joins running after its FIRST replay chunk, not after its
        # complete generation; several partial replays can run together.
        future_growth = sum(max(0, self._known_growth(r) -
                                (r.scheduled_tokens or 0))
                            for r in running)
        future_growth += sum(job.direction == "restore" for job in self._transfers.values())
        recovery_blocked = False
        if self._preempt_requested:
            self.metrics["recovery_preemption_waits"] += int(bool(waiting))
        else:
            for request in waiting:
                if queries >= self.token_budget:
                    break
                if not self._eligible(request):
                    continue
                restoring = self._needs_restore(request)
                if restoring and self._transfers:
                    continue
                needed = (request.kv_cache_ptr.cold_count + 1 if restoring
                          else self._known_growth(request))
                if pages + future_growth + needed + self._runtime_reserve() > self.kv_pool.num_free_pages:
                    self.metrics["recovery_memory_waits"] += 1
                    recovery_blocked = True
                    break
                if restoring:
                    self._submit_transfer(request, "restore")
                    future_growth += 1  # The restored Verify still appends one page.
                elif choose(request, reclaim=False):
                    self._waiting_replays.pop(request.request_id)
                    self.metrics["replay_admissions"] += 1
                    future_growth += max(0, needed - request.scheduled_tokens)
                else:
                    recovery_blocked = True
                    break
        # Existing work receives its budget before admitting fresh prompts.
        while (pending_requests and len(self._requests) < self.max_batch_size
               and queries < self.token_budget and not self._preempt_requested
               and not recovery_blocked):
            if self._current_cpu() >= self.cpu_admit_limit:
                self.metrics["cpu_watermark_waits"] += 1
                break
            request = pending_requests[0]
            query = min(request.len_input, self._args["chunk_size"], self.token_budget - queries)
            if pages + query + future_growth + self._runtime_reserve() > self.kv_pool.num_free_pages:
                self.metrics["admission_waits"] += 1
                break
            coordinated(self.group, "dynamic admit buffers", lambda: request.attach_kv_pool(self.kv_pool))
            self.active_requests = list(self._requests.values())
            self.add_request(request)
            self._requests[request.request_id] = request
            pending_requests.pop(0)
            if not choose(request, reclaim=False):
                break
            future_growth += max(0, self._known_growth(request) - request.scheduled_tokens)

        stalls = sorted((r for r in self._requests.values()
                         if r.status == Request.ReqExecType.STALL), key=lambda r: r.prev_idx)
        if (not chosen and not stalls and waiting and not self._transfers
                and not self._retired and not self._preempt_requested):
            # Hot KV can remain owned by waiting Verify requests. If nothing
            # can run, reclaim enough for the oldest recovery to make progress;
            # ordinary waiting admission never preempts running work.
            request = waiting[0]
            needed = (request.kv_cache_ptr.cold_count + 1 if self._needs_restore(request)
                      else self._known_growth(request))
            protected = {r.request_id for r in ordered if r.request_id <= request.request_id}
            self._make_room(needed + self._runtime_reserve(), protected)
            self.metrics["recovery_idle_reclaims"] += 1
        self.metrics["known_growth_reserve_peak"] = max(
            self.metrics["known_growth_reserve_peak"], future_growth)

        self.active_requests = chosen + stalls
        active_ids = {r.request_id for r in self.active_requests}
        self.offload_requests = [r for r in self._requests.values() if r.request_id not in active_ids]
        self._check_ready_batch([r.request_id for r in chosen])
        self.metrics["capacity_waits"] += len(self.offload_requests)
        self.metrics["max_live_requests"] = max(self.metrics["max_live_requests"], len(self._requests))
        self.metrics["scheduled_token_peak"] = max(self.metrics["scheduled_token_peak"], queries)
        prefill_count = sum(r.status == Request.ReqExecType.NORMAL and not r.replaying for r in chosen)
        self.metrics["prefill_request_peak"] = max(self.metrics["prefill_request_peak"], prefill_count)
        self.metrics["replay_query_tokens"] += sum(r.scheduled_tokens for r in chosen if r.replaying)
        self.metrics["prefill_query_tokens"] += sum(r.scheduled_tokens for r in chosen
                                                   if r.status == Request.ReqExecType.NORMAL and not r.replaying)
        if (not self.active_requests and not self._transfers and not self._retired
                and not self._preempt_requested):
            raise MemoryError("No dynamic request fits token, KV or metadata capacity")

    def prepare_forward_inputs(self):
        host, tensors = super().prepare_forward_inputs()
        if host is not None:
            self._sample_rows = [pos for i, r in enumerate(self.active_requests) if not r.replaying
                                 for pos in range(host["token_indptr"][i], host["token_indptr"][i + 1])]
        return host, tensors

    def _sample(self, logits, count):
        rows = self._sample_rows
        if len(rows) == count:
            return super()._sample(logits, count)
        tokens = torch.zeros(count, dtype=torch.int32, device=logits.device)
        if not rows:
            self.metrics["pure_replay_forwards"] += 1
            return tokens, torch.empty((count, 0), dtype=torch.float32, device=logits.device)
        indices = torch.tensor(rows, dtype=torch.long, device=logits.device)
        tp = self._args["tp_ranks"]
        # Gather token rows within each TP vocabulary shard. The native
        # sampler consumes rank-major gathered logits, not a canonical matrix.
        selected = logits.view(tp, logits.shape[0], -1).index_select(1, indices).contiguous()
        sampled, selected_probs = super()._sample(selected.view(len(rows), -1), len(rows))
        probs = torch.zeros((count, selected_probs.shape[1]), dtype=torch.float32, device=logits.device)
        tokens.index_copy_(0, indices, sampled)
        probs.index_copy_(0, indices, selected_probs)
        return tokens, probs

    def has_pending_preemption(self):
        return bool(self._preempt_requested)

    def _apply_preemptions(self):
        for identity in list(self._preempt_requested):
            request = self._requests.get(identity)
            if request is None:
                self._preempt_requested.pop(identity)
                continue
            if request.status == Request.ReqExecType.STALL or identity in self._transfers:
                continue
            ptr = request.kv_cache_ptr
            if self.debug_checks and not hasattr(request, "_replay_draft_snapshot"):
                request._replay_draft_snapshot = request.draft_probs[:request.len_draft_probs].clone()
                request._replay_tokens_snapshot = tuple(request.token_ids)
            if not request.replaying:
                request.begin_replay(ptr.len_kv_cache)
            self._retired.append(RetiredStorage(ptr, None, self._compute_fences.get(identity)))
            request.kv_cache_ptr = FullCachePtr(self.kv_pool)
            self._waiting_replays[identity] = None
            self._waiting_replays.move_to_end(identity, last=False)
            self._cold_lru.pop(identity, None)
            self._preempt_requested.pop(identity)
            self.metrics["preemptions"] += 1
            self.metrics["replay_starts"] += 1
        self._finish_replays()

    def _finish_replays(self):
        for request in self._requests.values():
            if not request.replaying or request.len_kv_cache != request.replay_target:
                continue
            if hasattr(request, "_replay_draft_snapshot"):
                if (tuple(request.token_ids) != request._replay_tokens_snapshot or
                        not torch.equal(request.draft_probs[:request.len_draft_probs],
                                        request._replay_draft_snapshot)):
                    raise RuntimeError("Replay changed retained draft tokens or original probabilities")
                del request._replay_draft_snapshot, request._replay_tokens_snapshot
                self.metrics["replay_draft_preservation_checks"] = self.metrics.get(
                    "replay_draft_preservation_checks", 0) + 1
            resume = request.replay_resume_status
            self._waiting_replays.pop(request.request_id, None)
            request.replay_target = request.replay_resume_status = None
            request.status = resume
            if resume == Request.ReqExecType.VERIFY:
                request.initial_idx = self.spec_stride - request.len_draft_probs
                request.cur_spec_idx = self.spec_stride
                request.rebuild_verify = True
            elif resume == Request.ReqExecType.DRAFT:
                request.kv_cache_ptr, _ = CompactCachePtr.from_kv_cache(request.kv_cache_ptr)
                request.cur_spec_idx = request.initial_idx = 0
            self.metrics["replay_completions"] += 1

    def update_requests(self, *args, **kwargs):
        completed = super().update_requests(*args, **kwargs)
        self._apply_preemptions()
        self._finish_replays()
        for identity in list(self._waiting_replays):
            if identity not in self._requests:
                self._waiting_replays.pop(identity)
        for identity in list(self._cold_lru):
            if identity not in self._requests or not getattr(self._requests[identity].kv_cache_ptr, "cold_count", 0):
                self._cold_lru.pop(identity, None)
        return completed
