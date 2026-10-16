"""One-round CPU/GPU pipeline on top of on-demand compact KV transfers.

Reuse native dummy-token patching and VERIFY->STALL result handling. Compute,
control and KV-transfer streams have explicit fences. Retired pages and raw
probability pointers outlive every consumer; the shared score snapshot is
overwritten only after the previous result's TopK has consumed it.
"""
import array
from dataclasses import dataclass
import json
import math
import os
import time

import torch
import torch.distributed as dist

from serve.request.request import Request
from serve.scheduler.async_compact_scheduler import AsyncCompactSpecScheduler
from serve.scheduler.compact_scheduler import coordinated
from serve.scheduler.spec_scheduler import SpecScheduler


@dataclass
class ForwardOutput:
    host: dict
    tensors: dict
    tokens: object
    probs: object
    ready: object


@dataclass
class RetiredStorage:
    ptr: object
    probs: object
    compute_done: object
    control_done: object = None


class AsyncCPUCompactSpecScheduler(AsyncCompactSpecScheduler):
    def __init__(self, kv_pool, model, **kwargs):
        if not kwargs.get("async_cpu") or not kwargs.get("compact_async_transfers"):
            raise ValueError("CPU compact pipeline requires both async_cpu and compact_async_transfers")
        super().__init__(kv_pool, model, **dict(kwargs, async_cpu=False))
        self.async_on = True
        self._args["async_cpu"] = self.spec_cache_args["async_cpu"] = True
        self.compute_stream = torch.cuda.Stream(device=kv_pool.device)
        self._compute_fences = {}
        self._retired = []
        self._output_keepalive = []
        self._patched_requests = []
        self.metrics.update(cpu_async_enabled=True, cpu_pipeline_submissions=0,
                            cpu_stall_only_rounds=0, cpu_updates_with_forward_inflight=0,
                            cpu_result_update_wall_seconds=0.,
                            cpu_result_readback_wall_seconds=0.,
                            deferred_retirements=0, max_deferred_retirements=0,
                            retirement_idle_wait_seconds=0.)

    def _current_cpu(self):
        return super()._current_cpu() + sum(
            getattr(item.ptr, "cold_count", 0) for item in self._retired)

    def _defer_retire(self, request):
        self._retired.append(RetiredStorage(request.kv_cache_ptr, request.draft_probs,
                                          self._compute_fences.get(request.request_id)))
        request.kv_cache_ptr, request.draft_probs = None, None
        request.len_draft_probs = 0
        self.metrics["deferred_retirements"] += 1
        self.metrics["max_deferred_retirements"] = max(
            self.metrics["max_deferred_retirements"], len(self._retired))

    def _poll_retired(self):
        if not self._retired:
            return
        local = [item.control_done is not None and item.control_done.query()
                 and (item.compute_done is None or item.compute_done.query())
                 for item in self._retired]
        observations = [local]
        if self.group is not None:
            observations = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(observations, local, group=self.group)
        if any(len(flags) != len(local) for flags in observations):
            raise RuntimeError("TP retirement queues diverged")
        keep = []
        for index, item in enumerate(self._retired):
            if all(flags[index] for flags in observations):
                if hasattr(item.ptr, "close"):
                    item.ptr.close()
                # FullCachePtr destructor releases pages when the strong
                # retirement reference is dropped; no raw-pointer consumers remain.
                item.ptr = item.probs = None
            else:
                keep.append(item)
        self._retired = keep

    def admit_requests(self, pending_requests):
        self._poll_retired()
        self._output_keepalive = [(output, event) for output, event in self._output_keepalive
                                  if not event.query()]
        super().admit_requests(pending_requests)
        # STALL entries take zero queries and exist for CPU result processing.
        # Include all of them even if that request is absent from the ready batch.
        stalls = sorted((r for r in self._requests.values()
                         if r.status == Request.ReqExecType.STALL), key=lambda r: r.prev_idx)
        self.active_requests += stalls
        active_ids = {r.request_id for r in self.active_requests}
        self.offload_requests = [r for r in self._requests.values() if r.request_id not in active_ids]

    def prepare_forward_inputs(self):
        host, tensors = SpecScheduler.prepare_forward_inputs(self)
        if host is None:
            # Native prepare returns early for an all-STALL batch. TopK still
            # needs the *current* row-to-page table after parking/reordering.
            pages, indptr = array.array("i"), array.array("i", [0])
            for request in self.active_requests:
                pages.extend(request.kv_cache_ptr.indices)
                indptr.append(len(pages))
            self.spec_cache_args.update(
                all_kv_indices_gpu=torch.tensor(pages, dtype=torch.int32, device=self.kv_pool.device),
                all_kv_indptr_cpu=indptr)
        elif self._transfers:
            self.metrics["forwards_with_transfer_inflight"] += 1
        return host, tensors

    def wait_for_ready_requests(self):
        if self._transfers:
            return super().wait_for_ready_requests()
        if not self._retired:
            raise RuntimeError("No ready request, transfer or deferred retirement")
        started = time.perf_counter()

        def wait():
            for item in self._retired:
                if item.compute_done is not None:
                    item.compute_done.synchronize()
                if item.control_done is None:
                    raise RuntimeError("Retirement has no control consumer fence")
                item.control_done.synchronize()

        coordinated(self.group, "retirement page wait", wait)
        self.metrics["retirement_idle_wait_seconds"] += time.perf_counter() - started
        self._poll_retired()

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        report = super().transfer_timing_snapshot(elapsed_seconds)
        report.update(mode="on_demand_async_kv_cpu_pipeline", cpu_scheduling="one-round pipeline",
                      deferred_retirements=len(self._retired),
                      cpu_overlap_scope="CPU handles previous result after submitting current forward")
        return report

    def _consume_previous(self, previous, current_host, current_tensors, model_done):
        started = time.perf_counter()
        control = torch.cuda.current_stream(self.kv_pool.device)
        if previous is not None:
            control.wait_event(previous.ready)
            previous.tokens.record_stream(control)
            previous.probs.record_stream(control)
            readback = time.perf_counter()
            self.update_metadata_host(current_host, current_tensors, previous.tokens)
            self.metrics["cpu_result_readback_wall_seconds"] += time.perf_counter() - readback
        if model_done is not None and not model_done.query():
            self.metrics["cpu_updates_with_forward_inflight"] += 1
        completed = self.update_requests(
            current_host["token_indptr"] if current_host is not None else None,
            previous.tokens if previous is not None else None,
            previous.probs if previous is not None else None,
            previous.host if previous is not None else None)
        consumed = control.record_event()
        for item in self._retired:
            if item.control_done is None:
                item.control_done = consumed
        if previous is not None:
            self._output_keepalive.append((previous, consumed))
        self.metrics["cpu_result_update_wall_seconds"] += time.perf_counter() - started
        return completed, consumed

    def run(self, pending_requests):
        pending_requests = list(pending_requests)
        for identity, request in enumerate(pending_requests):
            request.request_id = identity
            limit = request.len_oracle_ctx
            if limit is None or request.len_output < 1 or request.len_input < 1:
                raise ValueError("Compact requests require nonempty prompts and finite positive output limits")
            if max(self.spec_cache_args["num_min_budget"], math.ceil(
                    (limit + self.spec_stride + 1) * self.spec_cache_args["budget_ratio"])) >= 6144:
                raise ValueError("Request's maximum hot budget must be below 6144")
            if "retire" in request.__dict__:
                raise ValueError("CPU pipeline cannot replace an existing per-request retirement hook")
        completed, previous = [], None
        start = time.perf_counter()
        self._run_start_perf_counter = start
        try:
            for request in pending_requests:
                request.retire = lambda request=request: self._defer_retire(request)
                self._patched_requests.append(request)
            while pending_requests or self._requests or self._transfers:
                self.step_id += 1
                self._run_stage("CPU pipeline admission", lambda: self.admit_requests(pending_requests))
                if not self.active_requests:
                    self.wait_for_ready_requests()
                    continue
                computing = [r for r in self.active_requests if r.status != Request.ReqExecType.STALL]
                phases = [r.status for r in computing]

                def setup():
                    host, tensors = self.prepare_forward_inputs()
                    if host is not None:
                        self.model.register_input(**tensors)
                    return host, tensors

                host, tensors = self._run_stage("CPU pipeline metadata", setup)
                model_done, tokens, probs = None, None, None
                if host is not None:
                    if previous is not None:
                        self.update_metadata_device(host, tensors, previous.tokens,
                                                    self.compute_stream, torch.cuda.current_stream())
                    self.compute_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.compute_stream):
                        logits = self.model(**tensors)
                        tokens, probs = self._sample(logits, len(host["token_ids"]))
                        del logits
                        model_done = self.compute_stream.record_event()
                    for request in computing:
                        self._compute_fences[request.request_id] = model_done
                        self._last_served[request.request_id] = self.step_id
                    self.metrics["cpu_pipeline_submissions"] += 1
                    self._record_peak()
                else:
                    self.metrics["cpu_stall_only_rounds"] += 1
                newly_completed, consumed = self._run_stage(
                    "CPU previous result", lambda:
                    self._consume_previous(previous, host, tensors, model_done))
                completed.extend(newly_completed)
                if host is not None:
                    # Enqueue this only after the previous TopK/selection has
                    # been submitted. Waiting on an unrecorded event is unsafe.
                    self.compute_stream.wait_event(consumed)
                    with torch.cuda.stream(self.compute_stream):
                        host.update(self.kv_pool.output_snapshot(host))
                        ready = self.compute_stream.record_event()
                    previous = ForwardOutput(host, tensors, tokens, probs, ready)
                    self.kv_pool.step()
                    self.metrics["forwards"] += 1
                    self.metrics["request_verify_rounds"] += phases.count(Request.ReqExecType.VERIFY)
                    self.metrics["verify_forwards"] += int(Request.ReqExecType.VERIFY in phases)
                    self.metrics["batch_sum"] += len(phases)
                    self.metrics["max_actual_batch"] = max(self.metrics["max_actual_batch"], len(phases))
                    self.metrics["mixed_forwards"] += int(len(set(phases)) > 1)
                    self.prof.step()
                else:
                    previous = None
                if self.step_id % 100 == 0:
                    self._write_transfer_timing()
            self.compute_stream.synchronize()
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            self._poll_retired()
            self._output_keepalive.clear()
            self.metrics["elapsed_seconds"] = time.perf_counter() - start
            self.metrics["gpu_final_free_pages"] = self.kv_pool.num_free_pages
            self.metrics["gpu_capacity_pages"] = self.kv_pool.capacity
            if self._retired or self.kv_pool.num_free_pages != self.kv_pool.capacity:
                raise RuntimeError("CPU pipeline leaked retirement references or pool pages")
            self._write_transfer_timing(self.metrics["elapsed_seconds"], complete=True)
            if self._result_dir:
                with open(os.path.join(self._result_dir, f"compact-metrics.rank{self.rank}.json"), "w") as out:
                    json.dump(self.metrics, out, indent=2)
            return sorted(completed, key=lambda r: r.request_id)
        except BaseException:
            self._abort_failed_tp_worker()
            raise
        finally:
            self.compute_stream.synchronize()
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
            # Restore original Request.retire before ordinary cleanup. Strong
            # retired references remain valid until both streams finish.
            for request in self._patched_requests:
                request.__dict__.pop("retire", None)
                if request.kv_cache_ptr is not None:
                    request.retire()
            self._retired.clear()
            self._output_keepalive.clear()
            self._requests.clear()
            self._transfers.clear()
            self._compute_fences.clear()
            self._patched_requests.clear()
            self.active_requests, self.offload_requests = [], []
