"""Real-request component profile; k=8, native CPU pipeline.

GPU events are read after completion, never synchronized per forward. Detailed
layer ranges are sampled. The reservation mode uses the implemented VeriCache
calendar, natural EOS and native asynchronous KV transfers. Resident mode is
retained only for explicitly capped development checks.
"""
import argparse
from collections import Counter, deque
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from unittest.mock import patch

from verify_timeline import VerifyTimeline

import torch
import torch.distributed as dist

from serve.distribute.tp import setup_tp_worker
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import Attention, FeedForward, KVPool, Transformer
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.request import Request
from serve.sampling.rejection_sampler import RejectionSampler
from serve.scheduler.dynamic_compact_scheduler import DynamicCompactSpecScheduler
from serve.scheduler.reservation_scheduler import ReservationCompactSpecScheduler
from serve.utils import setup_seed

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "log/acceptance-u-k32-dynamic-20261020/full128-mem075/r0.05/lcb128-natural/worker-payload.json"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def window_action(elapsed, forwards, active, finished, started_at, warmup, duration, minimum_forwards):
    """Rank zero makes a time decision; all TP ranks receive that decision."""
    if active:
        return "stop" if elapsed-started_at>=duration else None
    if not finished and elapsed>=warmup and forwards>=minimum_forwards:
        return "start"
    return None


def attention_profile_targets(pool):
    # _wrappers[phase] == [attention, TopK]. The public wrappers property
    # exposes only the current phase; its second entry is not another buffer.
    for phase_wrappers in pool._wrappers:
        wrapper = phase_wrappers[0]
        yield wrapper, "run_attention", "batch_attention_core", True, True
        yield wrapper, "rematerialize_scores", "attention_selection_scores", True, True


class Recorder:
    def __init__(self, scheduler, output, every):
        self.scheduler, self.output, self.every = scheduler, output, every
        self.pending, self.batches, self.pool = deque(), {}, []
        self.batch_count = 0
        self.record_file = (output / f"components.rank{scheduler.rank}.jsonl").open("w", buffering=1024*1024)
        self.batch_file = (output / f"batches.rank{scheduler.rank}.jsonl").open("w", buffering=1024*1024)
        self.result_step, self.block = None, None
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin_cpu_ns = time.perf_counter_ns()
        self.origin.record()

    @property
    def step(self):
        return self.scheduler.step_id if self.result_step is None else self.result_step

    def detailed(self):
        return self.scheduler.capture_active and self.batches.get(self.step,{}).get("detail_sample",False)

    @contextmanager
    def measure(self, name, *, gpu=False, detailed=False):
        if not self.scheduler.capture_active:
            yield
            return
        if detailed and not self.detailed():
            yield
            return
        # Fine CUDA event pairs noticeably extend eager submission gaps.
        # Default to NVTX-only fine ranges; Nsight correlates their kernel
        # launches. Fine event timing remains an explicit diagnostic option.
        if detailed and not self.scheduler.observation.get("fine_gpu_events",False):
            gpu = False
        executing, producer = self.scheduler.step_id, self.step
        pair = (self.pool.pop() if self.pool else (
            torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))) if gpu else None
        started = time.perf_counter_ns()
        if pair:
            pair[0].record()
        label = f"cp.{name} case={self.output.name} producer={producer} executing={executing}"
        if producer in self.batches:
            b = self.batches[producer]
            label += f" D={b['D']} V={b['V']} P={b['P']} Q={b['query_tokens']}"
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            if pair:
                pair[1].record()
            torch.cuda.nvtx.range_pop()
            record = dict(component=name, producer_step=producer, executing_step=executing,
                          cpu_start_ns=started, cpu_end_ns=time.perf_counter_ns(),
                          gpu_events=pair)
            self.pending.append(record)

    def drain(self):
        while self.pending:
            row = self.pending[0]
            pair = row["gpu_events"]
            if pair and not pair[1].query():
                break
            self.pending.popleft()
            row.pop("gpu_events")
            row["cpu_wall_ms"] = (row["cpu_end_ns"] - row["cpu_start_ns"]) / 1e6
            if pair:
                row.update(gpu_ms=pair[0].elapsed_time(pair[1]),
                           gpu_start_ms=self.origin.elapsed_time(pair[0]),
                           gpu_end_ms=self.origin.elapsed_time(pair[1]))
                self.pool.append(pair)
            self.record_file.write(json.dumps(row) + "\n")

    def record_batch(self, state):
        # A deterministic mixed hash avoids periodic sampling aliasing with
        # draft/verify cycles, and does not consume the model's random state.
        value=state["step"]+0x9e3779b97f4a7c15
        value=((value^(value>>30))*0xbf58476d1ce4e5b9)&((1<<64)-1)
        value=((value^(value>>27))*0x94d049bb133111eb)&((1<<64)-1)
        state["detail_sample"]=self.batch_count<2 or (value^(value>>31))%self.every==0
        self.batches[state["step"]] = state
        self.batch_file.write(json.dumps(state) + "\n")
        self.batch_count += 1
        # Only the current / previous async producer needs its labels in RAM.
        while len(self.batches) > 16:
            del self.batches[next(iter(self.batches))]
        if self.batch_count % 100 == 0:
            self.record_file.flush()
            self.batch_file.flush()

    def write(self):
        self.drain()
        if self.pending:
            raise RuntimeError("Component events unfinished after native scheduler drained streams")
        self.record_file.close()
        self.batch_file.close()


class ObservationMixin:
    def __init__(self, *args, observation, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation = observation
        self.output = Path(observation["output"])
        self.case_output = self.output
        idle = self.case_output/"_idle"
        idle.mkdir(exist_ok=True)
        self.recorder = Recorder(self, idle, observation["detail_every"])
        self.decode_forwards = 0
        self.capture_started = self.capture_finished = False
        self.capture_active = False
        self.next_window_check=1
        self.window_broadcast=dist.broadcast_object_list
        self.window_started_at=None
        self.last_progress=-60.
        self.component_instrumentation=None
        self.last_context_median = 0
        self.boundaries = []
        self.window_index = 0
        self.windows_completed = 0
        self.next_window_at = observation["warmup_seconds"]
        self.window_contexts = []
        self.partial_acceptance_rounds_skipped = 0
        self.verify_timeline = (VerifyTimeline(idle/f"verify-timeline.rank{self.rank}.jsonl", self.rank)
                                if observation.get("policy") == "reservation" else None)
        self.idle_recorder, self.idle_timeline = self.recorder, self.verify_timeline
        self.verify_batch_ids = []

    def _open_profile_window(self):
        self.output = self.case_output.parent/f"{self.case_output.name}_w{self.window_index:02d}"
        self.output.mkdir(exist_ok=False)
        self.recorder = Recorder(self,self.output,self.observation["detail_every"])
        if self.observation.get("policy") == "reservation":
            self.verify_timeline = VerifyTimeline(self.output/f"verify-timeline.rank{self.rank}.jsonl",self.rank)
            self.transfer_records = (self.output/f"transfers.rank{self.rank}.jsonl").open("w",buffering=1)
            self._reservation_log = self.output/f"reservations.rank{self.rank}.jsonl"
        self.draft_start.clear()
        self.acceptance_rounds.clear()
        context = dict(output=self.output,recorder=self.recorder,timeline=self.verify_timeline,
                       transfers=getattr(self,"transfer_records",None),boundaries=[],window_id=self.window_index,
                       partial_acceptance_start=self.partial_acceptance_rounds_skipped)
        self.window_contexts.append(context)

    def _pending_transfer_snapshot(self):
        return [dict(request_id=key,direction=job.direction,cold_pages=job.cold_pages,
                     bytes=job.cold_pages*self.page_bytes) for key,job in self._transfers.items()]

    def _capture_boundary(self, start):
        # Added synchronization exists ONLY at explicit hardware-capture
        # boundaries, never at ordinary step or component boundaries.
        if self.group is not None:
            dist.barrier(group=self.group)
        self.compute_stream.synchronize()
        self.transfer_stream.synchronize()
        torch.cuda.current_stream(self.kv_pool.device).synchronize()
        if start:
            self._open_profile_window()
        # profilerStop may block on collector bookkeeping. Its own overhead
        # is outside the observed workload window and request-wait intervals.
        closing_ns = time.perf_counter_ns() if not start else None
        if not start:
            self.observation["telemetry_event"].clear()
        if self.rank == 0 and self.observation["capture"]:
            if start:
                torch.cuda.cudart().cudaProfilerStart()
            else:
                torch.cuda.cudart().cudaProfilerStop()
        if start:
            torch.cuda.nvtx.mark(f"cp.clock case={self.output.name} perf_ns={time.perf_counter_ns()}")
        if self.group is not None:
            dist.barrier(group=self.group)
        now=time.perf_counter_ns() if start else closing_ns
        lengths=[r.len_kv_cache for r in self._requests.values()]
        boundary=dict(step=self.step_id, start=start, cpu_ns=now,window_id=self.window_index,
            context_median=statistics.median(lengths) if lengths else 0,
            context_scope="all_live_requests_at_boundary",metrics=dict(self.metrics),
            pending_transfers=self._pending_transfer_snapshot(),pending_requests=self.pending_count,
            live_requests=len(self._requests))
        self.boundaries.append(boundary)
        self.window_contexts[-1]["boundaries"].append(boundary)
        if start:
            if self.verify_timeline is not None:
                self.verify_timeline.start(now, step=self.step_id)
                self._observe_verify_needs(carry_in=True)
            self.acceptance_log=(self.output/f"rounds.rank{self.rank}.jsonl").open("w")
            self.component_instrumentation=instrumentation(self)
            self.component_instrumentation.__enter__()
            self.observation["telemetry_event"].set()
        else:
            self.window_contexts[-1]["partial_acceptance_end"] = self.partial_acceptance_rounds_skipped
            if self.verify_timeline is not None:
                self.verify_timeline.stop(now)
            self.observation["telemetry_event"].clear()
            if self.component_instrumentation is not None:
                self.component_instrumentation.__exit__(None,None,None)
                self.component_instrumentation=None
            self.close_acceptance_log()
            self.acceptance_log=None
            self.recorder.drain()
            self.recorder.record_file.flush()
            self.recorder.batch_file.flush()
            if hasattr(self,"transfer_records"):
                self.transfer_records.flush()
        if self.rank==0:
            print("COMPONENT_WINDOW", "START" if start else "STOP",self.output.name,
                  dict(step=self.step_id,elapsed_seconds=time.perf_counter()-self._run_start_perf_counter),flush=True)

    def _poll_window(self):
        if self.step_id<self.next_window_check:
            return
        self.next_window_check=self.step_id+self.observation["window_check_steps"]
        command=[None]
        if self.rank==0:
            elapsed=time.perf_counter()-self._run_start_perf_counter
            command[0]=window_action(elapsed,self.metrics["forwards"],self.capture_active,
                self.capture_finished,self.window_started_at,self.next_window_at,
                self.observation["record_seconds"],self.observation["minimum_forwards"])
            if command[0]=="start" and self.metrics["restores"]<self.observation["minimum_restores"]:
                command[0]=None
            if elapsed-self.last_progress>=60:
                print("COMPONENT_PROGRESS",json.dumps(dict(case=self.output.name,
                    elapsed_seconds=round(elapsed,1),forwards=self.metrics["forwards"],
                    pending=self.pending_count,live=len(self._requests),
                    migrations=self.metrics["migrations"],restores=self.metrics["restores"],
                    recording=self.capture_active)),flush=True)
                self.last_progress=elapsed
        if self.capture_finished:
            return  # Requested windows collected; no more observer TP exchanges.
        # One small control exchange per checkpoint, not an exchange for each
        # forward. Local wall-clock decisions could diverge and deadlock TP.
        if self.group is not None:
            with self.recorder.measure("observer_window_control"):
                self.window_broadcast(command,src=0,group=self.group)
        if command[0]=="start":
            self.window_index += 1
            self._capture_boundary(True)
            self.capture_started=self.capture_active=True
            self.window_started_at=time.perf_counter()-self._run_start_perf_counter
        elif command[0]=="stop":
            self._capture_boundary(False)
            self.capture_active=False
            self.windows_completed += 1
            self.capture_finished=self.windows_completed>=self.observation.get("record_windows",1)
            self.next_window_at=(time.perf_counter()-self._run_start_perf_counter
                                 +self.observation.get("window_gap_seconds",180.))

    def admit_requests(self, pending):
        self.recorder.drain()
        self.pending_count = len(pending)
        self._poll_window()
        result=super().admit_requests(pending)
        self.pending_count=len(pending)
        if self.verify_timeline is not None and self.verify_timeline.active:
            self._observe_verify_needs()
            self._observe_verify_waits()
        return result

    def _write_transfer_timing(self, elapsed_seconds=None, complete=False):
        if self.capture_active or complete:
            return super()._write_transfer_timing(elapsed_seconds,complete=complete)

    def _run_stage(self, label, operation):
        names = {"CPU pipeline admission": "cpu_admission", "CPU pipeline metadata": "metadata_prepare",
                 "CPU previous result": "cpu_previous_result"}
        with self.recorder.measure(names.get(label, label), gpu=label == "CPU pipeline metadata"):
            return super()._run_stage(label, operation)

    def prepare_forward_inputs(self):
        host, tensors = super().prepare_forward_inputs()
        if host is not None:
            host["component_producer_step"] = self.step_id
            if not self.capture_active:
                return host,tensors
            counts = Counter()
            details = []
            for request in self.active_requests:
                if request.status == Request.ReqExecType.STALL:
                    continue
                kind = request.status.name
                counts[kind] += 1
                details.append(dict(request_id=request.request_id, phase=kind,
                    full_tokens=request.len_kv_cache, visible_kv_tokens=request.len_loading_kv_cache,
                    query_tokens=request.scheduled_tokens, retained_drafts=request.len_draft_probs,
                    cache_phase=getattr(request.kv_cache_ptr, "phase", "prefill")))
            ordinal = self.recorder.batch_count
            self.last_context_median = statistics.median(r["full_tokens"] for r in details)
            self.recorder.record_batch(dict(step=self.step_id, ordinal=ordinal,
                window_id=self.window_index,
                D=counts["DRAFT"], V=counts["VERIFY"], P=counts["NORMAL"],
                actual_batch=sum(counts.values()), query_tokens=len(host["token_ids"]),
                cpu_time_ns=time.perf_counter_ns(), free_kv_pages=self.kv_pool.num_free_pages,
                visible_kv_tokens=sum(r["visible_kv_tokens"] for r in details), requests=details,
                context_median=self.last_context_median, pending_requests=self.pending_count,
                live_requests=len(self._requests), waiting_requests=len(self.offload_requests),
                cpu_cold_pages=self._current_cpu(),
                transfers=[dict(request_id=key,direction=job.direction,cold_pages=job.cold_pages)
                           for key,job in self._transfers.items()]))
            self.verify_batch_ids = [r["request_id"] for r in details if r["phase"] == "VERIFY"]
            if counts["NORMAL"] == 0:
                self.decode_forwards += 1
        return host, tensors

    def _sample(self, *args, **kwargs):
        with self.recorder.measure("ordinary_sampling", gpu=True):
            return super()._sample(*args, **kwargs)

    def _consume_previous(self, previous, *args, **kwargs):
        old = self.recorder.result_step
        self.recorder.result_step = previous.host["component_producer_step"] if previous else -1
        try:
            with self.recorder.measure("cpu_result_processing"):
                return super()._consume_previous(previous, *args, **kwargs)
        finally:
            self.recorder.result_step = old

    def verify_outputs(self, request, *args, **kwargs):
        # Opening the log mid-run does not give native verify_outputs the
        # draft-start metadata of an already running round. Skip that partial
        # observation, then seed the next complete round exactly as native
        # verify_outputs does. Sampling, KV updates and throughput still run.
        if self.acceptance_log is not None and request.request_id not in self.draft_start:
            log = self.acceptance_log
            self.acceptance_log = None
            try:
                result = super().verify_outputs(request, *args, **kwargs)
            finally:
                self.acceptance_log = log
            self.draft_start[request.request_id] = (
                request.len_kv_cache, request.kv_cache_ptr.len_loading_indices, False)
            self.partial_acceptance_rounds_skipped += 1
            return result
        return super().verify_outputs(request, *args, **kwargs)

    def update_metadata_host(self, *args, **kwargs):
        with self.recorder.measure("cpu_result_readback_and_patch"):
            return super().update_metadata_host(*args, **kwargs)

    def is_request_complete(self, request, check_idx=-1):
        # Preserve the native completion decision, including its EOS trimming;
        # remember which native condition applied before KV is retired.
        boundary=(request.len_kv_cache+check_idx>=request.len_oracle_ctx-1
                  if request.status==Request.ReqExecType.NORMAL
                  else request.len_kv_cache>=request.len_oracle_ctx)
        complete=super().is_request_complete(request,check_idx=check_idx)
        if complete:
            request.profile_stop_reason="boundary" if boundary else "eos"
        return complete

    def run(self, pending):
        result = super().run(pending)
        if self.capture_active:
            self._capture_boundary(False)
            self.capture_active = False
        self.idle_recorder.write()
        window_statuses=[]
        for context in self.window_contexts:
            context["recorder"].write()
            boundaries=context["boundaries"]
            status=("complete" if len(boundaries)==2 and boundaries[1]["cpu_ns"]-boundaries[0]["cpu_ns"]
                    >=self.observation["record_seconds"]*1e9 else "shortened_by_completion")
            window_statuses.append(dict(window_id=context["window_id"],status=status,
                                        output=str(context["output"]),boundaries=boundaries))
            save(context["output"] / f"observer.rank{self.rank}.json", dict(metrics=self.metrics,
             capture_started=True, capture_finished=len(boundaries)==2,
             capture_boundaries=boundaries, detailed_every=context["recorder"].every,
             window_id=context["window_id"],parent_run=self.case_output.name,
             full_run_metrics_shared_across_windows=True,
             partial_acceptance_rounds_skipped=context.get("partial_acceptance_end",self.partial_acceptance_rounds_skipped)
                                                -context["partial_acceptance_start"],
             policy=self.observation.get("policy","resident"),
             recording_scope="window",requested_warmup_seconds=self.observation["warmup_seconds"],
             requested_record_seconds=self.observation["record_seconds"],
             window_status=status,event_origin_cpu_ns=context["recorder"].origin_cpu_ns))
        save(self.case_output/f"sampling-coverage.rank{self.rank}.json",dict(
            requested_windows=self.observation.get("record_windows",1),observed_windows=len(window_statuses),
            complete_windows=sum(r["status"]=="complete" for r in window_statuses),windows=window_statuses,
            missing_windows=max(0,self.observation.get("record_windows",1)-len(window_statuses)),
            requested_gap_seconds=self.observation.get("window_gap_seconds",180.),
            status="complete" if sum(r["status"]=="complete" for r in window_statuses)
                    ==self.observation.get("record_windows",1) else "insufficient_natural_runtime"))
        return result


class ObservedResident(ObservationMixin, DynamicCompactSpecScheduler):
    def _submit_transfer(self, request, direction):
        raise RuntimeError(f"No-offload diagnostic attempted {direction}; provision a larger resident pool")

    def _queue_preemption(self, protected):
        raise RuntimeError("No-offload diagnostic attempted preemption; workload is not fully resident")


class ObservedReservation(ObservationMixin, ReservationCompactSpecScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_records = (self.case_output/"_idle"/f"transfers.rank{self.rank}.jsonl").open("w",buffering=1)
        self.idle_transfer_records = self.transfer_records
        self.observed_transfers = {}
        self.transfer_sequence = 0
        self.completed_transfer_events = deque()

    def _observe_verify_needs(self, carry_in=False):
        for request in self._requests.values():
            self.verify_timeline.observe(request.request_id, request.status.name, carry_in=carry_in,
                step=self.step_id, logical_tokens=request.len_kv_cache,
                cold_pages=getattr(request.kv_cache_ptr,"cold_count",0),
                cache_phase=getattr(request.kv_cache_ptr,"phase","prefill"))

    def _early_verify(self, request, *args, **kwargs):
        result = super()._early_verify(request, *args, **kwargs)
        if self.verify_timeline is not None and self.verify_timeline.active:
            self.verify_timeline.observe(request.request_id, request.status.name,
                step=self.step_id, source="early_verify_checkpoint")
        return result

    def update_requests(self, *args, **kwargs):
        result = super().update_requests(*args, **kwargs)
        if self.verify_timeline is not None and self.verify_timeline.active:
            self._observe_verify_needs()
        return result

    def verify_outputs(self, request, *args, **kwargs):
        result = super().verify_outputs(request, *args, **kwargs)
        if self.verify_timeline is not None:
            self.verify_timeline.result(request.request_id)
        return result

    def _observe_verify_waits(self):
        table = self.reservation_table
        required_extra = self._growth_pages() + self._pack_workspace()
        free = self.kv_pool.num_free_pages
        selected = {r.request_id for r in self.active_requests if r.status.name == "VERIFY"}
        for request in self._requests.values():
            if request.status.name != "VERIFY":
                continue
            identity, ptr = request.request_id, request.kv_cache_ptr
            booking = table.reservations.get(identity)
            job = self._transfers.get(identity)
            required = getattr(ptr,"cold_count",0) + required_extra
            dependency_ready = booking is None or table.dependencies_ready(identity)
            flags = dict(free_pages=free, required_reload_pages=required,
                capacity_insufficient=required>free, transfer_busy=bool(self._transfers),
                dependencies_ready=dependency_ready, selected=identity in selected,
                reservation_start=booking.start if booking else None,
                reservation_deadline=booking.deadline if booking else None,
                step=self.step_id, cache_phase=ptr.phase)
            if job is not None:
                reason = "restore_inflight" if job.direction == "restore" else "own_repack"
            elif booking is None:
                reason = "reservation_unassigned"
            elif ptr.phase == "hot":
                if self._transfers:
                    reason = "transfer_busy"
                elif booking.start > self.step_id:
                    reason = "reservation_not_started"
                elif not dependency_ready:
                    reason = "release_dependency"
                elif required > free:
                    reason = "gpu_capacity"
                else:
                    reason = "reload_dispatch"
            elif self.step_id < booking.deadline:
                reason = "reservation_deadline"
            elif identity in selected:
                reason = "batch_preparation"
            else:
                reason = "ready_not_selected"
            self.verify_timeline.wait(identity, reason, **flags)

    def _verify_forward_submit(self):
        if self.verify_timeline is not None:
            for identity in self.verify_batch_ids:
                self.verify_timeline.submit_verify(identity, self.step_id)

    def _submit_transfer(self, request, direction):
        if not self.capture_active:
            return super()._submit_transfer(request,direction)
        pair = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        started = time.perf_counter_ns()
        cycle = self.verify_timeline.cycles[request.request_id]
        self.verify_timeline.note(direction+"_submit_begin", request.request_id, step=self.step_id)
        pair[0].record(self.transfer_stream)
        with ExitStack() as stack:
            if direction == "restore":
                original_allocate = request.kv_cache_ptr._allocate
                def allocate(count):
                    before=time.perf_counter_ns()
                    pages=original_allocate(count)
                    self.verify_timeline.note("restore_pages_allocated",request.request_id,
                        step=self.step_id, allocation_begin_cpu_ns=before, pages=count)
                    return pages
                stack.enter_context(patch.object(request.kv_cache_ptr,"_allocate",allocate))
            with self.recorder.measure("kv_"+direction):
                result = super()._submit_transfer(request,direction)
        pair[1].record(self.transfer_stream)
        self.verify_timeline.note(direction+"_submitted",request.request_id,step=self.step_id)
        if direction == "restore":
            self.verify_timeline.wait(request.request_id,"restore_inflight",step=self.step_id)
        job = self._transfers[request.request_id]
        self.transfer_sequence = self.metrics["transfers_submitted"]
        row = dict(sequence=self.transfer_sequence,request_id=request.request_id,direction=direction,
            submit_step=self.step_id,submit_cpu_ns=started,submit_end_cpu_ns=time.perf_counter_ns(),
            submit_forward=self.metrics["forwards"],cold_pages=job.cold_pages,
            bytes=job.cold_pages*self.page_bytes,source_pages=job.source_pages,
            logical_tokens=request.len_kv_cache, verify_cycle=cycle)
        self.observed_transfers[request.request_id] = (row,pair,self.window_contexts[-1])
        self.transfer_records.write(json.dumps(dict(row,event="submit"))+"\n")
        return result

    def _poll_transfers(self):
        # Native polling already queries completion on every TP rank. Read
        # timestamps only after its commit; add no wait or synchronization.
        if not self.verify_timeline.active and not self.observed_transfers:
            return super()._poll_transfers()
        before=set(self.observed_transfers)
        with ExitStack() as stack:
            for identity,job in self._transfers.items():
                if job.direction != "restore" or not self.verify_timeline.active:
                    continue
                original_commit=job.request.kv_cache_ptr.commit_restore
                def commit(*args, _original=original_commit, _identity=identity, **kwargs):
                    self.verify_timeline.note("restore_ready_observed",_identity,step=self.step_id)
                    result=_original(*args,**kwargs)
                    self.verify_timeline.note("restore_page_table_committed",_identity,step=self.step_id)
                    self.verify_timeline.wait(_identity,"await_admission",step=self.step_id)
                    return result
                stack.enter_context(patch.object(job.request.kv_cache_ptr,"commit_restore",commit))
            result=super()._poll_transfers()
        for identity in before-self._transfers.keys():
            row,pair,context=self.observed_transfers.pop(identity)
            row.update(event="commit",commit_step=self.step_id,commit_cpu_ns=time.perf_counter_ns(),
                forwards_while_inflight=self.metrics["forwards"]-row["submit_forward"])
            self.completed_transfer_events.append((row,pair,context))
        self._drain_transfer_events()
        return result

    def _reservation_event(self,*args,**kwargs):
        if getattr(self,"capture_active",False):
            return super()._reservation_event(*args,**kwargs)

    def _drain_transfer_events(self):
        while self.completed_transfer_events and self.completed_transfer_events[0][1][1].query():
            row,pair,context=self.completed_transfer_events.popleft()
            origin=context["recorder"].origin
            row.update(
                gpu_stream_span_ms=pair[0].elapsed_time(pair[1]),
                gpu_start_ms=origin.elapsed_time(pair[0]),gpu_end_ms=origin.elapsed_time(pair[1]))
            if row["direction"] == "restore":
                context["timeline"].emit("restore_device_ready",row["request_id"],
                    cycle=row["verify_cycle"],gpu_ready_ms=row["gpu_end_ms"],
                    clock="cuda_event_origin",sequence=row["sequence"],
                    timestamp_scope="End event on transfer stream, recorded after native submit")
            context["transfers"].write(json.dumps(row)+"\n")
            if not context["timeline"].active:
                context["timeline"].file.flush()

    def wait_for_ready_requests(self):
        with self.recorder.measure("cpu_wait_for_ready"):
            return super().wait_for_ready_requests()

    def run(self,pending):
        try:
            result=super().run(pending)
            self._drain_transfer_events()
            if self.observed_transfers or self.completed_transfer_events:
                raise RuntimeError("Unfinished transfer observations after native stream drain")
            return result
        finally:
            self.idle_transfer_records.close()
            if self.idle_timeline is not None:self.idle_timeline.close()
            for context in self.window_contexts:
                context["transfers"].close()
                if context["timeline"] is not None:context["timeline"].close()


@contextmanager
def instrumentation(worker):
    recorder, pool = worker.recorder, worker.kv_pool
    def wrap(function, label, *, gpu=False, detailed=False, block=None):
        def call(*args, **kwargs):
            old = recorder.block
            if block:
                recorder.block = block
            try:
                if label == "forward" and worker.verify_timeline is not None:
                    worker._verify_forward_submit()
                with recorder.measure(label, gpu=gpu, detailed=detailed):
                    return function(*args, **kwargs)
            finally:
                recorder.block = old
        return call

    comm = worker.model.model.ccl
    original_reduce = comm.all_reduce if comm else None
    def reduce(*args, **kwargs):
        label = "tp_all_reduce_" + (recorder.block or "embedding")
        with recorder.measure(label, gpu=True, detailed=True):
            return original_reduce(*args, **kwargs)

    with ExitStack() as stack:
        targets = [
            (worker.model, "forward", "forward", True, False),
            (worker.rejection_sampler, "forward", "rejection_sampling", True, False),
        ]
        targets.extend(attention_profile_targets(pool))
        for target, name, label, gpu, detailed in targets:
            stack.enter_context(patch.object(target, name, wrap(getattr(target, name), label,
                                                               gpu=gpu, detailed=detailed)))
        stack.enter_context(patch.object(Attention, "forward", wrap(Attention.forward,
            "attention_block", gpu=True, detailed=True, block="attention")))
        stack.enter_context(patch.object(FeedForward, "forward", wrap(FeedForward.forward,
            "mlp_block", gpu=True, detailed=True, block="mlp")))
        activation = __import__("flashinfer").activation
        stack.enter_context(patch.object(activation, "silu_and_mul", wrap(activation.silu_and_mul,
            "mlp_activation", gpu=True, detailed=True)))
        for layer in worker.model.model.layers:
            mlp = layer.mlp
            for name in ("w12", "down_proj"):
                module = getattr(mlp, name)
                stack.enter_context(patch.object(module, "forward", wrap(module.forward,
                    "mlp_input_gemm" if name == "w12" else "mlp_output_gemm", gpu=True, detailed=True)))
        if comm:
            stack.enter_context(patch.object(comm, "all_reduce", reduce))
        stack.enter_context(patch.object(RejectionSampler, "parse_output", staticmethod(wrap(
            RejectionSampler.parse_output, "cpu_rejection_readback_and_parse"))))
        original_selection = CompactCachePtr.dispatch_update_selected_indices.__func__
        stack.enter_context(patch.object(CompactCachePtr, "dispatch_update_selected_indices", classmethod(
            wrap(original_selection, "selection_update", gpu=True))))
        for name in ("all_gather_object", "broadcast_object_list"):
            stack.enter_context(patch.object(dist, name, wrap(getattr(dist, name), "cpu_tp_control")))
        stack.enter_context(patch.object(dist, "all_gather_into_tensor", wrap(
            dist.all_gather_into_tensor, "tp_vocab_all_gather", gpu=True, detailed=True)))
        yield


def pcie_sampler(devices, output, stop, active):
    import pynvml as nvml
    nvml.nvmlInit()
    handles = [(index, nvml.nvmlDeviceGetHandleByIndex(index)) for index in devices]
    try:
        with output.open("w", buffering=1) as out:
            while not stop.is_set():
                if not active.wait(.1):
                    continue
                for index, handle in handles:
                    if not active.is_set():
                        break
                    row = dict(cpu_time_ns=time.perf_counter_ns(), gpu=index,
                        rx_kib_per_s=nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES),
                        tx_kib_per_s=nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES))
                    out.write(json.dumps(row) + "\n")
                stop.wait(.05)
    finally:
        nvml.nvmlShutdown()


def calibrate_transfer_bandwidth(pool, group, output, rank):
    """One warm-up and three native-kernel transfers per direction.

    Runs before scheduling, against unowned pool pages. It deliberately drains
    these calibration events; ordinary observed forwards never gain a wait.
    """
    from serve.attention.kv_pack import pack_selected_kv, restore_to_pool
    count=min(8192,pool.capacity)
    physical=pool._physical_mem_gpu
    layers,_,_,_,heads,_=physical.shape
    ids=torch.arange(count,dtype=torch.int32,device=physical.device)
    selected=ids[None,None,:].expand(layers,heads,count).contiguous()
    pages=tuple(range(count))
    cold=torch.empty((layers,count,*physical.shape[2:]),dtype=physical.dtype,device="cpu",pin_memory=True)
    bytes_per_transfer=cold.numel()*cold.element_size()
    rows={}
    for name,operation in [("repack_cold_d2h",lambda:pack_selected_kv(physical,selected,cold,count)),
                           ("restore_h2d",lambda:restore_to_pool(cold,physical,pages))]:
        values=[]
        for repeat in range(4):
            if group is not None:
                dist.barrier(group=group)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();operation();end.record();end.synchronize()
            elapsed=start.elapsed_time(end)
            if repeat:values.append(elapsed)
        rows[name]=dict(samples_ms=values,median_ms=statistics.median(values),
                        gbps=bytes_per_transfer/(statistics.median(values)*1e6))
    local=dict(rank=rank,bytes_per_transfer=bytes_per_transfer,directions=rows)
    gathered=[local]
    if group is not None:
        gathered=[None]*dist.get_world_size(group)
        dist.all_gather_object(gathered,local,group=group)
    bandwidth=min(row["directions"][direction]["gbps"] for row in gathered for direction in rows)
    report=dict(ranks=gathered,reservation_bandwidth_gbps=bandwidth,
        tp=dist.get_world_size(group) if group is not None else 1,
        scope="Native cold gather/offload and restore kernels, idle model (concurrent across ranks when TP>1); excludes partition/hot gather and CPU allocation. Not a physical PCIe peak or overlap bandwidth guarantee.")
    save(output/f"bandwidth-calibration.rank{rank}.json",report)
    del cold,selected,ids
    return bandwidth


def main(args):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float16)
    setup_seed(330)
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.input.read_text())
    rows = source["requests"][:args.requests]
    if len(rows)!=args.requests:
        raise ValueError("Input has fewer requests than requested; no implicit duplication")
    if max(args.batches) > len(rows):
        raise ValueError("Requested max batch exceeds fixed request count")
    context_limit=min(args.context_limit,json.loads((Path(args.model)/"config.json").read_text())["max_position_embeddings"])
    caps=[context_limit-len(r["input_ids"]) for r in rows]
    if args.output_tokens is not None:
        caps=[min(cap,args.output_tokens) for cap in caps]
    if min(caps)<1:
        raise ValueError("A prompt exceeds the configured serving context boundary")
    known_pages = sum(len(r["input_ids"]) + cap + 2 * args.stride + 3 for r,cap in zip(rows,caps))
    # Retain the unchanged policy's temporary-repack reserve even though the
    # diagnostic never consumes it. It otherwise parks small resident tests.
    workspace_pages = max(192, math.ceil(max(len(r["input_ids"]) + cap
                                           + 2 * args.stride + 3 for r,cap in zip(rows,caps)) * .05)) + args.stride + 1
    capacity = args.pool_pages if args.policy=="reservation" else known_pages + workspace_pages
    buffer_batch_capacity=args.buffer_batch_capacity or max(args.batches)
    if buffer_batch_capacity<max(args.batches):
        raise ValueError("Buffer batch capacity must cover every scanned batch limit")
    cpu_capacity=(buffer_batch_capacity+1)*(context_limit+args.stride+1) if args.policy=="reservation" else capacity
    if args.policy=="reservation" and capacity<context_limit+workspace_pages+args.stride+1:
        raise ValueError("Pool must fit one full serving context plus native repack workspace")
    model = Transformer.from_pretrained(args.model, device=device, dtype=torch.float16)
    model = setup_tp_worker(model, world, device, torch.float16, "customized")
    group = dist.new_group(backend="gloo") if world > 1 else None
    cfg = model.config
    spec = dict(spec_stride=args.stride, budget_ratio=.05, num_min_budget=192,
                max_batch_size=buffer_batch_capacity, max_selected_kv_len=6144)
    RejectionSampler.MAX_SPEC_LEN = 10
    if args.stride != 8:
        raise ValueError("This controlled profile fixes k=8")
    CompactCachePtr.initialize_class_metadata(dtype=torch.float16, head_dim=cfg.head_dim,
        num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, num_layers=cfg.num_layers,
        capacity=capacity, device=device, max_batch_size=buffer_batch_capacity, spec_cache_args=spec)
    pool = KVPool(cfg, capacity, device, CompactCachePtr, dtype=torch.float16, allocate_legacy_cpu=False)
    model.attach_kv_pool(pool)
    bandwidth=None
    if args.policy=="reservation":
        bandwidth=calibrate_transfer_bandwidth(pool,group,args.output,rank)
    page_bytes=cfg.num_layers*2*cfg.num_kv_heads*cfg.head_dim*2
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        request_ids=[r["sample_id"] for r in rows], input_lengths=[len(r["input_ids"]) for r in rows],
        pool_pages=capacity, known_pages=known_pages, policy_workspace_pages=workspace_pages,
        serving_context_limit=context_limit,output_caps=caps,
        termination_policy="natural_eos_or_serving_context_boundary" if args.output_tokens is None else "explicit_development_output_cap",
        cpu_capacity_pages=cpu_capacity,cpu_logical_budget_bytes_per_rank=cpu_capacity*page_bytes,
        cpu_allocation="lazy per-request pinned cold backing, not preallocated to logical budget",
        reservation_bandwidth_gbps=bandwidth,reservation_iteration_ms=args.reservation_iteration_ms,
        iteration_estimate_source=args.reservation_iteration_source,
        kv_page_bytes_per_rank=page_bytes,kv_pool_bytes_per_rank=capacity*page_bytes,
        probability_rows=RejectionSampler.MAX_SPEC_LEN,
        buffer_batch_capacity=buffer_batch_capacity,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), tp=world, query_budget=args.query_budget,
        source_sha256={str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [Path(__file__).resolve(),Path(__file__).with_name("verify_timeline.py").resolve(),
                                    *sorted((REPO/"serve").rglob("*.py"))]},
        scope=("Implemented VeriCache BW/HBM reservations, real cached LiveCodeBench prompts, native CPU/transfer async, eager; natural EOS unless an explicit development cap is supplied."
               if args.policy=="reservation" else "Development-only resident no-offload profile, explicitly capped."))
    save(args.output/f"manifest.rank{rank}.json", manifest)
    stop = threading.Event()
    telemetry_event = threading.Event()
    thread = None
    if rank == 0:
        thread = threading.Thread(target=pcie_sampler,
            args=([int(v) for v in os.environ["CUDA_VISIBLE_DEVICES"].split(",")],args.output/"pcie-nvml.jsonl",stop,telemetry_event),daemon=True)
        thread.start()
    try:
        with torch.inference_mode():
            for batch in args.batches:
                case = args.output/f"b{batch}"
                case.mkdir(exist_ok=True)
                setup_seed(330)
                pool.phase_idx = 0
                worker_class=ObservedReservation if args.policy=="reservation" else ObservedResident
                reservation_args=dict(compact_reservation=True,reservation_bandwidth_gbps=bandwidth,
                    reservation_iteration_ms=args.reservation_iteration_ms,reservation_horizon=64,
                    reservation_target_length=args.stride) if args.policy=="reservation" else {}
                worker = worker_class(pool, CUDAGraphModelWrapper(model), **dict(spec,max_batch_size=batch),
                    observation=dict(output=str(case),detail_every=args.detail_every,capture=args.capture,
                                     fine_gpu_events=args.fine_gpu_events,policy=args.policy,
                                     warmup_seconds=args.warmup_seconds,record_seconds=args.record_seconds,
                                     record_windows=args.record_windows,window_gap_seconds=args.window_gap_seconds,
                                     window_check_steps=args.window_check_steps,
                                     minimum_forwards=args.minimum_forwards,
                                     minimum_restores=args.minimum_restores if args.policy=="reservation" else 0,
                                     telemetry_event=telemetry_event),
                    eos_token_id=[151645,151643], admit_policy="offloading", async_cpu=True, tp_ranks=world,
                    temperature=.6, enable_torch_profiler=False, profiler_result_dir=str(case),
                    batch_policy="greedy", enable_cuda_graph=False, chunk_size=256,
                    trace_name="component-profile",kv_cache="compact", compact_cpu_pages=cpu_capacity,
                    compact_async_transfers=True, compact_control_group=group, compact_debug_checks=False,
                    compact_dynamic_admission=True, max_num_batched_tokens=args.query_budget,
                    acceptance_log_path=None,**reservation_args)
                print("COMPONENT_CASE_START",rank,batch,flush=True)
                results = worker.run([Request(r["input_ids"],cap) for r,cap in zip(rows,caps)])
                worker.close_acceptance_log()
                worker.step_trace.close(True)
                assert len(results)==len(rows) and pool.num_free_pages==pool.capacity
                completion=dict(complete=True,
                     outputs=[dict(id=r.request_id,sample_id=rows[r.request_id]["sample_id"],
                                   input_tokens=r.len_input,output_tokens=r.len_token_ids-r.len_input,
                                   output_cap=r.len_output,
                                   stop_reason=("serving_context_boundary" if args.output_tokens is None else "development_output_cap")
                                               if r.profile_stop_reason=="boundary" else "eos")
                              for r in results],metrics=worker.metrics)
                save(case/f"completion.rank{rank}.json",completion)
                for context in worker.window_contexts:
                    save(context["output"]/f"completion.rank{rank}.json",completion)
                print("COMPONENT_CASE_DONE",rank,batch,worker.metrics["forwards"],flush=True)
                del worker, results
                gc.collect()
                if group is not None:
                    dist.barrier(group=group)
        for name,digest in manifest["source_sha256"].items():
            if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError("Source changed during profile")
    finally:
        stop.set()
        if thread:
            thread.join(timeout=3)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--model",default="/data/models/Qwen3-8B")
    parser.add_argument("--stride",type=int,default=8)
    parser.add_argument("--policy",choices=["reservation","resident"],default="reservation")
    parser.add_argument("--requests",type=int,default=128)
    parser.add_argument("--output-tokens",type=int,default=None,help="Development checks only; omitted in the formal natural-EOS run")
    parser.add_argument("--context-limit",type=int,default=32768,help="Serving total-context boundary, retained from aligned natural-EOS experiments")
    parser.add_argument("--pool-pages",type=int,default=262144,help="Fixed GPU KV pool shared by all batch configurations")
    parser.add_argument("--query-budget",type=int,default=2048)
    parser.add_argument("--reservation-iteration-ms",type=float,default=33.4)
    parser.add_argument("--reservation-iteration-source",default="Initial estimate from prior TP2 eager k8 profile; not calibrated for other TP sizes or a deadline guarantee")
    parser.add_argument("--batches",type=int,nargs="+",default=[16,32,48,64,72,128])
    parser.add_argument("--buffer-batch-capacity",type=int,default=None,
                        help="Fixed metadata/CPU-budget capacity across separate sweep workers; must cover max batch")
    parser.add_argument("--detail-every",type=int,default=8,help="Approximate inverse fine-NVTX sampling rate within the recording window")
    parser.add_argument("--fine-gpu-events",action="store_true",help="Diagnostic only: per-layer events perturb eager launches")
    parser.add_argument("--capture",action="store_true")
    parser.add_argument("--warmup-seconds",type=float,default=120.)
    parser.add_argument("--record-seconds",type=float,default=180.)
    parser.add_argument("--record-windows",type=int,default=1)
    parser.add_argument("--window-gap-seconds",type=float,default=180.)
    parser.add_argument("--window-check-steps",type=int,default=20)
    parser.add_argument("--minimum-forwards",type=int,default=12)
    parser.add_argument("--minimum-restores",type=int,default=2)
    args=parser.parse_args()
    if min(args.requests,args.detail_every,args.window_check_steps,args.minimum_forwards,args.context_limit,args.pool_pages,*args.batches)<1:
        parser.error("Positive request/iteration sizes required")
    if args.output_tokens is not None and args.output_tokens<1:
        parser.error("Development output cap must be positive")
    if args.policy=="resident" and args.output_tokens is None:
        parser.error("Resident development mode requires an explicit --output-tokens")
    if args.query_budget<args.stride+1 or not math.isfinite(args.reservation_iteration_ms) or args.reservation_iteration_ms<=0:
        parser.error("Require query budget >= k+1 and a finite positive iteration estimate")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds<0 or not math.isfinite(args.record_seconds) or args.record_seconds<=0:
        parser.error("Require finite warmup >= 0 and recording duration > 0")
    if args.minimum_restores<0:
        parser.error("Minimum restores must be nonnegative")
    if args.record_windows<1 or not math.isfinite(args.window_gap_seconds) or args.window_gap_seconds<0:
        parser.error("Require positive window count and finite nonnegative gap")
    if args.buffer_batch_capacity is not None and args.buffer_batch_capacity<max(args.batches):
        parser.error("Buffer batch capacity must be >= maximum requested batch")
    main(args)
