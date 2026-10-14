"""Eager serving path for head-local compact KV with real CPU offload.

The model projections, MLP and hot/cold attention remain batched. Verify
groups are bounded by explicit GPU/CPU arena headroom, including all restored
cold histories and replacement buffers.
The entire model forward precedes packing/DMA and the next forward. Admission
reserves headroom for a complete restore plus replacement, including CPU double
backing. No implicit reduction of configured k, r, or batch limit is performed.
"""

import array
import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from serve.attention.full_forward import FullForward, FullKVRegion
from serve.attention.kv_partition import PackedKVRegion
from serve.attention.packed_batch import PackedBatch
from serve.attention.packed_forward import PackedForward
from serve.sampling.rejection_sampler import RejectionSampler
from serve.sampling.sampler import Sampler, SamplingMetadata


def coordinated_call(group, label, operation, rollback=None):
    """Report allocation/setup failures on every rank before model collectives.

This cannot recover a CUDA device fault inside a TP model collective. It does
prevent a rank with an exhausted arena from leaving peers entering the model.
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
    if any(item is not None for item in errors):
        if result is not None and rollback is not None:
            rollback(result)
        raise RuntimeError(f"Packed KV {label} failed across ranks: {errors}")
    return result


@dataclass
class _State:
    request: object
    region: object
    prompt: list
    output: list = field(default_factory=list)
    drafts: list = field(default_factory=list)
    draft_probs: list = field(default_factory=list)
    round_start: int = 0
    round_id: int = 0


class PackedSpecScheduler:
    def __init__(self, model, gpu_pool, cpu_pool, *, max_batch_size=4, chunk_size=256,
                 spec_stride=4, budget_ratio=.1, num_min_budget=128, temperature=0.,
                 eos_token_id=(), tp_ranks=1, control_group=None, result_dir=None,
                 acceptance_log_path=None, chunk_pages=1024, dense_reference=False,
                 retain_full_reference=False):
        if (max_batch_size < 1 or chunk_size < 1 or spec_stride < 1 or chunk_pages < 1
                or num_min_budget < 1 or not math.isfinite(budget_ratio)
                or not 0 < budget_ratio <= 1 or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("Invalid compact scheduler parameters")
        if gpu_pool.device.type != "cuda" or not cpu_pool.is_pinned:
            raise ValueError("Serving requires a CUDA arena and a pinned CPU arena")
        if tp_ranks > 1 and control_group is None:
            raise ValueError("TP compact scheduling requires a shared Gloo control group")
        self.model, self.gpu_pool, self.cpu_pool = model, gpu_pool, cpu_pool
        self.config = model.config
        self.device, self.dtype = gpu_pool.device, next(model.parameters()).dtype
        self.max_batch_size, self.chunk_size = max_batch_size, chunk_size
        self.k, self.r, self.minimum = spec_stride, budget_ratio, num_min_budget
        self.chunk_pages, self.dense_reference = chunk_pages, dense_reference
        # Test-only control: use the same speculative schedule while retaining
        # dense KV, to separate verify batching error from split-attention error.
        self.retain_full_reference = retain_full_reference
        if dense_reference and retain_full_reference:
            raise ValueError("Select dense AR or dense self-spec as the reference")
        self.group, self.tp_ranks = control_group, tp_ranks
        self.rank = dist.get_rank() if control_group is not None else 0
        self.eos = set(token for token in eos_token_id if token is not None)
        self.sampler = Sampler(tp_ranks)
        self.metadata = SamplingMetadata(temperature, temperature == 0)
        self.reject = RejectionSampler()
        RejectionSampler.MAX_SPEC_LEN = max(RejectionSampler.MAX_SPEC_LEN, spec_stride)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.token_bytes = (self.config.num_layers * 2 * self.config.num_kv_heads
                            * self.config.head_dim * torch.empty((), dtype=self.dtype,
                                                                 device="cpu").element_size())
        self.step = 0
        self.accepted = self.proposed = 0
        self.metrics = dict(forwards=0, verify_forwards=0, migrations=0, restores=0,
                            admission_waits=0, max_actual_batch=0, mixed_prefill_draft_forwards=0,
                            mixed_verify_draft_forwards=0, mixed_verify_prefill_forwards=0, verify_model_forwards=0,
                            max_verify_batch=0, verify_batch_histogram={}, d2h_bytes=0, h2d_bytes=0, full_bytes_returned=0)
        self._events = self._rounds = None
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            self._events = open(os.path.join(result_dir, f"kv-events.rank{self.rank}.jsonl"), "w")
        if acceptance_log_path is None and result_dir is not None:
            acceptance_log_path = os.path.join(result_dir, "rounds.jsonl")
        if acceptance_log_path is not None and self.rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(acceptance_log_path)), exist_ok=True)
            self._rounds = open(acceptance_log_path, "w")

    def _log(self, event, **fields):
        if self._events is not None:
            row = dict(event=event, step=self.step, rank=self.rank,
                       gpu=self.gpu_pool.stats(), cpu=self.cpu_pool.stats(), **fields)
            self._events.write(json.dumps(row) + "\n")
            self._events.flush()

    def _broadcast(self, value):
        if self.group is None:
            return value
        items = [value if self.rank == 0 else None]
        dist.broadcast_object_list(items, src=0, group=self.group)
        return items[0]

    def _estimate(self, request, *, packed):
        """Worst-case history growth and shared one-verify headroom, in bytes."""
        length = request.len_input + request.len_output - 1
        hot = min(length, max(self.minimum, math.ceil(length * self.r), self.k + 2))
        # Cold is maximized at the minimum valid recent suffix (two tokens).
        cold = length - min(length, max(self.minimum, math.ceil(length * self.r), 2))
        if self.dense_reference or self.retain_full_reference:
            return (length + 1) * self.token_bytes, 0, 0, 0
        packed_base = hot + self.k + 1
        if packed:
            base, extra = packed_base, cold + packed_base
        else:
            # Cover both peaks: initial full KV + its packed replacement, and
            # later packed KV + restored cold + another packed replacement.
            # Full -> packed reduces the reserved base by a nonnegative amount;
            # it must not reserve a future restore on top of the retired full KV.
            base = max(request.len_input + self.k + 1, packed_base)
            extra = max(packed_base, cold + 2 * packed_base - base)
        return (base * self.token_bytes, extra * self.token_bytes,
                cold * self.token_bytes, cold * self.token_bytes)

    def _fits(self, active, request):
        estimates = [self._estimate(state.request, packed=isinstance(state.region, PackedKVRegion))
                     for state in active]
        estimates.append(self._estimate(request, packed=False))
        # Alignment may round each hot/cold/restore lease upward.
        padding = (len(estimates) * 2 + 4) * self.gpu_pool.alignment
        gpu = sum(e[0] for e in estimates) + max(e[1] for e in estimates) + padding
        cpu = sum(e[2] for e in estimates) + max(e[3] for e in estimates) + padding
        return gpu <= self.gpu_pool.capacity_bytes and cpu <= self.cpu_pool.capacity_bytes

    def _admit(self, pending, active):
        while pending and len(active) < self.max_batch_size:
            request = pending[0]
            fits = self._broadcast(self._fits(active, request) if self.rank == 0 else None)
            if not fits:
                self.metrics["admission_waits"] += 1
                self._log("admission_wait", request_id=request.request_id)
                if not active:
                    raise MemoryError("Request plus restore/repack headroom exceeds configured arenas; "
                                      "increase explicit arena budgets or reduce workload")
                break
            capacity = (request.len_input + request.len_output if (self.dense_reference or self.retain_full_reference)
                        else request.len_input + self.k + 1)
            region = coordinated_call(
                self.group, "admission allocation",
                lambda: FullKVRegion(self.gpu_pool, layers=self.config.num_layers,
                                     capacity=capacity, heads=self.config.num_kv_heads,
                                     dim=self.config.head_dim, dtype=self.dtype),
                lambda region: region.close(),
            )
            active.append(_State(request, region, list(request.token_ids)))
            pending.pop(0)
            self._log("admit", request_id=request.request_id, capacity_tokens=capacity)

    def _sample(self, logits):
        def operation():
            if self.rank != 0:
                return None
            tokens, probs = self.sampler(logits, self.metadata, logits.shape[0])
            return tokens.tolist(), probs
        result = coordinated_call(self.group, "sampling", operation)
        tokens = self._broadcast(result[0] if self.rank == 0 else None)
        return tokens, result[1] if self.rank == 0 else None

    def _forward(self, entries):
        # Each entry is (state, frame, input IDs, phase). Frame allocations are
        # coordinated by the caller, before the first TP model collective.
        def setup():
            adapter = PackedBatch([entry[1] for entry in entries])
            inputs = torch.tensor([token for _, _, ids, _ in entries for token in ids],
                                  dtype=torch.long, device=self.device)
            return adapter, inputs
        try:
            adapter, inputs = coordinated_call(self.group, "batch setup", setup,
                                               lambda pair: pair[0].abort())
        except Exception:
            for _, frame, _, _ in entries:
                if not frame._finished and not frame._aborted:
                    frame.abort()
            raise
        self.model.attach_kv_pool(adapter)
        try:
            logits = self.model(inputs)
            adapter.finish()
        except Exception:
            if not adapter._finished:
                adapter.abort()
            raise
        self.step += 1
        self.metrics["forwards"] += 1
        self.metrics["max_actual_batch"] = max(self.metrics["max_actual_batch"], len(entries))
        phases = [entry[3] for entry in entries]
        attention = adapter._gpu_attention.stats if adapter._gpu_attention is not None else None
        if attention is not None:
            aggregate = self.metrics.setdefault("attention_grid", dict(
                forwards=0, minimum_blocks=attention["grid_blocks"], maximum_blocks=0,
                estimated_idle_ratio_sum=0., workspace_peak_bytes=0))
            aggregate["forwards"] += 1
            aggregate["minimum_blocks"] = min(aggregate["minimum_blocks"], attention["grid_blocks"])
            aggregate["maximum_blocks"] = max(aggregate["maximum_blocks"], attention["grid_blocks"])
            aggregate["estimated_idle_ratio_sum"] += attention["model_idle_ratio"]
            aggregate["workspace_peak_bytes"] = max(aggregate["workspace_peak_bytes"], attention["workspace_bytes"])
        if "prefill" in phases and "draft" in phases:
            self.metrics["mixed_prefill_draft_forwards"] += 1
        if "verify" in phases and "draft" in phases:
            self.metrics["mixed_verify_draft_forwards"] += 1
        if "verify" in phases and "prefill" in phases:
            self.metrics["mixed_verify_prefill_forwards"] += 1
        self._log("forward", requests=[entry[0].request.request_id for entry in entries],
                  phases=phases, query_tokens=[len(entry[2]) for entry in entries],
                  logical_offsets=[entry[1].position_start for entry in entries], attention=attention)
        return self._sample(logits)

    def _complete(self, state, active, completed, restored=None):
        done = torch.cuda.current_stream(self.device).record_event()
        if restored is not None:
            restored.release(consumer_done=done)
        if isinstance(state.region, PackedKVRegion):
            state.region.close(consumer_done=done)
        else:
            state.region.close(consumer_done=done)
            done.synchronize()
        self.gpu_pool.reclaim()
        self.cpu_pool.reclaim()
        state.request.token_ids = array.array("i", state.prompt + state.output)
        state.request.draft_probs = None
        state.request.kv_cache_ptr = None
        completed.append(state.request)
        active.remove(state)
        self._log("complete", request_id=state.request.request_id,
                  output_tokens=len(state.output))

    def _emit(self, state, tokens):
        remaining = state.request.len_output - len(state.output)
        emitted = []
        for token in tokens[:remaining]:
            emitted.append(token)
            if token in self.eos:
                break
        state.output.extend(emitted)
        done = len(state.output) == state.request.len_output or (emitted and emitted[-1] in self.eos)
        return emitted, done

    def _prepare_verify(self, state):
        restored = None
        if isinstance(state.region, PackedKVRegion):
            restored = coordinated_call(
                self.group, "cold restore allocation",
                lambda: state.region.restore_chunk(0, state.region.cold_count,
                                                   stream=self.transfer_stream),
                lambda chunk: chunk.release(consumer_done=chunk.ready),
            )
            coordinated_call(self.group, "cold restore completion", lambda: restored.ready.synchronize())
            self.metrics["restores"] += 1
            self.metrics["h2d_bytes"] += state.region.cold_count * self.token_bytes
            self._log("restore", request_id=state.request.request_id, cold_tokens=state.region.cold_count)
        query = [state.output[-1]] + state.drafts
        def create_frame():
            if restored is not None:
                return PackedForward(state.region, mode="verify", query_tokens=len(query), restored=restored)
            return FullForward(state.region, query_tokens=len(query), write_offset=state.round_start, verify=True)
        try:
            frame = coordinated_call(self.group, "verify frame", create_frame, lambda item: item.abort())
            return frame, restored, query
        except Exception:
            if restored is not None:
                restored.release(consumer_done=restored.ready)
            raise

    def _commit_verify(self, state, active, completed, frame, restored, sampled, target_probs):
        """Commit one slice from the completed batched target forward."""
        try:
            def reject():
                if self.rank != 0:
                    return None
                if not state.drafts:
                    return sampled
                count = len(state.drafts)
                outputs = self.reject(
                    torch.tensor(state.drafts, dtype=torch.long, device=self.device), [count],
                    # Native rejection uses cumulative END offsets, without
                    # the leading zero used by attention indptr tensors.
                    torch.tensor([count], dtype=torch.int32, device=self.device), self.k,
                    torch.cat(state.draft_probs, dim=0).contiguous(), target_probs[:count].contiguous(),
                    torch.tensor([[sampled[-1]]], dtype=torch.long, device=self.device),
                )
                return self.reject.parse_output(outputs, self.config.vocab_size)[0]
            emitted = self._broadcast(coordinated_call(self.group, "rejection sampling", reject))
            proposed = len(state.drafts)
            retained, finished = self._emit(state, emitted)
            self.accepted += min(proposed, len(retained) - 1)
            self.proposed += proposed
            self.metrics["verify_forwards"] += 1
            state.round_id += 1
            if self._rounds is not None:
                self._rounds.write(json.dumps(dict(
                    request_id=state.request.request_id, round=state.round_id,
                    draft_length=proposed, budget_ratio=self.r,
                    accepted_draft_tokens=min(proposed, len(retained) - 1),
                    accepted_tokens=len(retained), logical_history=state.round_start,
                    full_context_round=restored is None,
                    hot_tokens=(state.region.hot_count if restored is not None else state.round_start),
                    cold_tokens=(state.region.cold_count if restored is not None else 0),
                )) + "\n")
                self._rounds.flush()
            state.drafts.clear()
            state.draft_probs.clear()
            target_probs = None
            if finished:
                self._complete(state, active, completed, restored)
                restored = None
                return
            frame.accept(len(retained))
            if self.retain_full_reference:
                state.round_start = state.region.length
                return
            old_bytes = state.region._lease.allocated_bytes if restored is None else 0
            hot_bytes, _, cold_bytes, _ = self._estimate(state.request, packed=True)
            capacities = dict(hot_capacity=hot_bytes // self.token_bytes,
                              cold_capacity=cold_bytes // self.token_bytes)
            def repack():
                if restored is None:
                    return frame.start_pack(budget_ratio=self.r, min_tokens=self.minimum,
                                            tail_capacity=self.k + 1, cpu_pool=self.cpu_pool,
                                            gpu_pool=self.gpu_pool, stream=self.transfer_stream,
                                            chunk_pages=self.chunk_pages, **capacities)
                return frame.start_repack(budget_ratio=self.r, min_tokens=self.minimum,
                                         cpu_pool=self.cpu_pool, gpu_pool=self.gpu_pool,
                                         stream=self.transfer_stream, chunk_pages=self.chunk_pages, **capacities)
            replacement = coordinated_call(
                self.group, "repack allocation", repack,
                lambda region: region.close(consumer_done=region.offload_ready),
            )
            coordinated_call(self.group, "repack completion", lambda: replacement.finish_offload())
            state.region = frame.adopt(replacement)
            restored = None  # adopt returns the old restored workspace.
            state.round_start = state.region.history_length
            self.gpu_pool.reclaim()
            self.cpu_pool.reclaim()
            self.metrics["migrations"] += 1
            self.metrics["d2h_bytes"] += replacement.cold_count * self.token_bytes
            self.metrics["full_bytes_returned"] += old_bytes
            self._log("pack", request_id=state.request.request_id,
                      hot_tokens=replacement.hot_count, cold_tokens=replacement.cold_count,
                      logical_history=replacement.history_length, returned_full_bytes=old_bytes)
        finally:
            if restored is not None and restored._lease is not None:
                done = torch.cuda.current_stream(self.device).record_event()
                restored.release(consumer_done=done)

    def _verify_group(self, ready):
        """Plan a feasible group before restoring anything; no OOM retry/fallback.

        Simulate first-fit allocation of all restores followed by all replacement
        hot/cold buffers. This is conservative: actual replacements are adopted
        sequentially and retire old buffers. The pool reservation stays fixed.
        """
        def choose():
            self.gpu_pool.reclaim()
            self.cpu_pool.reclaim()
            def reserve(intervals, size, alignment, *, from_end=False):
                size = (size + alignment - 1) // alignment * alignment
                if not size:
                    return True
                indices = range(len(intervals) - 1, -1, -1) if from_end else range(len(intervals))
                for i in indices:
                    offset, length = intervals[i]
                    if length >= size:
                        start = ((offset + length - size) // alignment * alignment
                                 if from_end else offset)
                        remaining = []
                        if start > offset:
                            remaining.append((offset, start - offset))
                        if start + size < offset + length:
                            remaining.append((start + size, offset + length - start - size))
                        intervals[i:i + 1] = remaining
                        return True
                return False
            chosen = 0
            for count in range(1, len(ready) + 1):
                gpu, cpu = list(self.gpu_pool._free), list(self.cpu_pool._free)
                replacements = []
                fits = True
                for state in ready[:count]:
                    packed = isinstance(state.region, PackedKVRegion)
                    restore = state.region.cold_count * self.token_bytes if packed else 0
                    fits &= reserve(gpu, restore, self.gpu_pool.alignment, from_end=True)
                    if not self.retain_full_reference and state.request.len_output - len(state.output) > 1:
                        hot_bytes, _, cold_bytes, _ = self._estimate(state.request, packed=True)
                        replacements.append((hot_bytes, cold_bytes))
                if not self.retain_full_reference:
                    for hot, cold in replacements:
                        fits &= reserve(gpu, hot, self.gpu_pool.alignment)
                        fits &= reserve(cpu, cold, self.cpu_pool.alignment)
                if not fits:
                    break
                chosen = count
            if not chosen:
                raise MemoryError("No verify request fits restore/repack arena headroom")
            return chosen
        count = self._broadcast(coordinated_call(self.group, "verify group capacity",
                                                lambda: choose() if self.rank == 0 else None))
        self._log("verify_group", ready=len(ready), selected=count,
                  requests=[s.request.request_id for s in ready[:count]])
        return ready[:count]

    def _ready_for_verify(self, state):
        return (bool(state.output) and not self.dense_reference
                and len(state.drafts) >= min(self.k, state.request.len_output - len(state.output) - 1))

    def _nonverify_entries(self, states):
        """Prepare non-ready requests without extending any draft beyond k."""
        if not states:
            return []
        entries = []
        def create_entries():
            for state in states:
                if self._ready_for_verify(state):
                    raise RuntimeError("A ready request must verify before drafting again")
                if not state.output:
                    start = state.region.length
                    ids = state.prompt[start:start + self.chunk_size]
                    frame = FullForward(state.region, query_tokens=len(ids))
                    phase = "prefill"
                else:
                    ids = [state.drafts[-1] if state.drafts else state.output[-1]]
                    frame = (PackedForward(state.region, mode="draft", query_tokens=1)
                             if isinstance(state.region, PackedKVRegion)
                             else FullForward(state.region, query_tokens=1))
                    phase = "ar" if self.dense_reference else "draft"
                entries.append((state, frame, ids, phase))
            return entries
        try:
            return coordinated_call(self.group, "prefill/draft frames", create_entries,
                                    lambda rows: [frame.abort() for _, frame, _, _ in rows])
        except Exception:
            for _, frame, _, _ in entries:
                if not frame._finished and not frame._aborted:
                    frame.abort()
            raise

    def _commit_nonverify(self, entries, sampled, probs, active, completed):
        offset = 0
        for state, _, ids, phase in entries:
            end = offset + len(ids)
            if phase == "prefill":
                if state.region.length == state.request.len_input:
                    _, finished = self._emit(state, [sampled[end - 1]])
                    state.round_start = state.region.length
                    if finished:
                        self._complete(state, active, completed)
            elif phase == "ar":
                _, finished = self._emit(state, [sampled[offset]])
                if finished:
                    self._complete(state, active, completed)
            else:
                state.drafts.append(sampled[offset])
                if self.rank == 0:
                    state.draft_probs.append(probs[offset:end].clone())
            offset = end

    def _verify_many(self, states, active, completed):
        prepared, others = [], []
        try:
            for state in states:
                frame, restored, query = self._prepare_verify(state)
                prepared.append((state, frame, restored, query))
            selected = {id(state) for state in states}
            # Ready requests excluded by capacity remain parked. Other requests
            # already own their tail/prefill storage and can compute in the same
            # model forward without adding a restore or replacement lease.
            others = self._nonverify_entries([state for state in active
                                             if id(state) not in selected and not self._ready_for_verify(state)])
            sampled, probs = self._forward([(s, f, q, "verify") for s, f, _, q in prepared] + others)
            self.metrics["verify_model_forwards"] += 1
            self.metrics["max_verify_batch"] = max(self.metrics["max_verify_batch"], len(prepared))
            histogram = self.metrics["verify_batch_histogram"]
            histogram[str(len(prepared))] = histogram.get(str(len(prepared)), 0) + 1
            offset = 0
            for state, frame, restored, query in prepared:
                end = offset + len(query)
                self._commit_verify(state, active, completed, frame, restored, sampled[offset:end],
                                    probs[offset:end] if probs is not None else None)
                offset = end
            self._commit_nonverify(others, sampled[offset:], probs[offset:] if probs is not None else None,
                                   active, completed)
        finally:
            for _, frame, _, _ in others:
                if not frame._finished and not frame._aborted:
                    frame.abort()
            for state, frame, restored, _ in prepared:
                if not frame._finished and not frame._aborted:
                    frame.abort()
                if restored is not None and restored._lease is not None:
                    restored.release(consumer_done=torch.cuda.current_stream(self.device).record_event())

    def _verify(self, state, active, completed):
        """Single-request entry retained for focused correctness diagnostics."""
        return self._verify_many([state], active, completed)

    def run(self, requests):
        if not requests:
            raise ValueError("Packed serving needs at least one request")
        for index, request in enumerate(requests):
            if request.len_input < 1 or request.len_output is None or request.len_output < 1:
                raise ValueError("Requests need nonempty input and positive bounded output length")
            request.request_id = index
        pending, active, completed = list(requests), [], []
        self._log("start", k=self.k, r=self.r, max_batch_size=self.max_batch_size,
                  dense_reference=self.dense_reference, retain_full_reference=self.retain_full_reference)
        try:
            while pending or active:
                self._admit(pending, active)
                # Restore and verify the largest ready prefix whose complete
                # restore/replacement peak fits the explicitly reserved arenas.
                ready = [state for state in active if self._ready_for_verify(state)]
                if ready:
                    self._verify_many(self._verify_group(ready), active, completed)
                    continue
                entries = self._nonverify_entries(active)
                sampled, probs = self._forward(entries)
                self._commit_nonverify(entries, sampled, probs, active, completed)
                probs = None
            torch.cuda.current_stream(self.device).synchronize()
            self.metrics.update(accepted_draft_tokens=self.accepted, proposed_draft_tokens=self.proposed,
                                gpu=self.gpu_pool.stats(), cpu=self.cpu_pool.stats())
            self._log("finish", metrics=self.metrics)
            if self.metrics["gpu"]["in_use_bytes"] or self.metrics["cpu"]["in_use_bytes"]:
                raise RuntimeError("Completed requests still own KV arena leases")
            return sorted(completed, key=lambda request: request.request_id)
        finally:
            for state in active:
                if not state.region._forward_active:
                    done = torch.cuda.current_stream(self.device).record_event()
                    state.region.close(consumer_done=done)
            if self._events is not None:
                self._events.close()
            if self._rounds is not None:
                self._rounds.close()
