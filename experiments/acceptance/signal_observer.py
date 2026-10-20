"""Read-only, rank-local signals for a frozen fixed-k serving experiment.

Selection event t is tagged with the *next* verify round. Current q/p is
diagnostic/label information; q-first is explicitly an after-one-draft feature.
GPU snapshots use event-polled pinned readbacks, without per-step device sync.
"""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch


class SignalObserver:
    def __init__(self, scheduler, output, rank):
        self.owner, self.output, self.rank = scheduler, Path(output), rank
        self.pending, self.prob_work, self.selection_work = deque(), [], []
        self.selection_context = None
        self.last_verified = {}
        self.counts = dict(selection=0, probability=0)
        self.waits, self.submit_seconds = 0, 0.
        self.layers, self.heads = scheduler.kv_pool.num_layers, scheduler.kv_pool.num_kv_heads
        self.k = scheduler.spec_stride
        self.files = {}
        for kind in self.counts:
            self.files[kind] = (
                (self.output / f"{kind}-signals.rank{rank}.jsonl").open("x"),
                (self.output / f"{kind}-signals.rank{rank}.f32").open("xb"))
        self.closed = False
        self.original_forward = scheduler.rejection_sampler.forward
        scheduler.rejection_sampler.forward = self.sampling
        # KVPool alternates two wrapper sets. Hook both, otherwise every other
        # verify event silently bypasses the observer.
        self.original_topks = []
        for wrappers in scheduler.kv_pool._wrappers:
            wrapper = wrappers[1]
            original = wrapper.run
            self.original_topks.append((wrapper, original))

            def hooked_topk(_original=original, **kwargs):
                return self.topk(_original, **kwargs)

            wrapper.run = hooked_topk
        cache_class = scheduler.spec_cache_class
        self.cache_class = cache_class
        self.original_selection = cache_class.dispatch_update_selected_indices.__func__
        observer = self

        def dispatch(cls, **kwargs):
            metadata = kwargs["input_metadata"]
            if metadata is None:
                observer.selection_context = []
            else:
                tasks = []
                for item in kwargs["selected_indices_metadatas"]:
                    prev = item["prev_idx"]
                    if prev >= 0 and metadata["request_type"][prev] == 2:
                        request = scheduler.active_requests[item["idx"]]
                        rid = request.request_id
                        tasks.append(dict(
                            request_id=rid, round=scheduler.acceptance_rounds[rid],
                            selection_step=scheduler.step_id,
                            context_tokens=request.len_kv_cache,
                            selected_kv_tokens=request.kv_cache_ptr.len_loading_indices,
                            selected_prefix_tokens=item["num_selected_tokens"],
                            eligible_prefix_tokens=item["num_effective_kv_len"],
                            source_accepted=observer.last_verified[rid][0],
                            source_draft_tokens=observer.last_verified[rid][1],
                            score_start=int(metadata["verify_kv_indptr"][prev]),
                            available="before_this_round_draft"))
                observer.selection_context = tasks
            try:
                return observer.original_selection(cls, **kwargs)
            finally:
                observer.selection_context = None

        cache_class.dispatch_update_selected_indices = classmethod(dispatch)

    def topk(self, original, **kwargs):
        values, indices = original(**kwargs)
        if self.selection_context:
            if len(self.selection_context) != kwargs["num_requests"]:
                raise RuntimeError("Selection signal row alignment failed")
            self.selection_work.append((list(self.selection_context), values, kwargs["in_val"]))
        return values, indices

    def sampling(self, **kwargs):
        # Snapshot identity before verify_outputs increments the round counter.
        requests = [r for r in self.owner.active_requests
                    if r.status.name == ("STALL" if self.owner.async_on else "VERIFY")]
        counts = list(map(int, kwargs["num_draft_tokens"]))
        if len(requests) != len(counts):
            raise RuntimeError("Probability signal request alignment failed")
        context = []
        for request, count in zip(requests, counts, strict=True):
            rid = request.request_id
            length, selected, cold = self.owner.draft_start[rid]
            context.append(dict(request_id=rid, round=self.owner.acceptance_rounds.get(rid, 0),
                verify_step=self.owner.step_id, actual_draft_tokens=count,
                context_tokens=length, selected_kv_tokens=selected, cold_start=cold,
                available="q_first_after_first_draft; p_alpha_after_verify"))
        result = self.original_forward(**kwargs)
        self.prob_work.append((context, kwargs["draft_probs"], kwargs["target_probs"],
                               kwargs["draft_token_ids"]))
        return result

    def verified(self, request, count, drafted):
        self.last_verified[request.request_id] = (count, drafted)

    def enqueue(self, kind, context, gpu):
        if len(self.pending) >= 64:
            self.waits += 1
            self.pending[0][0].synchronize()
            self.drain()
        cpu = torch.empty(tuple(gpu.shape), dtype=torch.float32, device="cpu", pin_memory=True)
        cpu.copy_(gpu, non_blocking=True)
        ready = torch.cuda.current_stream().record_event()
        self.pending.append((ready, kind, context, cpu, gpu))

    def submit(self):
        started = time.perf_counter()
        # This runs after native parse/update. It does not hold up the native
        # rejection result's mandatory CPU readback with diagnostic reductions.
        for context, values, scores in self.selection_work:
            planes = []
            for i, row in enumerate(context):
                count, length = row["selected_prefix_tokens"], row["eligible_prefix_tokens"]
                # The starts were sent to CUDA by native TopK; recover their
                # host counterparts from the original verify layout instead.
                start = row["score_start"]
                old = scores[..., start:start + length].float()
                chosen = values[i, ..., :count].float()
                retained = chosen.sum(-1)
                total = old.sum(-1)
                concentration = old.square().sum(-1)
                floor = chosen.amin(-1) if count else torch.zeros_like(total)
                planes.append(torch.stack((retained, total, concentration, floor)))
            self.enqueue("selection", context, torch.stack(planes))
        self.selection_work.clear()
        for context, q, p, ids in self.prob_work:
            positions = torch.arange(len(ids), dtype=torch.long, device=ids.device)
            proposed = ids.long()
            qp = q[positions, proposed]
            pp = p[positions, proposed]
            alpha = (pp / qp.clamp_min(1e-30)).clamp(0, 1)
            starts = np.r_[0, np.cumsum([r["actual_draft_tokens"] for r in context])[:-1]]
            starts_gpu = torch.tensor(starts, dtype=torch.long, device=q.device)
            first = q.index_select(0, starts_gpu)
            entropy = -(first * first.clamp_min(1e-30).log()).sum(-1)
            confidence = first.amax(-1)
            paths = []
            for i, (row, start) in enumerate(zip(context, starts, strict=True)):
                n = row["actual_draft_tokens"]
                end = int(start) + n
                trace = torch.full((3 * self.k + 2,), float("nan"), dtype=torch.float32,
                                   device=q.device)
                trace[:n] = alpha[int(start):end]
                trace[self.k:self.k + n] = qp[int(start):end]
                trace[2*self.k:2*self.k + n] = pp[int(start):end]
                trace[-2], trace[-1] = entropy[i], confidence[i]
                paths.append(trace)
            self.enqueue("probability", context, torch.stack(paths))
        self.prob_work.clear()
        self.submit_seconds += time.perf_counter() - started
        self.drain()

    def drain(self, force=False):
        while self.pending:
            ready, kind, context, cpu, gpu = self.pending[0]
            if force:
                ready.synchronize()
            elif not ready.query():
                break
            self.pending.popleft()
            array = cpu.numpy()
            if len(array) != len(context):
                raise RuntimeError("Signal readback shape mismatch")
            metadata, binary = self.files[kind]
            for row, values in zip(context, array, strict=True):
                if kind == "selection":
                    retained, total, squared, floor = values
                    if not np.isfinite(values).all() or np.any(total < -1e-5):
                        raise RuntimeError("Invalid selection probability mass")
                    dropped = np.maximum(total - retained, 0)
                    ratio = retained / np.maximum(total, 1e-20)
                    row.update(dropped_mass_mean=float(dropped.mean()),
                        dropped_mass_p90=float(np.quantile(dropped, .9)),
                        dropped_mass_max=float(dropped.max()),
                        retained_prefix_fraction_mean=float(ratio.mean()),
                        retained_prefix_fraction_p10=float(np.quantile(ratio, .1)))
                else:
                    n = row["actual_draft_tokens"]
                    alpha = values[:n]
                    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                        raise RuntimeError("Invalid conditional acceptance probabilities")
                    survival = np.cumprod(alpha.astype(np.float64))
                    row.update(conditional_mean=float(survival.sum()),
                               first_q_entropy=float(values[-2]), first_q_max=float(values[-1]))
                row["binary_row"] = self.counts[kind]
                self.counts[kind] += 1
                values.astype("<f4", copy=False).tofile(binary)
                metadata.write(json.dumps(row, separators=(",", ":")) + "\n")

    def flush(self):
        self.drain()
        for pair in self.files.values():
            for file in pair:
                file.flush()

    def close(self, success):
        if self.closed:
            return
        self.closed = True
        try:
            self.submit()
            self.drain(force=True)
        finally:
            for pair in self.files.values():
                for file in pair:
                    file.close()
            self.owner.rejection_sampler.forward = self.original_forward
            for wrapper, original in self.original_topks:
                wrapper.run = original
            self.cache_class.dispatch_update_selected_indices = classmethod(self.original_selection)
            (self.output / f"signal-summary.rank{self.rank}.json").write_text(json.dumps(dict(
                successful=success, rank=self.rank, k=self.k, counts=self.counts,
                selection_record_shape=[4, self.layers, self.heads],
                selection_planes=["selected_prefix_mass", "eligible_prefix_mass",
                                  "eligible_prefix_squared_scores", "minimum_selected_score"],
                probability_record_shape=[3*self.k + 2],
                probability_layout=[f"alpha[0:{self.k}]", f"q_token[{self.k}:{2*self.k}]",
                                    f"p_token[{2*self.k}:{3*self.k}]", "q_first_entropy", "q_first_max"],
                dtype="little-endian float32", readback_queue_waits=self.waits,
                cpu_submit_seconds=self.submit_seconds,
                notes="Read-only instrumentation; additional reduction/readback overhead. Selection scores average all previous verify queries, including uncommitted suffixes."
            ), indent=2) + "\n")


def install(module, name, output, rank):
    base = getattr(module, name)

    class SignalScheduler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.signals = SignalObserver(self, output, rank)
        def verify_outputs(self, request, output_token_ids, **kwargs):
            self.signals.verified(request, len(output_token_ids) - 1, request.len_draft_probs)
            return super().verify_outputs(request, output_token_ids, **kwargs)

        def update_requests(self, *args, **kwargs):
            result = super().update_requests(*args, **kwargs)
            self.signals.submit()
            if self.step_id % 100 == 0:
                self.signals.flush()
            return result

        def run(self, requests):
            success = False
            try:
                result = super().run(requests)
                success = True
                return result
            finally:
                self.signals.close(success)

    setattr(module, name, SignalScheduler)
