"""Bounded diagnostic snapshots of actual draft indices, including recent KV.

Copies use the producing control stream. CPU reads require Event.query();
only closing the diagnostic waits for its remaining events. Snapshot work
still perturbs execution, so these runs are not throughput measurements.
"""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.reproduction.footprint_stats import summarize_selection


class SelectionProbe:
    def __init__(self, output, rank, settings):
        self.directory = Path(output) / f"selection.rank{rank}"
        self.directory.mkdir()
        self.settings = settings
        self.rank = rank
        self.samples = (self.directory / "samples.jsonl").open("w")
        self.windows_file = (self.directory / "verify-windows.jsonl").open("w")
        self.pending = deque()
        self.next_verify = {}
        self.previous_union = {}
        self.submitted = self.written = self.dropped = 0
        self.pending_bytes = self.peak_pending_bytes = 0

    def capture(self, scheduler, host, device):
        self.drain()
        for request in scheduler.active_requests:
            if request.status.name == "VERIFY":
                for sample, started in self.next_verify.pop(request.request_id, []):
                    self.windows_file.write(json.dumps(dict(
                        sample=sample, request_id=request.request_id, next_verify_step=scheduler.step_id,
                        host_prepare_gap_seconds=time.perf_counter() - started)) + "\n")
        if (scheduler.step_id % self.settings["interval"]
                or self.submitted >= self.settings["max_samples"]):
            return
        candidates = [(idx, request) for idx, request in enumerate(scheduler.active_requests)
                      if request.status.name == "DRAFT"
                      and request.len_kv_cache >= self.settings["min_context_tokens"]
                      and request.kv_cache_ptr.len_selected_indices_gpu > 0]
        if not candidates:
            return
        if len(self.pending) >= 4:
            self.dropped += 1
            return
        idx, request = candidates[self.submitted % len(candidates)]
        cache = request.kv_cache_ptr
        start = int(host["kv_indptr"][idx])
        count = request.len_loading_kv_cache
        physical = np.asarray(list(cache.indices), dtype=np.int32)
        if cache.page_size != 1 or len(physical) != request.len_kv_cache:
            raise ValueError("Snapshot assumes the current one-token KV pages")
        source = device["flatten_indices"][..., start:start + count]
        if source.ndim != 3 or source.shape[-1] != count or count <= 0:
            raise ValueError("Invalid effective draft buffer slice")
        cpu = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
        cpu.copy_(source, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        sample = self.submitted
        row = dict(sample=sample, rank=self.rank, request_id=request.request_id,
                   step_id=scheduler.step_id, spec_stride=scheduler.spec_stride,
                   budget_ratio=scheduler.spec_cache_args["budget_ratio"],
                   context_tokens=request.len_kv_cache, draft_kv_tokens=count,
                   selected_prefix_tokens=cache.len_selected_indices_gpu,
                   recent_tokens=count - cache.len_selected_indices_gpu, cold_start=False,
                   active_requests=len(scheduler.active_requests),
                   kv_free_pages=scheduler.kv_pool.num_free_pages,
                   kv_pool_capacity=scheduler.kv_pool.capacity,
                   head_dim=scheduler.kv_pool.head_dim,
                   element_bytes=scheduler.kv_pool._physical_mem_gpu.element_size())
        self.next_verify.setdefault(request.request_id, []).append((sample, time.perf_counter()))
        size = cpu.numel() * cpu.element_size()
        self.pending.append((event, cpu, physical, row, size))
        self.pending_bytes += size
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        self.submitted += 1

    def drain(self, wait=False):
        while self.pending:
            event, cpu, physical, row, size = self.pending[0]
            if wait:
                event.synchronize()
            elif not event.query():
                break
            self.pending.popleft()
            self.pending_bytes -= size
            mapping = {int(page): token for token, page in enumerate(physical)}
            if len(mapping) != len(physical):
                raise ValueError("Duplicate physical pages in request")
            values = cpu.numpy()
            logical = np.fromiter((mapping[int(page)] for page in values.flat),
                                  dtype=np.int32, count=values.size).reshape(values.shape)
            row.update(summarize_selection(logical, row["context_tokens"], row["head_dim"], row["element_bytes"]))
            union = np.unique(logical)
            previous = self.previous_union.get(row["request_id"])
            row["previous_sample_step"] = previous[0] if previous else None
            row["new_union_tokens_since_previous_sample"] = (
                int(len(np.setdiff1d(union, previous[1]))) if previous else None)
            self.previous_union[row["request_id"]] = (row["step_id"], union)
            filename = f"sample{row['sample']:05d}.npz"
            np.savez_compressed(self.directory / filename, logical_indices=logical,
                                physical_indices=values, physical_to_logical_page_order=physical)
            row["indices_file"] = filename
            self.samples.write(json.dumps(row) + "\n")
            self.samples.flush()
            self.written += 1

    def close(self, successful):
        try:
            self.drain(wait=True)
        finally:
            self.samples.close()
            self.windows_file.close()
            (self.directory / "summary.json").write_text(json.dumps(dict(
                successful=successful, settings=self.settings, submitted=self.submitted,
                written=self.written, dropped_queue_full=self.dropped,
                peak_pending_host_bytes=self.peak_pending_bytes,
                unresolved_verify_windows=sum(map(len, self.next_verify.values())),
                timing_note="Host prepare gap only, not GPU/DMA latency"), indent=2) + "\n")


def scheduler_class(base_scheduler, output, rank, settings):
    class SelectionObservedScheduler(base_scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.selection_probe = SelectionProbe(output, rank, settings)

        def prepare_forward_inputs(self):
            host, device = super().prepare_forward_inputs()
            if host is not None:
                self.selection_probe.capture(self, host, device)
            return host, device

        def run(self, requests):
            successful = False
            try:
                result = super().run(requests)
                successful = True
                return result
            finally:
                self.selection_probe.close(successful)

    return SelectionObservedScheduler
