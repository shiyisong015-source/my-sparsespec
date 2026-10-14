"""Dense KV ownership for chunked prefill and the first full-context round.

Full and packed histories lease the same GPU arena. A full lease is returned
only after hot/cold packing finishes, so admission can reuse its actual bytes.
"""

import math

import torch

from serve.attention.kv_partition import PackedKVRegion, partition_pages
from serve.attention.packed_attention import attention_segment, selection_scores


class FullKVRegion:
    def __init__(self, pool, *, layers, capacity, heads, dim, dtype):
        self._lease = pool.allocate((layers, capacity, 2, 1, heads, dim), dtype, from_end=True)
        self.kv = self._lease.tensor
        self.device, self.dtype = self.kv.device, dtype
        self.layers, self.capacity, self.heads, self.dim = layers, capacity, heads, dim
        self.length = 0
        self._forward_active = self._forward_failed = False
        self._forward_ready = None
        self._forward_epoch = 0
        self._closed = False

    def close(self, *, consumer_done=None):
        if self._closed:
            return
        if self._forward_active:
            raise RuntimeError("Finish or abort the full forward before releasing KV")
        events = tuple(event for event in (self._forward_ready, consumer_done) if event is not None)
        self._lease.release(after=events)
        self.kv = None
        self._closed = True


class FullForward:
    """Same model adapter contract as PackedForward, with dense causal attention."""

    def __init__(self, region, *, query_tokens, write_offset=None, verify=False,
                 backend="flashinfer"):
        if region._closed or region._forward_active or region._forward_failed:
            raise RuntimeError("Full KV is not available for a fresh forward")
        if type(query_tokens) is not int or query_tokens < 1:
            raise ValueError("query_tokens must be positive")
        write_offset = region.length if write_offset is None else write_offset
        if type(write_offset) is not int or not 0 <= write_offset <= region.length:
            raise ValueError("Invalid full KV write offset")
        if write_offset + query_tokens > region.capacity:
            raise ValueError("Full KV capacity is insufficient")
        self.region, self.query_tokens = region, query_tokens
        self.position_start = self.write_offset = write_offset
        self.verify, self.backend = verify, backend
        self._layers_done = set()
        self._finished = self._failed = self._aborted = self._accepted = False
        self._compute_stream = None
        self.ready = None
        self._epoch = region._forward_epoch + 1
        self.scores = (torch.empty((region.layers, region.heads, write_offset + query_tokens),
                                  dtype=torch.float16, device=region.device) if verify else None)
        self._setup_ready = (torch.cuda.current_stream(region.device).record_event()
                             if region.device.type == "cuda" else None)
        region._forward_active = True

    def prepare_layer(self, layer, q, keys, values):
        """Validate and establish dependencies before either batch or reference writes."""
        region = self.region
        if self._finished or self._failed or layer in self._layers_done:
            raise RuntimeError("Full frame layer is no longer writable")
        if (type(layer) is not int or not 0 <= layer < region.layers
                or q.ndim != 3 or q.shape[0] != self.query_tokens
                or q.shape[1] < 1 or q.shape[1] % region.heads or q.shape[-1] != region.dim
                or keys.shape != (self.query_tokens, region.heads, region.dim)
                or values.shape != keys.shape
                or any(t.device != region.device or t.dtype != region.dtype for t in (q, keys, values))):
            raise ValueError("Full frame Q/K/V geometry mismatch")
        self._prepare_batch_layer(layer, torch.cuda.current_stream(q.device)
                                  if q.device.type == "cuda" else None)

    def _prepare_batch_layer(self, layer, current):
        """Use batch-validated Q/K/V geometry while preserving frame guards."""
        region = self.region
        if self._finished or self._failed or layer in self._layers_done:
            raise RuntimeError("Full frame layer is no longer writable")
        if type(layer) is not int or not 0 <= layer < region.layers:
            raise ValueError("Invalid layer")
        if current is not None:
            if self._compute_stream is not None and current != self._compute_stream:
                raise RuntimeError("Full frame layers must share one compute stream")
            if self._compute_stream is None:
                current.wait_event(self._setup_ready)
                if region._forward_ready is not None:
                    current.wait_event(region._forward_ready)
                for tensor in (region.kv, self.scores):
                    if tensor is not None:
                        tensor.record_stream(current)
            self._compute_stream = current
        self._writes_started = True

    def complete_layer(self, layer):
        self._layers_done.add(layer)

    def layer(self, layer, q, keys, values):
        self.prepare_layer(layer, q, keys, values)
        region = self.region
        try:
            end = self.write_offset + self.query_tokens
            region.kv[layer, self.write_offset:end, 0, 0].copy_(keys)
            region.kv[layer, self.write_offset:end, 1, 0].copy_(values)
            kv = region.kv[layer, :end]
            state = attention_segment(q, kv, causal=True, backend=self.backend)
            if self.verify:
                selection_scores(q, kv, state.lse, causal=True,
                                 backend=self.backend, out=self.scores[layer])
        except Exception:
            self._failed = region._forward_failed = True
            region._forward_active = False
            raise
        self._layers_done.add(layer)
        return state.output

    def finish(self):
        if self._failed or self._finished or len(self._layers_done) != self.region.layers:
            raise RuntimeError("Finish every full frame layer exactly once")
        self.region.length = self.write_offset + self.query_tokens
        self.ready = self._compute_stream.record_event() if self._compute_stream is not None else None
        self.region._forward_ready = self.ready
        self.region._forward_epoch = self._epoch
        self.region._forward_active = False
        self._finished = True
        return self.scores

    def abort(self):
        if self._finished:
            raise RuntimeError("Cannot abort a finished full frame")
        if not self._aborted:
            self._aborted = self._failed = True
            self.region._forward_active = False
            if self._layers_done or getattr(self, "_writes_started", False) or self.region._forward_failed:
                self.region._forward_failed = True
            self.ready = self._compute_stream.record_event() if self._compute_stream is not None else None
            if self.ready is not None:
                self.region._forward_ready = self.ready
        return self.ready

    def accept(self, count):
        if (not self.verify or not self._finished or self._failed or self._accepted
                or self.region._forward_epoch != self._epoch
                or type(count) is not int or not 1 <= count <= self.query_tokens):
            raise ValueError("Invalid full verification commit")
        self.region.length = self.write_offset + count
        self._accepted = True
        return self.region.length

    def start_pack(self, *, budget_ratio, min_tokens, tail_capacity, cpu_pool, gpu_pool,
                   stream, chunk_pages=1024, hot_capacity=None, cold_capacity=None):
        if not self._accepted or getattr(self, "_replacement", None) is not None:
            raise RuntimeError("Commit this full verification before packing it once")
        if (not math.isfinite(budget_ratio) or not 0 < budget_ratio <= 1
                or type(min_tokens) is not int or min_tokens < 1 or self.write_offset < 1):
            raise ValueError("Invalid full-to-packed selection")
        length = self.region.length
        recent = length - self.write_offset + 1
        budget = min(length, max(min_tokens, math.ceil(length * budget_ratio), recent))
        count = budget - recent
        selected = torch.topk(self.scores[..., :self.write_offset - 1], count,
                              dim=-1, sorted=True).indices.int()
        full = torch.arange(length, dtype=torch.int32, device=self.region.device)
        plan = partition_pages(full, selected, count, pool_capacity=self.region.capacity,
                               recent_count=recent)
        replacement = PackedKVRegion.start_offload(
            self.region.kv, plan, tail_capacity=tail_capacity,
            cpu_pool=cpu_pool, gpu_pool=gpu_pool, stream=stream,
            chunk_pages=chunk_pages, source_ready=self.ready,
            hot_capacity=hot_capacity, cold_capacity=cold_capacity,
        )
        self._replacement = replacement
        return replacement

    def adopt(self, replacement):
        if (replacement is not getattr(self, "_replacement", None)
                or replacement.phase != "packed" or not self._accepted):
            raise RuntimeError("Finish packing on all ranks before releasing full KV")
        # These are slots in this lease, not IDs owned by legacy KVPool.
        replacement.source_pages_to_release()
        self.region.close(consumer_done=replacement.offload_ready)
        self.scores = None
        return replacement
