"""Rank-local packed KV and explicit offload/restore phase boundaries.

The packed scheduler uses this component at whole-forward phase boundaries.
Original page ownership remains with its caller until a completed migration
is explicitly adopted. No original token IDs are attached to packed KV.
"""

from dataclasses import dataclass
import math

import torch

from serve.attention.kv_buffer_pool import BufferLease, KVBufferPool
from serve.attention.kv_pack import pack_selected_kv


@dataclass
class PagePartition:
    full_pages: torch.Tensor
    hot_pages: torch.Tensor
    cold_pages: torch.Tensor
    selected_count: int
    recent_count: int

    @property
    def hot_count(self):
        return self.hot_pages.shape[-1]

    @property
    def cold_count(self):
        return self.cold_pages.shape[-1]


def partition_pages(
    full_pages, selected_pages, selected_count, *, pool_capacity, recent_count=0,
    validate=False,
):
    """Partition a request's physical pages independently per layer/local head.

    full_pages is [request_length], in request order; selected_pages is
    [layers, local_heads, selection_capacity]. The recent suffix is always
    hot and follows the selected prefix. Cold pages keep their original order.

    Only temporary ownership metadata is used; this does not create a
    permanent logical-token mapping. Work stays on the input device.
    The initial complement builder uses a GPU sort; a dedicated compaction
    kernel may replace it after the migration path is validated.

    Caller guarantees unique valid full pages and unique selected entries
    from the non-recent prefix. validate=True checks these preconditions but
    synchronizes CUDA tensors; reserve it for preflight/debugging.
    """
    if full_pages.ndim != 1 or selected_pages.ndim != 3:
        raise ValueError("Expected full_pages [N] and selection [layers, heads, capacity]")
    if full_pages.device != selected_pages.device:
        raise ValueError("Full and selected pages must be on the same device")
    if any(x.dtype not in (torch.int32, torch.int64) for x in (full_pages, selected_pages)):
        raise ValueError("Physical page numbers must be int32/int64")
    length = full_pages.numel()
    if type(pool_capacity) is not int or pool_capacity <= 0 or length > pool_capacity:
        raise ValueError("Invalid pool capacity")
    if (type(selected_count) is not int or not 0 <= selected_count <= selected_pages.shape[-1]
            or type(recent_count) is not int or not 0 <= recent_count <= length
            or selected_count + recent_count > length):
        raise ValueError("Invalid selected/recent counts")
    layers, heads = selected_pages.shape[:2]
    if min(layers, heads) <= 0:
        raise ValueError("Layer and local-head dimensions must be nonempty")
    selected = selected_pages[..., :selected_count]
    if validate:
        full_cpu, selected_cpu = full_pages.cpu().long(), selected.cpu().long()
        if (torch.any(full_cpu < 0) or torch.any(full_cpu >= pool_capacity)
                or torch.unique(full_cpu).numel() != length):
            raise ValueError("Full pages must be unique and within the physical pool")
        prefix = full_cpu[:length - recent_count]
        if not torch.isin(selected_cpu, prefix).all():
            raise ValueError("Selected pages must belong to the non-recent prefix")
        ordered = selected_cpu.sort(dim=-1).values
        if selected_count > 1 and torch.any(ordered[..., 1:] == ordered[..., :-1]):
            raise ValueError("Duplicate selection within a layer/head")

    # Translate physical pages only while constructing the complement.
    inverse = torch.full((pool_capacity,), -1, dtype=torch.int64, device=full_pages.device)
    inverse[full_pages.long()] = torch.arange(length, device=full_pages.device)
    positions = inverse[selected.long()]
    membership = torch.zeros((layers, heads, length), dtype=torch.bool, device=full_pages.device)
    membership.scatter_(-1, positions, True)
    if recent_count:
        membership[..., length - recent_count:] = True
    order = torch.arange(length, device=full_pages.device).expand(layers, heads, length)
    # Unique keys place cold entries first, with no device-to-host count read.
    cold_order = (order + membership.long() * length).sort(dim=-1).indices
    cold_count = length - selected_count - recent_count
    cold = full_pages[cold_order[..., :cold_count]].contiguous()
    recent = full_pages[length - recent_count:].expand(layers, heads, recent_count)
    hot = torch.cat((selected, recent), dim=-1).contiguous()
    return PagePartition(full_pages.clone(), hot, cold, selected_count, recent_count)


@dataclass
class RestoredChunk:
    kv: torch.Tensor | None
    offset: int
    count: int
    ready: torch.cuda.Event
    _lease: BufferLease | None = None

    def release(self, *, consumer_done):
        """Return an automatically pooled workspace after attention finishes.

        The caller records consumer_done after its last read. Externally
        supplied destinations remain owned and released by their caller.
        """
        if self._lease is None:
            raise RuntimeError("This restore workspace is not owned by a pool lease")
        self._lease.release(after=(self.ready, consumer_done))
        self._lease = None
        self.kv = None


class PackedKVRegion:
    """Compact immutable history plus a separately managed speculative tail.

    The first implementation waits once at finish_offload(), after all layers
    have forwarded and selection is finalized, before the next forward. TP
    ranks create separate instances using their own local heads/page IDs.
    The scheduler must still coordinate request state and capacity across TP
    ranks. This class performs no cross-rank collectives or pool-page frees.
    """

    @classmethod
    def start_offload(
        cls, source, plan, *, tail_capacity, chunk_pages=1024, source_ready=None,
        stream=None, num_programs=None, cpu_pool=None, gpu_pool=None,
        cold_source=None, source_hot_length=None, hot_capacity=None, cold_capacity=None,
    ):
        """Pack and offload, optionally using explicitly budgeted reusable arenas.

        Pass both pools together. Their capacities must be included in host
        and GPU budgets; they do not reduce the existing KVPool reservation.
        Pool exhaustion occurs before launches and leaves source ownership
        intact. Each TP rank supplies pools on its own local CUDA device.
        """
        if source.ndim != 6 or source.device.type != "cuda" or source.shape[2:4] != (2, 1):
            raise ValueError("Source must be CUDA KV with the original six-dimensional layout")
        if (not source.is_contiguous() or min(source.shape) <= 0
                or source.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError("Source must be nonempty contiguous floating KV")
        layers, _, _, _, heads, dim = source.shape
        for pages in (plan.hot_pages, plan.cold_pages):
            if (pages.ndim != 3 or pages.shape[:2] != (layers, heads)
                    or pages.device != source.device
                    or pages.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Partition geometry/device must match local source KV")
        if (plan.full_pages.ndim != 1 or plan.full_pages.device != source.device
                or plan.hot_count + plan.cold_count != plan.full_pages.numel()):
            raise ValueError("Partition must cover the request history")
        if cold_source is not None:
            if (cold_source.ndim != 6 or cold_source.shape[0] != layers
                    or cold_source.shape[2:] != source.shape[2:]
                    or cold_source.device != source.device or cold_source.dtype != source.dtype
                    or not cold_source.is_contiguous()
                    or type(source_hot_length) is not int
                    or not 0 < source_hot_length <= source.shape[1]):
                raise ValueError("Invalid rank-local hot/cold repacking sources")
        elif source_hot_length is not None:
            raise ValueError("source_hot_length requires restored cold KV")
        if (type(tail_capacity) is not int or tail_capacity < 0
                or type(chunk_pages) is not int or chunk_pages < 1):
            raise ValueError("Invalid tail/transfer capacity")
        hot_capacity = plan.hot_count + tail_capacity if hot_capacity is None else hot_capacity
        cold_capacity = plan.cold_count if cold_capacity is None else cold_capacity
        if (type(hot_capacity) is not int or hot_capacity < plan.hot_count + tail_capacity
                or type(cold_capacity) is not int or cold_capacity < plan.cold_count):
            raise ValueError("Reserved KV capacity must cover the live packed tensors")
        if num_programs is not None and (type(num_programs) is not int or num_programs < 1):
            raise ValueError("num_programs must be positive")
        stream = torch.cuda.current_stream(source.device) if stream is None else stream
        if stream.device != source.device:
            raise ValueError("Transfer stream must belong to the source device")
        if (cpu_pool is None) != (gpu_pool is None):
            raise ValueError("Provide both CPU and GPU arenas, or neither")
        if cpu_pool is not None:
            if (not isinstance(cpu_pool, KVBufferPool) or cpu_pool.device.type != "cpu"
                    or not cpu_pool.is_pinned):
                raise ValueError("Offload CPU arena must have pinned storage")
            if not isinstance(gpu_pool, KVBufferPool) or gpu_pool.device != source.device:
                raise ValueError("Offload GPU arena must belong to the source device")
        region = cls()
        region.device, region.dtype = source.device, source.dtype
        region.layers, region.heads, region.dim = layers, heads, dim
        region.history_length = plan.full_pages.numel()
        region.hot_count, region.cold_count = plan.hot_count, plan.cold_count
        region.selected_count, region.recent_count = plan.selected_count, plan.recent_count
        region.tail_capacity, region.tail_length = tail_capacity, 0
        region.hot_capacity, region.cold_capacity = hot_capacity, cold_capacity
        region.phase = "offloading"
        region._gpu_pool = gpu_pool
        region._leases = {}
        region._restore_events = []
        region._forward_active = region._forward_failed = region._needs_repack = False
        region._verification_pending = False
        region._forward_epoch = 0
        region._forward_ready = None
        # Repacking slots are not original KVPool pages and must never be freed
        # through that pool. Keep borrowed sources alive until copies finish.
        region._source_claimed = cold_source is not None
        region._source = (source, cold_source)
        region._source_pages = plan.full_pages.clone() if cold_source is None else None
        shapes = dict(
            hot=(layers, plan.hot_count + tail_capacity, 2, 1, heads, dim),
            cold=(layers, plan.cold_count, 2, 1, heads, dim),
        )
        reserved_shapes = dict(hot=(layers, hot_capacity, 2, 1, heads, dim),
                               cold=(layers, cold_capacity, 2, 1, heads, dim))
        def live_view(tensor, shape):
            # Slack is after the whole live tensor, not between its layers.
            # Thus both gather and H2D still operate on contiguous live KV.
            return tensor.reshape(-1)[:math.prod(shape)].view(shape)
        region._staging = None  # Direct mapped-host writes need no GPU staging.
        try:
            if cpu_pool is None:
                region._hot = live_view(torch.empty(reserved_shapes["hot"], dtype=source.dtype,
                                                     device=source.device), shapes["hot"])
                region._cold_cpu = live_view(torch.empty(reserved_shapes["cold"], dtype=source.dtype,
                                                         device="cpu", pin_memory=True), shapes["cold"])
            else:
                for name, pool in (("cold", cpu_pool), ("hot", gpu_pool)):
                    region._leases[name] = pool.allocate(reserved_shapes[name], source.dtype)
                region._hot = live_view(region._leases["hot"].tensor, shapes["hot"])
                region._cold_cpu = live_view(region._leases["cold"].tensor, shapes["cold"])
        except Exception:
            # Nothing was submitted, so a partial reservation is safe to return.
            for lease in region._leases.values():
                lease.release()
            raise
        # Retain selection tensors while kernels on another stream may use them.
        region._pending_plan = plan
        setup_ready = torch.cuda.current_stream(source.device).record_event()
        with torch.cuda.device(source.device), torch.cuda.stream(stream):
            try:
                stream.wait_event(setup_ready)
                if source_ready is not None:
                    stream.wait_event(source_ready)
                for tensor in (source, plan.hot_pages, plan.cold_pages, region._hot):
                    tensor.record_stream(stream)
                if cold_source is not None:
                    cold_source.record_stream(stream)
                pack_selected_kv(source, plan.hot_pages, region._hot, plan.hot_count,
                                 num_programs=num_programs, cold_source=cold_source,
                                 source_hot_length=source_hot_length)
                pack_selected_kv(source, plan.cold_pages, region._cold_cpu, plan.cold_count,
                                 num_programs=num_programs, cold_source=cold_source,
                                 source_hot_length=source_hot_length)
                region.offload_ready = stream.record_event()
            except Exception:
                # Earlier copies may already be in flight. Keep pooled CPU/GPU
                # intervals reserved until those copies have actually completed.
                failed_ready = stream.record_event()
                for lease in region._leases.values():
                    lease.release(after=(failed_ready,))
                if not region._leases:
                    failed_ready.synchronize()  # Protect standalone pinned CPU lifetime.
                raise
        return region

    @classmethod
    def start_repartition(cls, previous, restored, plan, *, cpu_pool=None, gpu_pool=None,
                          tail_capacity=None, chunk_pages=1024, stream=None,
                          source_ready=None, num_programs=None, hot_capacity=None, cold_capacity=None):
        """Build next-round hot/cold from compact history plus accepted tail.

        The temporary slot space is [previous.hot[:hot_length], restored.cold].
        The old region and restored workspace remain owned by the caller.
        After finish_offload() and TP readiness coordination, explicitly adopt
        the new region before retiring those sources.
        """
        previous._require_packed()
        if getattr(previous, "_forward_active", False):
            raise RuntimeError("Complete every layer before repartitioning KV")
        if (restored.kv is None or restored.offset != 0 or restored.count != previous.cold_count
                or restored.kv.shape[0] != previous.layers
                or restored.kv.shape[1] < restored.count
                or plan.full_pages.numel() != previous.history_length + previous.tail_length):
            raise ValueError("Repartitioning requires full cold history and accepted tail")
        torch.cuda.current_stream(previous.device).wait_event(restored.ready)
        replacement = cls.start_offload(
            previous.hot_kv, plan,
            tail_capacity=previous.tail_capacity if tail_capacity is None else tail_capacity,
            cpu_pool=cpu_pool, gpu_pool=gpu_pool, chunk_pages=chunk_pages,
            source_ready=source_ready, stream=stream, num_programs=num_programs,
            cold_source=restored.kv, source_hot_length=previous.hot_length,
            hot_capacity=hot_capacity, cold_capacity=cold_capacity,
        )
        replacement._previous_region = previous
        replacement._restored_source = restored
        return replacement

    @staticmethod
    def _copy_contiguous_layers(destination, source):
        # A whole-history transfer can use one DMA instead of one per layer.
        # Proper subranges retain layer gaps, so copy each contiguous layer.
        if destination.is_contiguous() and source.is_contiguous():
            destination.copy_(source, non_blocking=True)
        else:
            for layer in range(source.shape[0]):
                destination[layer].copy_(source[layer], non_blocking=True)

    def finish_offload(self, *, wait=True):
        """Wait at the explicit migration boundary, then permit source retirement."""
        if self.phase == "packed":
            return True
        if self.phase != "offloading":
            raise RuntimeError("Region is closed")
        if wait:
            self.offload_ready.synchronize()
        elif not self.offload_ready.query():
            return False
        self.phase = "packed"
        self._source = self._staging = self._pending_plan = None
        return True

    def source_pages_to_release(self):
        """Hand off original pages once; caller must detach its old owner first."""
        self._require_packed()
        if self._source_claimed:
            raise RuntimeError("Original pages have already been handed off")
        self._source_claimed = True
        pages, self._source_pages = self._source_pages, None
        return pages

    def close(self, *, consumer_done, wait=True):
        """Retire packed storage after all copies and the final hot-KV reader.

        consumer_done must cover hot/tail use on all compute streams. Cold
        CPU storage also waits for every outstanding H2D. With pools, wait=False
        returns ownership immediately and leaves reclamation to event polling.
        Original KVPool pages and restored chunks remain caller-owned.
        """
        if self.phase == "closed":
            return
        if getattr(self, "_forward_active", False):
            raise RuntimeError("Finish or fail the active forward before closing packed KV")
        if (not callable(getattr(consumer_done, "query", None))
                or not callable(getattr(consumer_done, "synchronize", None))):
            raise ValueError("Record an event after the final packed-KV consumer")
        if not wait and not self._leases:
            raise ValueError("Nonblocking retirement requires pooled storage")
        dependencies = (self.offload_ready, *self._restore_events, consumer_done)
        if wait:
            for event in dependencies:
                event.synchronize()
        for name, lease in self._leases.items():
            events = ((self.offload_ready, *self._restore_events) if name == "cold"
                      else (self.offload_ready, consumer_done))
            lease.release(after=() if wait else events)
        self._leases.clear()
        self._source = self._staging = self._pending_plan = None
        self._hot = self._cold_cpu = self._source_pages = None
        self._restore_events.clear()
        self._previous_region = self._restored_source = None
        self.phase = "closed"

    def _require_packed(self):
        if self.phase != "packed":
            raise RuntimeError("Finish the offload phase before using packed KV")
        if getattr(self, "_forward_failed", False):
            raise RuntimeError("Packed KV was partially overwritten by a failed forward")

    @property
    def hot_kv(self):
        self._require_packed()
        return self._hot

    @property
    def cold_cpu(self):
        self._require_packed()
        return self._cold_cpu

    @property
    def hot_length(self):
        return self.hot_count + self.tail_length

    def write_tail(self, layer, keys, values, offset):
        """Write this layer's ordered tail; publish tail length after all layers."""
        self._require_packed()
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[1:] != (self.heads, self.dim):
            raise ValueError("Tail K/V must be [query_tokens, local_heads, head_dim]")
        if (keys.device != self.device or values.device != self.device
                or keys.dtype != self.dtype or values.dtype != self.dtype):
            raise ValueError("Tail K/V device and dtype must match the region")
        if (type(layer) is not int or not 0 <= layer < self.layers
                or type(offset) is not int or offset < 0
                or offset + keys.shape[0] > self.tail_capacity):
            raise ValueError("Invalid tail write range")
        start, end = self.hot_count + offset, self.hot_count + offset + keys.shape[0]
        self._hot[layer, start:end, 0, 0].copy_(keys)
        self._hot[layer, start:end, 1, 0].copy_(values)

    def set_tail_length(self, length):
        """Publish/truncate tail after writes on all layers are ordered before reads."""
        self._require_packed()
        if type(length) is not int or not 0 <= length <= self.tail_capacity:
            raise ValueError("Invalid effective tail length")
        self.tail_length = length

    def restore_chunk(self, offset, count, *, destination=None, stream=None, consumer_done=None):
        """Restore a contiguous cold slice; keep the immutable CPU backing.

        Caller waits for returned ready before attention. A reused destination
        must no longer be read; consumer_done can establish that dependency.
        The returned tensor may have capacity greater than its effective count.
        """
        self._require_packed()
        if (type(offset) is not int or type(count) is not int or offset < 0
                or count < 0 or offset + count > self.cold_count):
            raise ValueError("Invalid cold restore range")
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Restore stream must belong to the region device")
        lease = None
        if destination is None:
            shape = (self.layers, count, 2, 1, self.heads, self.dim)
            if self._gpu_pool is None:
                destination = torch.empty(shape, dtype=self.dtype, device=self.device)
            else:
                lease = self._gpu_pool.allocate(shape, self.dtype, from_end=True)
                destination = lease.tensor
        if (destination.ndim != 6 or destination.shape[0] != self.layers
                or destination.shape[1] < count
                or destination.shape[2:] != (2, 1, self.heads, self.dim)
                or destination.device != self.device or destination.dtype != self.dtype
                or not destination.is_contiguous()):
            raise ValueError("Invalid rank-local restore destination")
        if (destination.nbytes and self._hot.nbytes
                and destination.data_ptr() < self._hot.data_ptr() + self._hot.nbytes
                and self._hot.data_ptr() < destination.data_ptr() + destination.nbytes):
            raise ValueError("Restore workspace must not overlap selected KV or its tail")
        setup_ready = torch.cuda.current_stream(self.device).record_event()
        self._restore_events = [event for event in self._restore_events if not event.query()]
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            try:
                stream.wait_event(setup_ready)
                if consumer_done is not None:
                    stream.wait_event(consumer_done)
                destination.record_stream(stream)
                self._copy_contiguous_layers(
                    destination[:, :count], self._cold_cpu[:, offset:offset + count],
                )
                ready = stream.record_event()
            except Exception:
                failed_ready = stream.record_event()
                self._restore_events.append(failed_ready)
                if lease is not None:
                    lease.release(after=(failed_ready,))
                raise
            self._restore_events.append(ready)
        return RestoredChunk(destination, offset, count, ready, lease)
