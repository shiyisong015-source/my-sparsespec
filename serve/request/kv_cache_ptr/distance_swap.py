"""Chunked chronological backing with an independent, stable Draft view.

Unlike compact repack, raw historical pages are saved in chronological order.
This costs a CPU copy of the selected prefix too, but lets each physical page
be released immediately after its own D2H, without per-head release hazards.
All host mutations must run under the scheduler's residency lock.
"""
import array
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from serve.attention.kv_pack import gather_to_pool, pack_selected_kv, restore_to_pool


@triton.jit
def _hot_from_host(source, positions, destinations, output, offset, count,
                   n, heads: tl.constexpr, dim: tl.constexpr,
                   sl, sp, sk, sh, il, ih, dl, dp, dk, dh,
                   WIDTH: tl.constexpr):
    layer, head = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    j = tl.program_id(0) * 8 + tl.arange(0, 8)
    pos = tl.load(positions + layer * il + head * ih + j, j < n, 0)
    dst = tl.load(destinations + j, j < n, 0).to(tl.int64)
    lane = tl.arange(0, WIDTH)
    kv, channel = lane // dim, lane % dim
    mask = ((j[:, None] < n) & (pos[:, None] >= offset)
            & (pos[:, None] < offset + count) & (lane[None, :] < 2 * dim))
    value = tl.load(source + layer * sl + (pos[:, None] - offset).to(tl.int64) * sp
                    + kv[None, :] * sk + head * sh + channel[None, :], mask, 0)
    tl.store(output + layer * dl + dst[:, None] * dp
             + kv[None, :] * dk + head * dh + channel[None, :], value, mask)


class CUDASwapTransport:
    def __init__(self, pool):
        self.pool = pool
        self.down = torch.cuda.Stream(device=pool.device)
        self.up = torch.cuda.Stream(device=pool.device)
        self.measure = False
        self.timings = {}

    def _timing_start(self, stream):
        if not self.measure:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def _timing_end(self, stream, start):
        event = torch.cuda.Event(enable_timing=start is not None)
        event.record(stream)
        if start is not None:
            self.timings[id(event)] = start, event
        return event

    def pop_timing(self, event):
        pair = self.timings.pop(id(event), None)
        return pair[0].elapsed_time(pair[1]) if pair is not None else None

    def selection(self, ptr, history):
        source = torch.tensor(history, dtype=torch.int32, device=self.pool.device)
        chosen = ptr.selected_indices_buffer_gpu[..., :ptr.len_selected_indices_gpu].clone()
        inverse = torch.full((self.pool.capacity,), -1, dtype=torch.int32, device=self.pool.device)
        inverse[source.long()] = torch.arange(len(history), dtype=torch.int32, device=self.pool.device)
        positions = inverse[chosen.long()]
        return chosen, positions, torch.cuda.current_stream(self.pool.device).record_event()

    def cpu_buffer(self, count):
        physical = self.pool._physical_mem_gpu
        return torch.empty((physical.shape[0], count, *physical.shape[2:]),
                           dtype=physical.dtype, device="cpu", pin_memory=True)

    def hot(self, storage, pages, from_host=False):
        physical = self.pool._physical_mem_gpu
        with torch.cuda.stream(self.up):
            self.up.wait_event(storage.source_ready)
            start = self._timing_start(self.up)
            if from_host:
                destinations = torch.tensor(pages, dtype=torch.int32, device=self.pool.device)
                pos = storage.positions
                for chunk in storage.chunks:
                    _hot_from_host[(triton.cdiv(len(pages), 8), physical.shape[0], physical.shape[4])](
                        chunk.cpu, pos, destinations, physical, chunk.offset, chunk.count,
                        len(pages), physical.shape[4], physical.shape[5],
                        chunk.cpu.stride(0), chunk.cpu.stride(1), chunk.cpu.stride(2), chunk.cpu.stride(4),
                        pos.stride(0), pos.stride(1),
                        physical.stride(0), physical.stride(1), physical.stride(2), physical.stride(4),
                        WIDTH=triton.next_power_of_2(2 * physical.shape[5]), num_warps=8)
            else:
                gather_to_pool(physical, storage.chosen, pages,
                               source_page_ids=tuple(p for p in storage.history if p is not None))
            ptr = storage.ptr
            destinations = torch.tensor(pages, dtype=torch.int32, device=self.pool.device)
            ptr.selected_indices_buffer_gpu[..., :len(pages)].copy_(destinations)
            return self._timing_end(self.up, start)

    def download(self, storage, chunk):
        pages = storage.history[chunk.offset:chunk.offset + chunk.count]
        with torch.cuda.stream(self.down):
            self.down.wait_event(storage.source_ready)
            if storage.hot_event is not None:
                self.down.wait_event(storage.hot_event)
            start = self._timing_start(self.down)
            ids = torch.tensor(pages, dtype=torch.int32, device=self.pool.device)
            selected = ids.expand(storage.ptr.NUM_LAYERS, storage.ptr.NUM_KV_HEADS, -1).contiguous()
            pack_selected_kv(self.pool._physical_mem_gpu, selected, chunk.cpu, chunk.count,
                             num_programs=8)
            return self._timing_end(self.down, start), selected

    def upload(self, storage, chunk, pages):
        with torch.cuda.stream(self.up):
            start = self._timing_start(self.up)
            restore_to_pool(chunk.cpu, self.pool._physical_mem_gpu, pages)
            return self._timing_end(self.up, start)

    def synchronize(self):
        self.down.synchronize()
        self.up.synchronize()


@dataclass
class Chunk:
    offset: int
    count: int
    cpu: object = None
    saved: bool = False
    job: object = None


@dataclass
class ChunkJob:
    direction: str
    ready: object
    pages: tuple
    keepalive: object = None


class SwapStorage:
    """One frozen selection epoch; appends remain in ptr.tail_pages."""
    def __init__(self, ptr, transport, chunk_pages, generation):
        if ptr.needs_pack or ptr.phase != "resident":
            raise RuntimeError("Adopt completed selection before creating swap storage")
        self.ptr, self.transport, self.generation = ptr, transport, generation
        self.recent = tuple(ptr.indices[-ptr.recent_count:]) if ptr.recent_count else ()
        self.history = list(ptr.indices[:len(ptr.indices) - len(self.recent)])
        self.chunks = [Chunk(i, min(chunk_pages, len(self.history) - i))
                       for i in range(0, len(self.history), chunk_pages)]
        self.chosen, self.positions, self.source_ready = transport.selection(ptr, self.history)
        self.hot_pages, self.hot_event = (), None
        self.hot_ready = ptr.len_selected_indices_gpu == 0
        self.pinned = False
        self.retired = False

    @property
    def cpu_pages(self):
        return sum(c.count for c in self.chunks if c.cpu is not None)

    @property
    def busy(self):
        return any(c.job is not None for c in self.chunks) or (self.hot_event is not None)

    @property
    def complete(self):
        return all(p is not None for p in self.history) and not any(
            c.job is not None for c in self.chunks)

    @property
    def missing(self):
        return sum(p is None for p in self.history)

    def start_hot(self):
        if self.hot_ready or self.hot_event is not None:
            return False
        count = self.ptr.len_selected_indices_gpu
        if self.ptr.kv_pool.num_free_pages < count:
            return False
        if any(c.job is not None and c.job.direction == "down" for c in self.chunks):
            return False
        from_host = any(p is None for p in self.history)
        if from_host and not all(c.saved for c in self.chunks):
            return False
        pages = self.ptr._allocate(count)
        try:
            self.hot_event = self.transport.hot(self, pages, from_host)
        except BaseException:
            self.transport.synchronize()
            self.ptr._release(pages)
            raise
        self.hot_pages = pages
        return True

    def publish(self):
        ptr = self.ptr
        if self.complete:
            ptr.indices = array.array("i", self.history + list(self.recent) + ptr.tail_pages)
            ptr.phase = "resident"
        else:
            ptr.indices = array.array("i", self.hot_pages + self.recent + tuple(ptr.tail_pages))
            ptr.phase = "hot"
        ptr.hot_pages = self.hot_pages + self.recent
        ptr.len_loading_indices = ptr.len_selected_indices_gpu + len(self.recent) + len(ptr.tail_pages)

    def start_down(self, chunk):
        if chunk.job is not None or any(p is None for p in self.history[chunk.offset:chunk.offset + chunk.count]):
            return False
        # Raw prefix is immutable. Saved copies can be reused in this epoch.
        if chunk.saved:
            if not self.hot_ready or not self.source_ready.query():
                return False
            pages = tuple(self.history[chunk.offset:chunk.offset + chunk.count])
            self.ptr._release(pages)
            self.history[chunk.offset:chunk.offset + chunk.count] = [None] * chunk.count
            self.publish()
            return True
        pages = tuple(self.history[chunk.offset:chunk.offset + chunk.count])
        event, keepalive = self.transport.download(self, chunk)
        chunk.job = ChunkJob("down", event, pages, keepalive)
        self.publish()
        return True

    def start_up(self, chunk):
        if chunk.job is not None or not chunk.saved or any(
                p is not None for p in self.history[chunk.offset:chunk.offset + chunk.count]):
            return False
        pages = self.ptr._allocate(chunk.count)
        try:
            event = self.transport.upload(self, chunk, pages)
        except BaseException:
            self.transport.synchronize()
            self.ptr._release(pages)
            raise
        chunk.job = ChunkJob("up", event, pages)
        return True

    def poll(self):
        completed = []
        hot_completed = False
        if self.hot_event is not None and self.hot_event.query():
            self.hot_ready, self.hot_event = True, None
            hot_completed = True
        for chunk in self.chunks:
            job = chunk.job
            if job is None or not job.ready.query():
                continue
            if job.direction == "down":
                chunk.saved = True
                self.ptr._release(job.pages)
                self.history[chunk.offset:chunk.offset + chunk.count] = [None] * chunk.count
            else:
                self.history[chunk.offset:chunk.offset + chunk.count] = job.pages
            completed.append((job.direction, chunk))
            chunk.job = None
        if completed or hot_completed:
            self.publish()
        return completed
