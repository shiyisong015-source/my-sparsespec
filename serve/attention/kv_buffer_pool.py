"""Bounded tensor arenas for rank-local packed KV and transfer workspaces.

Following LMCache's allocation/completion separation, returning a lease does
not make its bytes reusable until all supplied CUDA events have completed.
These pools are owned by one scheduler thread; they do not launch transfers.
"""

import math

import torch


class BufferLease:
    """A tensor view with explicit ownership of an arena interval.

    Do not use the tensor after release(), even if a Python reference survives.
    No destructor implicitly returns memory while device work may be pending.
    """

    def __init__(self, pool, offset, allocated_bytes, tensor):
        self.tensor = tensor
        self.offset = offset
        self.allocated_bytes = allocated_bytes
        self._pool = pool
        self._events = ()
        self._state = "active"

    def release(self, *, after=()):
        """Return ownership, deferring reuse until the recorded events finish.

        after must cover every reader/writer on every stream. record_stream()
        alone cannot protect suballocations in a persistent arena.
        """
        if self._state != "active":
            raise RuntimeError("Buffer lease has already been returned")
        events = tuple(after)
        if any(not callable(getattr(event, "query", None))
               or not callable(getattr(event, "synchronize", None)) for event in events):
            raise ValueError("Lease completion dependencies must be recorded events")
        self._events = events
        self._state = "pending"
        self._pool.reclaim()


class KVBufferPool:
    """Preallocated, byte-budgeted CPU or GPU arena with reusable tensor views.

    CPU arenas are pinned by default. pin_memory=False is for CPU metadata
    tests; PackedKVRegion rejects such arenas for actual DMA. Exhaustion
    raises MemoryError without growing the arena or silently falling back.
    CUDA pools must be included in the caller's total KV memory budget.
    """

    def __init__(self, capacity_bytes, *, device="cpu", pin_memory=True, alignment=256):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if (type(alignment) is not int or alignment < 8
                or alignment & (alignment - 1)):
            raise ValueError("alignment must be a power of two of at least eight bytes")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("KV arenas support CPU and CUDA devices")
        self._buffer = torch.empty(capacity_bytes, dtype=torch.uint8, device=device,
                                   pin_memory=pin_memory if device.type == "cpu" else False)
        self.device = self._buffer.device
        self.is_pinned = self._buffer.is_pinned() if self.device.type == "cpu" else False
        self.capacity_bytes = capacity_bytes
        self.alignment = alignment
        self._free = [(0, capacity_bytes)]
        self._leases = {}
        self._used_bytes = self._peak_bytes = 0
        self._closed = False

    def allocate(self, shape, dtype, *, from_end=False):
        """Lease contiguous storage; temporary restores can use the opposite end.

        Persistent hot KV uses low addresses. Placing temporary restores at
        high addresses prevents replacement hot leases from splitting the
        space that a slightly larger restore will need in the next round.
        """
        if self._closed:
            raise RuntimeError("KV arena is closed")
        if type(from_end) is not bool:
            raise ValueError("from_end must be a bool")
        shape = tuple(shape)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Tensor dimensions must be nonnegative integers")
        item_bytes = torch.empty((), dtype=dtype, device="cpu").element_size()
        raw_bytes = math.prod(shape) * item_bytes
        allocated_bytes = (raw_bytes + self.alignment - 1) // self.alignment * self.alignment
        self.reclaim()
        if not raw_bytes:
            lease = BufferLease(self, 0, 0, self._buffer[:0].view(dtype).view(shape))
            self._leases[id(lease)] = lease
            return lease
        indices = range(len(self._free) - 1, -1, -1) if from_end else range(len(self._free))
        for index in indices:
            offset, free_bytes = self._free[index]
            if free_bytes < allocated_bytes:
                continue
            start = ((offset + free_bytes - allocated_bytes) // self.alignment * self.alignment
                     if from_end else offset)
            # Construct before changing accounting, so bad views cannot leak space.
            tensor = self._buffer[start:start + raw_bytes].view(dtype).view(shape)
            lease = BufferLease(self, start, allocated_bytes, tensor)
            remaining = []
            if start > offset:
                remaining.append((offset, start - offset))
            end = start + allocated_bytes
            if end < offset + free_bytes:
                remaining.append((end, offset + free_bytes - end))
            self._free[index:index + 1] = remaining
            self._leases[id(lease)] = lease
            self._used_bytes += allocated_bytes
            self._peak_bytes = max(self._peak_bytes, self._used_bytes)
            return lease
        raise MemoryError(
            f"KV arena exhausted on {self.device}: need {allocated_bytes} bytes, "
            f"free {self.capacity_bytes - self._used_bytes}, "
            f"largest interval {max((size for _, size in self._free), default=0)}, "
            f"capacity {self.capacity_bytes}"
        )

    def reclaim(self):
        """Poll completion events and coalesce reusable intervals; never wait."""
        completed = [lease for lease in self._leases.values()
                     if lease._state == "pending" and all(event.query() for event in lease._events)]
        for lease in completed:
            del self._leases[id(lease)]
            self._used_bytes -= lease.allocated_bytes
            if lease.allocated_bytes:
                self._free.append((lease.offset, lease.allocated_bytes))
            lease._state = "released"
            lease._events = ()
            lease.tensor = None
        if completed:
            merged = []
            for offset, size in sorted(self._free):
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    previous, previous_size = merged[-1]
                    merged[-1] = (previous, previous_size + size)
                else:
                    merged.append((offset, size))
            self._free = merged
        return len(completed)

    def stats(self):
        """Return reserved/in-use/free/pending bytes, including alignment."""
        self.reclaim()
        return dict(
            capacity_bytes=self.capacity_bytes,
            reserved_bytes=0 if self._closed else self.capacity_bytes,
            in_use_bytes=self._used_bytes,
            free_bytes=self.capacity_bytes - self._used_bytes, peak_bytes=self._peak_bytes,
            largest_free_bytes=max((size for _, size in self._free), default=0),
            pending_bytes=sum(lease.allocated_bytes for lease in self._leases.values()
                              if lease._state == "pending"),
            active_leases=sum(lease._state == "active" for lease in self._leases.values()),
        )

    def close(self, *, wait=False):
        """Close an idle arena; optionally wait for leases already returned.

        Active owners must release their leases first. wait=True synchronizes
        only the supplied completion events, without a device-wide barrier.
        """
        if self._closed:
            return
        if any(lease._state == "active" for lease in self._leases.values()):
            raise RuntimeError("Release active KV buffer leases before closing the arena")
        if wait:
            for lease in tuple(self._leases.values()):
                for event in lease._events:
                    event.synchronize()
        self.reclaim()
        if self._leases:
            raise RuntimeError("KV arena still has pending transfers or readers")
        self._buffer = None
        self._closed = True
