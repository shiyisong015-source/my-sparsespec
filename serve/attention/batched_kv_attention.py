"""Batched attention over independently allocated hot/cold KV regions.

Every task reads one request/head/query tile and one KV chunk. Both segments
share a grid. KV pointers and lengths are runtime metadata; no physical KV
concatenation or length-specific compilation is needed. Chunk results carry
base-two LSE and are merged before the single batched score-rematerialization
launch. Only GPU-resident KV is read here; cold restoration precedes verify.
"""

import ctypes
from functools import lru_cache
import math

import torch
import triton
import triton.language as tl


# Per-request int64 metadata. Pointers are CUDA virtual addresses, strides are
# in tensor elements. Hot and cold retain [L, N, 2, 1, Hkv, D] layouts.
Q0, QN, HP, HL, HN, CP, CL, CN, WRITE, HS, HSL, CS, CSL, P0, NS = (
    tl.constexpr(i) for i in range(15))
FIELDS = tl.constexpr(15)


@triton.jit(do_not_specialize=["layer", "total", "ks0", "vs0"])
def _append_batch(keys, values, metadata, owners, layer, total, ks0, vs0,
                  HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token = x // (HEADS * DIM)
    hd = x % (HEADS * DIM)
    valid = token < total
    request = tl.load(owners + token, mask=valid, other=0).to(tl.int64)
    m = metadata + request * FIELDS
    q0 = tl.load(m + Q0)
    pointer = tl.load(m + HP).to(tl.pointer_type(keys.dtype.element_ty))
    stride_layer = tl.load(m + HL)
    write = tl.load(m + WRITE)
    position = write + token - q0
    address = pointer + layer * stride_layer + position * (2 * HEADS * DIM) + hd
    k = tl.load(keys + token * ks0 + hd, mask=valid, other=0)
    v = tl.load(values + token * vs0 + hd, mask=valid, other=0)
    tl.store(address, k, mask=valid)
    tl.store(address + HEADS * DIM, v, mask=valid)


@triton.jit(do_not_specialize=["layer", "qs0"])
def _split_attention(q, metadata, jobs, partial, partial_lse, layer, qs0,
                     HEADS: tl.constexpr, GROUP: tl.constexpr, DIM: tl.constexpr,
                     ROWS: tl.constexpr, TOKENS: tl.constexpr):
    job = jobs + tl.program_id(0).to(tl.int64) * 6
    request = tl.load(job)
    query_tile = tl.load(job + 1)
    segment = tl.load(job + 2)
    begin = tl.load(job + 3)
    end = tl.load(job + 4)
    split = tl.load(job + 5)
    head = tl.program_id(1).to(tl.int64)
    m = metadata + request * FIELDS
    q0, qn = tl.load(m + Q0), tl.load(m + QN)
    pointer = tl.load(m + tl.where(segment == 0, HP, CP)).to(tl.pointer_type(q.dtype.element_ty))
    stride_layer = tl.load(m + tl.where(segment == 0, HL, CL))
    length = tl.load(m + tl.where(segment == 0, HN, CN))
    row = query_tile * ROWS + tl.arange(0, ROWS)
    qi, qh = row // GROUP, head * GROUP + row % GROUP
    channel = tl.arange(0, DIM)
    queries = tl.load(q + (q0 + qi[:, None]) * qs0 + qh[:, None] * DIM + channel[None, :],
                      mask=qi[:, None] < qn, other=0)
    maximum = tl.full((ROWS,), -float("inf"), tl.float32)
    denominator = tl.full((ROWS,), 0, tl.float32)
    accumulator = tl.full((ROWS, DIM), 0, tl.float32)
    base = pointer + layer * stride_layer
    for start in range(begin, end, TOKENS):
        token = start + tl.arange(0, TOKENS)
        k = tl.load(base + token[None, :] * (2 * HEADS * DIM) + head * DIM + channel[:, None],
                    mask=token[None, :] < end, other=0)
        v = tl.load(base + token[:, None] * (2 * HEADS * DIM) + HEADS * DIM
                    + head * DIM + channel[None, :], mask=token[:, None] < end, other=0)
        logits = tl.dot(queries, k) * (math.log2(math.e) / math.sqrt(DIM))
        visible = (qi[:, None] < qn) & (token[None, :] < end)
        visible = visible & ((segment != 0) | (token[None, :] <= length - qn + qi[:, None]))
        logits = tl.where(visible, logits, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        alpha = tl.where(maximum == -float("inf"), 0, tl.exp2(maximum - new_maximum))
        p = tl.where(visible, tl.exp2(logits - new_maximum[:, None]), 0)
        denominator = denominator * alpha + tl.sum(p, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(p.to(q.dtype.element_ty), v)
        maximum = new_maximum
    safe_denominator = tl.where(denominator > 0, denominator, 1)
    result = accumulator / safe_denominator[:, None]
    lse = tl.where(denominator > 0, maximum + tl.log2(safe_denominator), -float("inf"))
    output_row = tl.load(m + P0) + split * qn + qi
    tl.store(partial + output_row[:, None] * (HEADS * GROUP * DIM)
             + qh[:, None] * DIM + channel[None, :], result, mask=qi[:, None] < qn)
    tl.store(partial_lse + output_row * (HEADS * GROUP) + qh, lse, mask=qi < qn)


@triton.jit
def _merge_batch(partial, partial_lse, metadata, owners, output, lse,
                 HEADS: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr):
    token, head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    request = tl.load(owners + token).to(tl.int64)
    m = metadata + request * FIELDS
    qi = token - tl.load(m + Q0)
    qn, count, p0 = tl.load(m + QN), tl.load(m + NS), tl.load(m + P0)
    split, channel = tl.arange(0, SPLITS), tl.arange(0, DIM)
    row = p0 + split * qn + qi
    state_lse = tl.load(partial_lse + row * HEADS + head, mask=split < count, other=-float("inf"))
    maximum = tl.max(state_lse, axis=0)
    weights = tl.exp2(state_lse - maximum)
    denominator = tl.sum(weights, axis=0)
    values = tl.load(partial + row[:, None] * (HEADS * DIM) + head * DIM + channel[None, :],
                     mask=split[:, None] < count, other=0)
    result = tl.sum(values * weights[:, None], axis=0) / denominator
    tl.store(output + token * (HEADS * DIM) + head * DIM + channel, result)
    tl.store(lse + token * HEADS + head, maximum + tl.log2(denominator))


@triton.jit(do_not_specialize=["layer", "qs0"])
def _batch_scores(q, global_lse, metadata, jobs, layer, qs0,
                  HEADS: tl.constexpr, GROUP: tl.constexpr, DIM: tl.constexpr,
                  ROWS: tl.constexpr, TOKENS: tl.constexpr):
    job = jobs + tl.program_id(0).to(tl.int64) * 3
    request, segment, begin = tl.load(job), tl.load(job + 1), tl.load(job + 2)
    head = tl.program_id(1).to(tl.int64)
    m = metadata + request * FIELDS
    q0, qn = tl.load(m + Q0), tl.load(m + QN)
    length = tl.load(m + tl.where(segment == 0, HN, CN))
    pointer = tl.load(m + tl.where(segment == 0, HP, CP)).to(tl.pointer_type(q.dtype.element_ty))
    stride_layer = tl.load(m + tl.where(segment == 0, HL, CL))
    score_pointer = tl.load(m + tl.where(segment == 0, HS, CS)).to(tl.pointer_type(tl.float16))
    score_layer_stride = tl.load(m + tl.where(segment == 0, HSL, CSL))
    row, channel, token = tl.arange(0, ROWS), tl.arange(0, DIM), begin + tl.arange(0, TOKENS)
    qi, qh = row // GROUP, head * GROUP + row % GROUP
    queries = tl.load(q + (q0 + qi[:, None]) * qs0 + qh[:, None] * DIM + channel[None, :],
                      mask=qi[:, None] < qn, other=0)
    keys = tl.load(pointer + layer * stride_layer + token[None, :] * (2 * HEADS * DIM)
                   + head * DIM + channel[:, None], mask=token[None, :] < length, other=0)
    logits = tl.dot(queries, keys) * (math.log2(math.e) / math.sqrt(DIM))
    normalizer = tl.load(global_lse + (q0 + qi) * (HEADS * GROUP) + qh, mask=qi < qn, other=0)
    visible = (qi[:, None] < qn) & (token[None, :] < length)
    visible = visible & ((segment != 0) | (token[None, :] <= length - qn + qi[:, None]))
    probabilities = tl.where(visible, tl.exp2(logits - normalizer[:, None]), 0)
    visible_queries = tl.where(segment == 0, tl.minimum(qn, length - token), qn)
    scores = tl.sum(probabilities, axis=0) / (tl.maximum(visible_queries, 1) * GROUP)
    tl.store(score_pointer + layer * score_layer_stride + head * length + token,
             scores, mask=token < length)


@lru_cache(maxsize=None)
def _hardware(device_index):
    p = torch.cuda.get_device_properties(device_index)
    return p.multi_processor_count


@lru_cache(maxsize=None)
def _occupancy(function, threads, shared_bytes):
    library = ctypes.CDLL("libcuda.so.1")
    api = library.cuOccupancyMaxActiveBlocksPerMultiprocessor
    api.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    api.restype = ctypes.c_int
    blocks = ctypes.c_int()
    status = api(ctypes.byref(blocks), ctypes.c_void_p(function), threads, shared_bytes)
    if status:
        raise RuntimeError(f"CUDA occupancy query failed: {status}")
    return blocks.value


def kernel_occupancy(kernel, warps=4):
    """Cache maximum resident TBs for each compiled CUfunction/configuration."""
    return _occupancy(kernel.function, 32 * warps, kernel.metadata.shared)


class BatchedKVAttention:
    """One reusable metadata/workspace plan for all layers of a model forward."""

    def __init__(self, frames, offsets, *, query_heads, query_tile=32, chunk_tokens=None):
        first = frames[0].region
        self.device, self.dtype = first.device, first.dtype
        self.heads, self.dim = first.heads, first.dim
        self.group = query_heads // self.heads
        if query_heads % self.heads or self.dim not in (32, 64, 128, 256):
            raise ValueError("Unsupported grouped-query geometry")
        self.query_tile = query_tile
        self.total = offsets[-1]
        rows, owners, descriptions = [], [], []
        weighted_tokens = 0
        for frame, q0 in zip(frames, offsets[:-1]):
            region, qn = frame.region, frame.query_tokens
            packed = hasattr(region, "hot_count")
            if packed:
                hot = region.hot_kv
                hn, write = region.hot_count + frame.tail_length, region.hot_count + frame.write_offset
                cold = frame.restored.kv if frame.mode == "verify" else None
                cn = region.cold_count if cold is not None else 0
                hot_scores, cold_scores = frame.hot_scores, frame.cold_scores
            else:
                hot, cold = region.kv, None
                hn, write, cn = frame.write_offset + qn, frame.write_offset, 0
                hot_scores, cold_scores = frame.scores, None
            if not hot.is_contiguous() or (cold is not None and not cold.is_contiguous()):
                raise ValueError("Batched KV leases must be contiguous")
            mt = triton.cdiv(qn * self.group, query_tile)
            weighted_tokens += mt * (hn + cn)
            descriptions.append((q0, qn, hot, hn, cold, cn, write, hot_scores, cold_scores, mt))
        target_jobs = max(1, triton.cdiv(2 * _hardware(self.device.index), self.heads))
        if chunk_tokens is None:
            chunk_tokens = min(2048, max(128, 128 * triton.cdiv(weighted_tokens, target_jobs * 128)))
        if type(chunk_tokens) is not int or chunk_tokens < 64 or chunk_tokens % 64:
            raise ValueError("KV chunk size must be a positive multiple of 64")
        self.chunk_tokens = chunk_tokens
        tasks, score_tasks, partial_rows, maximum_splits, score_rows = [], [], 0, 1, 16
        self._borrowed = []
        for request, desc in enumerate(descriptions):
            q0, qn, hot, hn, cold, cn, write, hs, cs, mt = desc
            nsplit = triton.cdiv(hn, chunk_tokens) + triton.cdiv(cn, chunk_tokens)
            pointer = lambda t: 0 if t is None else t.data_ptr()
            stride = lambda t: 0 if t is None else t.stride(0)
            rows.append((q0, qn, pointer(hot), stride(hot), hn, pointer(cold), stride(cold), cn,
                         write, pointer(hs), stride(hs), pointer(cs), stride(cs), partial_rows, nsplit))
            owners.extend([request] * qn)
            split = 0
            for segment, length in ((0, hn), (1, cn)):
                for begin in range(0, length, chunk_tokens):
                    for tile in range(mt):
                        tasks.append((request, tile, segment, begin, min(length, begin + chunk_tokens), split))
                    split += 1
                if hs is not None:
                    for begin in range(0, length, 64):
                        score_tasks.append((request, segment, begin))
            if hs is not None:
                score_rows = max(score_rows, triton.next_power_of_2(qn * self.group))
            partial_rows += nsplit * qn
            maximum_splits = max(maximum_splits, nsplit)
            self._borrowed.extend(t for t in (hot, cold, hs, cs) if t is not None)
        # One pinned upload per forward. Keep the host allocation alive until
        # the plan retires; all GPU views share the same allocation/lifetime.
        flat_rows = [x for row in rows for x in row]
        flat_jobs = [x for row in tasks for x in row]
        flat_scores = [x for row in score_tasks for x in row]
        self._host_metadata = torch.tensor(flat_rows + owners + flat_jobs + flat_scores,
                                          dtype=torch.int64, device="cpu", pin_memory=True)
        self._metadata_buffer = self._host_metadata.to(self.device, non_blocking=True)
        a, b, c = len(flat_rows), len(flat_rows) + len(owners), len(flat_rows) + len(owners) + len(flat_jobs)
        self.metadata = self._metadata_buffer[:a].view(-1, 15)
        self.owners = self._metadata_buffer[a:b]
        self.jobs = self._metadata_buffer[b:c].view(-1, 6)
        self.score_jobs = self._metadata_buffer[c:].view(-1, 3)
        self.partial = torch.empty((partial_rows, query_heads, self.dim), dtype=torch.float32, device=self.device)
        self.partial_lse = torch.empty((partial_rows, query_heads), dtype=torch.float32, device=self.device)
        self.output = torch.empty((self.total, query_heads, self.dim), dtype=self.dtype, device=self.device)
        self.lse = torch.empty((self.total, query_heads), dtype=torch.float32, device=self.device)
        self.max_splits, self.score_rows = triton.next_power_of_2(maximum_splits), score_rows
        self.ready = torch.cuda.current_stream(self.device).record_event()
        self._compute_stream = None
        self.stats = dict(chunk_tokens=chunk_tokens, grid_blocks=len(tasks) * self.heads,
                          score_blocks=len(score_tasks) * self.heads, query_tile=query_tile,
                          SMs=_hardware(self.device.index), maximum_splits=maximum_splits,
                          workspace_bytes=sum(t.nbytes for t in (self.partial, self.partial_lse,
                                                               self.output, self.lse)))

    def run(self, q, keys, values, layer):
        current = torch.cuda.current_stream(self.device)
        if self._compute_stream is None:
            current.wait_event(self.ready)
            for t in (self._metadata_buffer, self.partial, self.partial_lse,
                      self.output, self.lse, *self._borrowed):
                t.record_stream(current)
            self._compute_stream = current
        elif current != self._compute_stream:
            raise RuntimeError("All layers of a batched attention plan must share one compute stream")
        _append_batch[(triton.cdiv(self.total * self.heads * self.dim, 256),)](
            keys, values, self.metadata, self.owners, layer, self.total, keys.stride(0), values.stride(0),
            HEADS=self.heads, DIM=self.dim, BLOCK=256, num_warps=4)
        compiled = _split_attention[(self.jobs.shape[0], self.heads)](
            q, self.metadata, self.jobs, self.partial, self.partial_lse, layer, q.stride(0),
            HEADS=self.heads, GROUP=self.group, DIM=self.dim, ROWS=self.query_tile, TOKENS=64, num_warps=4)
        _merge_batch[(self.total, self.heads * self.group)](
            self.partial, self.partial_lse, self.metadata, self.owners, self.output, self.lse,
            HEADS=self.heads * self.group, DIM=self.dim, SPLITS=self.max_splits, num_warps=4)
        if self.score_jobs.shape[0]:
            _batch_scores[(self.score_jobs.shape[0], self.heads)](
                q, self.lse, self.metadata, self.score_jobs, layer, q.stride(0),
                HEADS=self.heads, GROUP=self.group, DIM=self.dim, ROWS=self.score_rows, TOKENS=64, num_warps=4)
        if "TBs_per_SM" not in self.stats:
            b = kernel_occupancy(compiled)
            g, n = self.stats["grid_blocks"], self.stats["SMs"]
            waves = math.ceil(g / (b * n))
            tail = math.ceil(g / b - n * (waves - 1))
            self.stats.update(TBs_per_SM=b, estimated_waves=waves,
                              model_idle_ratio=(n - tail) / (n * waves),
                              registers=compiled.n_regs, shared_bytes=compiled.metadata.shared)
        return self.output
