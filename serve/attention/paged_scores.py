"""Graph-safe selection scores over all verification queries in paged KV.

The attention forward and its base-two LSE remain native FlashInfer. Only the
score rematerialization uses this path when GQA * (k + 1) exceeds 64 rows.
"""
import math

import torch
import triton
import triton.language as tl


MAX_PACKED_SCORE_QO = 256
TOKEN_TILE = 64


@triton.jit
def _plan_tiles(all_indptr, request_types, tile_indptr, batch_size,
                BLOCK: tl.constexpr):
    request = tl.arange(0, BLOCK)
    kind = tl.load(request_types + request, request < batch_size, other=0)
    start = tl.load(all_indptr + request, request < batch_size, other=0)
    end = tl.load(all_indptr + request + 1, request < batch_size, other=0)
    tiles = tl.where((request < batch_size) & (kind == 2), tl.cdiv(end - start, 64), 0)
    prefix = tl.cumsum(tiles)
    tl.store(tile_indptr, 0)
    tl.store(tile_indptr + request + 1, prefix, request < batch_size)


@triton.jit
def _paged_scores(q, kv, lse, indices, all_indptr, qo_indptr, score_indptr,
                  tile_indptr, batch_size_ptr, output,
                  QS0: tl.constexpr, QS1: tl.constexpr,
                  KS0: tl.constexpr, KS_HEAD: tl.constexpr,
                  LS0: tl.constexpr, LS1: tl.constexpr,
                  OS0: tl.constexpr, HEADS: tl.constexpr, GROUP: tl.constexpr,
                  DIM: tl.constexpr, ROWS: tl.constexpr, PROGRAMS: tl.constexpr,
                  SCALE: tl.constexpr):
    task = tl.program_id(0)
    head = tl.program_id(1)
    batch_size = tl.load(batch_size_ptr)
    total = tl.load(tile_indptr + batch_size)
    row = tl.arange(0, ROWS)
    qi = row // GROUP
    qh = head * GROUP + row % GROUP
    channel = tl.arange(0, DIM)
    offset = tl.arange(0, 64)
    # Persistent tasks bound the launch size even when captured with many
    # NORMAL/DRAFT requests. The GPU prefix changes between graph replays.
    while task < total:
        lo, hi = 0, batch_size
        while lo < hi:
            mid = (lo + hi) // 2
            end_tile = tl.load(tile_indptr + mid + 1)
            before = end_tile <= task
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        request = lo
        first_tile = tl.load(tile_indptr + request)
        token = (task - first_tile) * 64 + offset
        q0 = tl.load(qo_indptr + request)
        qn = tl.load(qo_indptr + request + 1) - q0
        kv0 = tl.load(all_indptr + request)
        length = tl.load(all_indptr + request + 1) - kv0
        destination = tl.load(score_indptr + request)
        queries = tl.load(q + (q0 + qi[:, None]) * QS0 + qh[:, None] * QS1
                          + channel[None, :], qi[:, None] < qn, other=0)
        pages = tl.load(indices + kv0 + token, token < length, other=0).to(tl.int64)
        keys = tl.load(kv + pages[None, :] * KS0 + head * KS_HEAD + channel[:, None],
                       token[None, :] < length, other=0)
        logits = tl.dot(queries, keys) * SCALE
        normalizer = tl.load(lse + (q0 + qi) * LS0 + qh * LS1, qi < qn, other=0)
        visible = ((qi[:, None] < qn) & (token[None, :] < length)
                   & (token[None, :] <= length - qn + qi[:, None]))
        probability = tl.where(visible, tl.exp2(logits - normalizer[:, None]), 0.)
        denominator = tl.maximum(tl.minimum(qn, length - token), 1) * GROUP
        score = tl.sum(probability, axis=0) / denominator
        tl.store(output + head * OS0 + destination + token, score, token < length)
        task += PROGRAMS


def prepare_paged_scores(metadata):
    """Plan once per forward outside graph replay, using existing GPU metadata."""
    batch = metadata["batch_size_host"]
    _plan_tiles[(1,)](metadata["all_kv_indptr"], metadata["request_type"],
                       metadata["score_tile_indptr"], batch,
                       BLOCK=triton.next_power_of_2(max(1, batch)))


def paged_selection_scores(q, kv, lse, metadata, layer, max_queries):
    """Write the native [head, verify-KV] score buffer, with no extra KV copy."""
    heads, group = kv.shape[3], q.shape[1] // kv.shape[3]
    rows = triton.next_power_of_2(max(16, max_queries * group))
    if (kv.ndim != 5 or kv.shape[1:3] != (2, 1) or q.shape[1] % heads
            or rows > MAX_PACKED_SCORE_QO or q.shape[-1] != kv.shape[-1]
            or q.shape[-1] not in (64, 128, 256)
            or q.dtype not in (torch.float16, torch.bfloat16)
            or kv.dtype != q.dtype or lse.dtype != torch.float32
            or q.stride(-1) != 1 or kv.stride(-1) != 1):
        raise ValueError("Unsupported paged score geometry/dtype")
    output = metadata["dump_logits"][layer]
    programs = 108  # Bounded persistent grid; independent of graph padding.
    _paged_scores[(programs, heads)](
        q, kv, lse, metadata["kv_indices"], metadata["all_kv_indptr"],
        metadata["qo_indptr"], metadata["verify_kv_indptr"],
        metadata["score_tile_indptr"], metadata["batch_size_tensor"], output,
        *q.stride()[:2], kv.stride(0), kv.stride(3), *lse.stride(), output.stride(0),
        heads, group, q.shape[-1], rows, programs,
        1 / math.sqrt(q.shape[-1]) / math.log(2), num_warps=8 if rows >= 128 else 4,
    )
