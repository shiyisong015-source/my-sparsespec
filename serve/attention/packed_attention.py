"""Attention over rank-local compact KV, with globally normalized selection.

GPU attention uses FlashInfer FA2 with explicit KV strides. The torch backend
is an independent mathematical reference for CPU tests, not a serving backend.
All LSE values here use base two, matching vendored FlashInfer state.cuh.
Keys must already have RoPE applied; history needs no original token IDs.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class AttentionState:
    output: torch.Tensor
    lse: torch.Tensor


def _check_segment(q, kv, causal):
    if q.ndim != 3 or kv.ndim != 5 or kv.shape[1:3] != (2, 1):
        raise ValueError("Expected q [Q, query_heads, D] and layer KV [N, 2, 1, heads, D]")
    if (min(q.shape) <= 0 or min(kv.shape[3:]) <= 0
            or q.shape[1] % kv.shape[3] or q.shape[-1] != kv.shape[-1]):
        raise ValueError("Invalid rank-local grouped-query geometry")
    if q.device != kv.device or q.dtype != kv.dtype:
        raise ValueError("Q and KV must have matching devices and dtypes")
    if q.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("Head channels must be contiguous")
    if causal and kv.shape[0] < q.shape[0]:
        raise ValueError("Causal hot KV must contain every query's ordered tail")


def _reference_logits(q, kv, causal):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    queries, heads, dim = q.shape
    kv_heads = kv.shape[3]
    grouped_q = q.to(dtype).reshape(queries, kv_heads, heads // kv_heads, dim)
    logits = torch.einsum("qhgd,nhd->qhgn", grouped_q, kv[:, 0, 0].to(dtype)) / math.sqrt(dim)
    visible = torch.ones((queries, kv.shape[0]), dtype=torch.bool, device=q.device)
    if causal:
        visible = (torch.arange(kv.shape[0], device=q.device)[None, :]
                   <= kv.shape[0] - queries + torch.arange(queries, device=q.device)[:, None])
    return logits, visible


def attention_segment(q, kv, *, causal, backend="flashinfer"):
    """Return attention and base-two LSE for one compact layer segment.

    causal=True aligns the ordered query tail with the end of hot KV. Cold
    history uses causal=False and has no copy of the speculative tail.
    """
    _check_segment(q, kv, causal)
    if backend not in ("flashinfer", "torch"):
        raise ValueError("Unknown packed attention backend")
    if backend == "flashinfer" and (q.device.type != "cuda"
                                     or q.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError("FlashInfer packed attention requires CUDA FP16/BF16")
    if kv.shape[0] == 0:
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        return AttentionState(torch.zeros_like(q), torch.full(q.shape[:2], -torch.inf,
                              dtype=dtype, device=q.device))
    if backend == "flashinfer":
        import flashinfer

        output, lse = flashinfer.single_prefill_with_kv_cache(
            q, kv[:, 0, 0], kv[:, 1, 0], causal=causal, kv_layout="NHD",
            pos_encoding_mode="NONE", backend="fa2", return_lse=True,
        )
        return AttentionState(output, lse)
    logits, visible = _reference_logits(q, kv, causal)
    masked = logits.masked_fill(~visible[:, None, None, :], -torch.inf)
    lse = torch.logsumexp(masked, dim=-1)
    probabilities = torch.exp(masked - lse[..., None])
    output = torch.einsum("qhgn,nhd->qhgd", probabilities, kv[:, 1, 0].to(logits.dtype))
    return AttentionState(output.reshape_as(q).to(q.dtype),
                          lse.reshape(q.shape[:2]) / math.log(2))


def merge_attention_states(first, second, *, backend="flashinfer"):
    """Merge disjoint KV segments using their softmax denominators."""
    if (first.output.shape != second.output.shape or first.lse.shape != second.lse.shape
            or first.output.shape[:2] != first.lse.shape
            or first.output.device != second.output.device
            or first.lse.device != first.output.device or second.lse.device != first.output.device
            or first.output.dtype != second.output.dtype):
        raise ValueError("Attention states must have matching query geometry/device/dtype")
    if backend == "flashinfer":
        import flashinfer

        output, lse = flashinfer.merge_state(first.output, first.lse, second.output, second.lse)
        return AttentionState(output, lse)
    if backend != "torch":
        raise ValueError("Unknown packed attention backend")
    maximum = torch.maximum(first.lse, second.lse)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    first_weight = torch.exp2(first.lse - safe_maximum)
    second_weight = torch.exp2(second.lse - safe_maximum)
    denominator = first_weight + second_weight
    output = (first.output.to(maximum.dtype) * first_weight[..., None]
              + second.output.to(maximum.dtype) * second_weight[..., None])
    output = output / denominator.clamp_min(1)[..., None]
    lse = safe_maximum + torch.log2(denominator)
    return AttentionState(output.to(first.output.dtype), lse)


@triton.jit
def _segment_selection_scores(
    q, keys, global_lse, scores,
    Q: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr, DIM: tl.constexpr,
    qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr,
    ks0: tl.constexpr, ks1: tl.constexpr, ks2: tl.constexpr,
    ls0: tl.constexpr, ls1: tl.constexpr, os0: tl.constexpr, os1: tl.constexpr,
    CAUSAL: tl.constexpr, SCALE_LOG2: tl.constexpr,
    ROWS: tl.constexpr, CHANNELS: tl.constexpr, TOKENS: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, ROWS)
    query = row // GROUP
    query_head = head * GROUP + row % GROUP
    channel = tl.arange(0, CHANNELS)
    token = tl.program_id(1).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    q_values = tl.load(q + query[:, None] * qs0 + query_head[:, None] * qs1
                       + channel[None, :] * qs2,
                       mask=(query[:, None] < Q) & (channel[None, :] < DIM), other=0)
    k_values = tl.load(keys + token[None, :] * ks0 + head * ks1 + channel[:, None] * ks2,
                       mask=(token[None, :] < N) & (channel[:, None] < DIM), other=0)
    logits = tl.dot(q_values, k_values) * SCALE_LOG2
    lse = tl.load(global_lse + query * ls0 + query_head * ls1, mask=query < Q, other=0)
    visible = (query[:, None] < Q) & (token[None, :] < N)
    if CAUSAL:
        visible = visible & (token[None, :] <= N - Q + query[:, None])
    probabilities = tl.where(visible, tl.exp2(logits - lse[:, None]), 0)
    visible_queries = tl.full((TOKENS,), Q, tl.int32)
    if CAUSAL:
        visible_queries = tl.minimum(Q, N - token)
    score = tl.sum(probabilities, axis=0) / (tl.maximum(visible_queries, 1) * GROUP)
    tl.store(scores + head * os0 + token * os1, score, mask=token < N)


def selection_scores(q, kv, global_lse, *, causal, backend="flashinfer", out=None):
    """Mean globally normalized probabilities over visible queries and GQA heads.

    This matches SparseSpec's score definition, including the smaller divisor
    for late causal-tail positions. Passing each segment's local LSE would
    incorrectly give hot and cold equal normalization mass.
    """
    _check_segment(q, kv, causal)
    if (global_lse.shape != q.shape[:2] or global_lse.device != q.device
            or global_lse.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Expected one global base-two LSE per query/head")
    shape = (kv.shape[3], kv.shape[0])
    if out is None:
        out = torch.empty(shape, dtype=torch.float16, device=q.device)
    if (out.shape != shape or out.device != q.device
            or not out.is_contiguous() or not out.is_floating_point()):
        raise ValueError("Invalid compact selection score destination")
    if backend == "torch":
        logits, visible = _reference_logits(q, kv, causal)
        lse = global_lse.reshape(q.shape[0], kv.shape[3], q.shape[1] // kv.shape[3])
        probabilities = torch.exp2(logits / math.log(2) - lse[..., None])
        probabilities = probabilities.masked_fill(~visible[:, None, None, :], 0)
        divisor = visible.sum(dim=0) * (q.shape[1] // kv.shape[3])
        out.copy_(probabilities.sum(dim=(0, 2)) / divisor.clamp_min(1))
        return out
    if (backend != "flashinfer" or q.device.type != "cuda"
            or q.dtype not in (torch.float16, torch.bfloat16)
            or global_lse.dtype != torch.float32
            or out.dtype not in (torch.float16, torch.float32)):
        raise ValueError("GPU score rematerialization requires CUDA FP16/BF16 and FP32 LSE")
    if kv.shape[0]:
        keys = kv[:, 0, 0]
        group = q.shape[1] // kv.shape[3]
        with torch.cuda.device(q.device):
            _segment_selection_scores[(kv.shape[3], triton.cdiv(kv.shape[0], 64))](
                q, keys, global_lse, out, q.shape[0], kv.shape[0], group, q.shape[-1],
                *q.stride(), *keys.stride(), *global_lse.stride(), *out.stride(),
                causal, 1 / math.sqrt(q.shape[-1]) / math.log(2),
                ROWS=max(16, triton.next_power_of_2(q.shape[0] * group)),
                CHANNELS=max(16, triton.next_power_of_2(q.shape[-1])), TOKENS=64, num_warps=4,
            )
    return out


class PackedAttention:
    """Layer API for compact draft and split verify; no request phase changes."""

    def __init__(self, *, backend="flashinfer"):
        if backend not in ("flashinfer", "torch"):
            raise ValueError("Unknown packed attention backend")
        self.backend = backend

    @staticmethod
    def _hot_layer(region, layer, tail_length):
        if type(layer) is not int or not 0 <= layer < region.layers:
            raise ValueError("Invalid layer")
        if type(tail_length) is not int or not 0 <= tail_length <= region.tail_capacity:
            raise ValueError("Invalid effective speculative tail length")
        return region.hot_kv[layer, :region.hot_count + tail_length]

    def draft_layer(self, q, region, layer, *, tail_length=None):
        tail_length = region.tail_length if tail_length is None else tail_length
        hot = self._hot_layer(region, layer, tail_length)
        return attention_segment(q, hot, causal=True, backend=self.backend)

    def verify_layer(self, q, region, restored, layer, *, tail_length=None,
                     hot_scores=None, cold_scores=None):
        """Verify with the entire cold history restored before this forward.

        The ordered tail is rewritten only in hot. Both score arrays are
        recomputed using the merged LSE, for next-round KV selection.
        """
        tail_length = q.shape[0] if tail_length is None else tail_length
        if tail_length != q.shape[0]:
            raise ValueError("Verify must rewrite and attend to its entire query tail")
        hot = self._hot_layer(region, layer, tail_length)
        if (restored.offset != 0 or restored.count != region.cold_count or restored.kv is None
                or restored.kv.ndim != 6 or restored.kv.shape[0] != region.layers
                or restored.kv.shape[1] < restored.count
                or restored.kv.shape[2:] != region.hot_kv.shape[2:]):
            raise ValueError("Verify requires a complete rank-local cold history")
        if q.device.type == "cuda":
            torch.cuda.current_stream(q.device).wait_event(restored.ready)
        cold = restored.kv[layer, :restored.count]
        hot_state = attention_segment(q, hot, causal=True, backend=self.backend)
        # Empty cold has zero mass; skip the FlashInfer merge kernel in that case.
        if region.cold_count:
            cold_state = attention_segment(q, cold, causal=False, backend=self.backend)
            state = merge_attention_states(hot_state, cold_state, backend=self.backend)
        else:
            state = hot_state
        hot_scores = selection_scores(q, hot, state.lse, causal=True,
                                      backend=self.backend, out=hot_scores)
        cold_scores = selection_scores(q, cold, state.lse, causal=False,
                                       backend=self.backend, out=cold_scores)
        return state, hot_scores, cold_scores
