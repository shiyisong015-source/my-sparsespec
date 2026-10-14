"""Compare split KV with independent full-context attention; GPU is opt-in."""

import os
import unittest

import torch

from serve.attention.kv_partition import PackedKVRegion, RestoredChunk, partition_pages
from serve.attention.packed_attention import (
    AttentionState, PackedAttention, attention_segment, merge_attention_states,
)
from serve.tests.test_kv_buffer_pool import CompletionEvent


def reference_gather(source, pages):
    result = torch.empty((source.shape[0], pages.shape[-1], *source.shape[2:]),
                         dtype=source.dtype, device=source.device)
    for layer in range(source.shape[0]):
        for head in range(source.shape[4]):
            result[layer, :, :, 0, head] = source[layer, :, :, 0, head].index_select(
                0, pages[layer, head].long().to(source.device),
            )
    return result


def full_reference(q, kv, history_tokens):
    """Independent dense attention in original-token order, using natural logs."""
    group = q.shape[1] // kv.shape[3]
    keys = kv[:, 0, 0].double().repeat_interleave(group, dim=1)
    values = kv[:, 1, 0].double().repeat_interleave(group, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.double(), keys) / (q.shape[-1] ** 0.5)
    visible = (torch.arange(kv.shape[0], device=q.device)[None, :]
               <= history_tokens + torch.arange(q.shape[0], device=q.device)[:, None])
    logits = logits.masked_fill(~visible[None, :, :], -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, values)
    lse = torch.logsumexp(logits, dim=-1).T / torch.log(torch.tensor(2., dtype=torch.float64))
    scores = probabilities.reshape(kv.shape[3], group, q.shape[0], kv.shape[0]).sum(dim=(1, 2))
    scores /= visible.sum(dim=0).clamp_min(1) * group
    return output, lse, scores


def make_region(queries=3, *, heads=2, layers=2, dim=8, all_hot=False,
                device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(330)
    history = torch.randn((layers, 11, 2, 1, heads, dim), generator=generator,
                           dtype=torch.float64).to(device=device, dtype=dtype)
    tail = torch.randn((layers, queries, 2, 1, heads, dim), generator=generator,
                       dtype=torch.float64).to(device=device, dtype=dtype)
    q = torch.randn((queries, heads * 4, dim), generator=generator,
                    dtype=torch.float64).to(device=device, dtype=dtype)
    selection = torch.stack([torch.stack([torch.randperm(9, generator=generator)[:2]
                                         for _ in range(heads)]) for _ in range(layers)])
    plan = partition_pages(torch.arange(11), selection, 0 if all_hot else 2,
                           pool_capacity=11, recent_count=11 if all_hot else 2, validate=True)
    region = PackedKVRegion()
    region.phase = "packed"
    region.layers, region.heads, region.dim = layers, heads, dim
    region.device, region.dtype = history.device, dtype
    region.hot_count, region.cold_count = plan.hot_count, plan.cold_count
    region.history_length = 11
    region.recent_count, region.selected_count = plan.recent_count, plan.selected_count
    region.tail_capacity, region.tail_length = queries + 2, queries
    region._hot = torch.full((layers, plan.hot_count + queries + 2, 2, 1, heads, dim),
                             10000., device=device, dtype=dtype)
    region._hot[:, :plan.hot_count] = reference_gather(history, plan.hot_pages)
    region._hot[:, plan.hot_count:plan.hot_count + queries] = tail
    cold = reference_gather(history, plan.cold_pages)
    ready = (torch.cuda.current_stream().record_event() if device == "cuda" else CompletionEvent(True))
    restored = RestoredChunk(cold, 0, plan.cold_count, ready)
    return q, history, tail, region, restored, plan


class PackedAttentionTest(unittest.TestCase):
    def test_split_verify_matches_full_context_per_head(self):
        for queries in (1, 3, 5):
            q, history, tail, region, restored, plan = make_region(queries)
            for layer in range(region.layers):
                hot_scores = torch.empty((region.heads, region.hot_count + queries), dtype=torch.float64)
                cold_scores = torch.empty((region.heads, region.cold_count), dtype=torch.float64)
                state, hot_scores, cold_scores = PackedAttention(backend="torch").verify_layer(
                    q, region, restored, layer, hot_scores=hot_scores, cold_scores=cold_scores,
                )
                expected, lse, scores = full_reference(q, torch.cat((history[layer], tail[layer])), 11)
                torch.testing.assert_close(state.output, expected, rtol=1e-11, atol=1e-11)
                torch.testing.assert_close(state.lse, lse, rtol=1e-11, atol=1e-11)
                for head in range(region.heads):
                    hot_ids = torch.cat((plan.hot_pages[layer, head], torch.arange(11, 11 + queries)))
                    torch.testing.assert_close(hot_scores[head], scores[head, hot_ids], rtol=1e-11, atol=1e-11)
                    torch.testing.assert_close(cold_scores[head], scores[head, plan.cold_pages[layer, head]],
                                               rtol=1e-11, atol=1e-11)

    def test_draft_uses_only_selected_recent_and_tail(self):
        q, _, _, region, _, _ = make_region(1)
        hot = region.hot_kv[0, :region.hot_length]
        state = PackedAttention(backend="torch").draft_layer(q, region, 0)
        expected, lse, _ = full_reference(q, hot, region.hot_count)
        torch.testing.assert_close(state.output, expected, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(state.lse, lse, rtol=1e-11, atol=1e-11)

    def test_tail_is_causal_and_appears_once(self):
        q, history, tail, region, restored, _ = make_region(3)
        before, _, _ = PackedAttention(backend="torch").verify_layer(q, region, restored, 0)
        region._hot[0, region.hot_count + 2].fill_(10000)
        after, _, _ = PackedAttention(backend="torch").verify_layer(q, region, restored, 0)
        torch.testing.assert_close(before.output[:2], after.output[:2], rtol=0, atol=0)
        expected, _, _ = full_reference(q, torch.cat((history[0], tail[0])), 11)
        torch.testing.assert_close(before.output, expected, rtol=1e-11, atol=1e-11)

    def test_global_score_mass_includes_both_partitions(self):
        q, _, _, region, restored, _ = make_region(1)
        _, hot, cold = PackedAttention(backend="torch").verify_layer(
            q, region, restored, 0,
            hot_scores=torch.empty((region.heads, region.hot_length), dtype=torch.float64),
            cold_scores=torch.empty((region.heads, region.cold_count), dtype=torch.float64),
        )
        torch.testing.assert_close(hot.sum(-1) + cold.sum(-1), torch.ones(region.heads).double())
        self.assertTrue(torch.all(hot.sum(-1) < .99))

    def test_empty_cold_and_empty_state_merge(self):
        q, history, tail, region, restored, _ = make_region(3, all_hot=True)
        state, _, cold = PackedAttention(backend="torch").verify_layer(q, region, restored, 0)
        expected, lse, _ = full_reference(q, torch.cat((history[0], tail[0])), 11)
        torch.testing.assert_close(state.output, expected, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(state.lse, lse, rtol=1e-11, atol=1e-11)
        self.assertEqual(cold.shape[-1], 0)
        empty = attention_segment(q, history[0, :0], causal=False, backend="torch")
        merged = merge_attention_states(empty, empty, backend="torch")
        self.assertTrue(torch.all(merged.output == 0))
        self.assertTrue(torch.all(torch.isneginf(merged.lse)))

    def test_tp_local_head_slices_match_global(self):
        q, _, _, region, restored, _ = make_region(3, heads=4)
        global_state, _, _ = PackedAttention(backend="torch").verify_layer(q, region, restored, 0)
        outputs = []
        for start in (0, 2):
            local = PackedKVRegion()
            local.__dict__.update(region.__dict__)
            local.heads = 2
            local._hot = region.hot_kv[..., start:start + 2, :]
            cold = RestoredChunk(restored.kv[..., start:start + 2, :], 0, restored.count, restored.ready)
            state, _, _ = PackedAttention(backend="torch").verify_layer(q[:, start * 4:(start + 2) * 4],
                                                                       local, cold, 0)
            outputs.append(state.output)
        torch.testing.assert_close(torch.cat(outputs, dim=1), global_state.output, rtol=1e-11, atol=1e-11)

    def test_partial_restore_cannot_verify(self):
        q, _, _, region, restored, _ = make_region(3)
        partial = RestoredChunk(restored.kv[:, :1], 0, 1, restored.ready)
        with self.assertRaises(ValueError):
            PackedAttention(backend="torch").verify_layer(q, region, partial, 0)


@unittest.skipUnless(os.environ.get("RUN_PACKED_KV_GPU") == "1" and torch.cuda.is_available(),
                     "Packed CUDA attention awaits an available GPU")
class PackedAttentionCudaTest(unittest.TestCase):
    def test_strided_kv_and_global_selection_scores(self):
        for dtype in (torch.float16, torch.bfloat16):
            q, history, tail, region, restored, plan = make_region(15, heads=4, dim=128,
                                                                  device="cuda", dtype=dtype)
            state, hot, cold = PackedAttention().verify_layer(
                q, region, restored, 0,
                hot_scores=torch.empty((4, region.hot_length), device="cuda", dtype=torch.float32),
                cold_scores=torch.empty((4, region.cold_count), device="cuda", dtype=torch.float32),
            )
            expected, lse, scores = full_reference(q, torch.cat((history[0], tail[0])), 11)
            torch.testing.assert_close(state.output.double(), expected, rtol=1e-2, atol=3e-3)
            torch.testing.assert_close(state.lse.double(), lse, rtol=3e-3, atol=3e-3)
            for head in range(4):
                hot_ids = torch.cat((plan.hot_pages[0, head], torch.arange(11, 26))).cuda()
                torch.testing.assert_close(hot[head].double(), scores[head, hot_ids], rtol=2e-2, atol=2e-4)
                torch.testing.assert_close(cold[head].double(), scores[head, plan.cold_pages[0, head].cuda()],
                                           rtol=2e-2, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
