"""Independent dense references for ragged, long and all-hot batch segments."""

import os
import unittest

import torch

from serve.attention.full_forward import FullForward, FullKVRegion
from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.kv_partition import PackedKVRegion, RestoredChunk, partition_pages
from serve.attention.packed_batch import PackedBatch
from serve.attention.packed_forward import PackedForward
from serve.tests.test_packed_attention import full_reference, reference_gather


@unittest.skipUnless(os.environ.get("RUN_PACKED_KV_GPU") == "1" and torch.cuda.is_available(),
                     "Requires an available CUDA GPU")
class BatchedAttentionCudaTest(unittest.TestCase):
    def test_cuda_default_device_and_stream_change_guard(self):
        with torch.device("cuda"):
            frame, q, tail, _, _ = self.packed(133, 1, 4, torch.bfloat16, draft=True)
            batch = PackedBatch([frame])
            batch.append_batch_kv_cache(tail[0, :, 0, 0], tail[0, :, 1, 0], 0)
            output = batch.batch_attn_fwd(q, 0)
            self.assertTrue(torch.isfinite(output).all())
            for host in (batch._host_rope_metadata, batch._gpu_attention._host_metadata):
                self.assertEqual(host.device.type, "cpu")
                self.assertTrue(host.is_pinned())
            batch.append_batch_kv_cache(tail[1, :, 0, 0], tail[1, :, 1, 0], 1)
            with torch.cuda.stream(torch.cuda.Stream()):
                with self.assertRaisesRegex(RuntimeError, "share one compute stream"):
                    batch.batch_attn_fwd(q, 1)
            self.assertTrue(frame.region._forward_failed)

    def packed(self, history_length, queries, heads, dtype, *, draft=False, all_hot=False):
        layers, dim = 2, 128
        history = torch.randn((layers, history_length, 2, 1, heads, dim), device="cuda", dtype=dtype)
        tail = torch.randn((layers, queries, 2, 1, heads, dim), device="cuda", dtype=dtype)
        q = torch.randn((queries, heads * 4, dim), device="cuda", dtype=dtype)
        nselected = min(47, history_length - 2)
        selection = torch.stack([torch.stack([torch.randperm(history_length - 2, device="cuda")[:nselected]
                                             for _ in range(heads)]) for _ in range(layers)]).int()
        plan = partition_pages(torch.arange(history_length, device="cuda"), selection,
                               0 if all_hot else nselected, pool_capacity=history_length,
                               recent_count=history_length if all_hot else 2, validate=True)
        region = PackedKVRegion()
        region.phase = "packed"
        region.device, region.dtype = q.device, dtype
        region.layers, region.heads, region.dim = layers, heads, dim
        region.history_length = history_length
        region.hot_count, region.cold_count = plan.hot_count, plan.cold_count
        region.recent_count, region.selected_count = plan.recent_count, plan.selected_count
        region.tail_capacity, region.tail_length = queries + 2, 0 if draft else queries - 1
        region._hot = torch.full((layers, plan.hot_count + queries + 2, 2, 1, heads, dim),
                                 float("nan"), device="cuda", dtype=dtype)
        region._hot[:, :plan.hot_count] = reference_gather(history, plan.hot_pages)
        cold = reference_gather(history, plan.cold_pages)
        restored = RestoredChunk(cold, 0, plan.cold_count, torch.cuda.current_stream().record_event())
        frame = PackedForward(region, mode="draft" if draft else "verify", query_tokens=queries,
                              restored=None if draft else restored)
        return frame, q, tail, history, plan

    def test_unsupported_head_channel_strides_fail_before_kv_writes(self):
        frame, q, tail, _, _ = self.packed(133, 1, 4, torch.bfloat16, draft=True)
        batch = PackedBatch([frame])
        strided = torch.empty((*q.shape[:-1], 2 * q.shape[-1]), device="cuda", dtype=q.dtype)[..., ::2]
        batch.append_batch_kv_cache(tail[0, :, 0, 0], tail[0, :, 1, 0], 0)
        with self.assertRaisesRegex(ValueError, "contiguous head/channel"):
            batch.batch_attn_fwd(strided, 0)
        self.assertFalse(frame.region._forward_active)
        self.assertFalse(getattr(frame.region, "_forward_failed", False))

    def test_long_ragged_batch_outputs_lse_and_selection_scores(self):
        torch.manual_seed(714)
        for heads, dtype in ((4, torch.float16), (8, torch.bfloat16)):
            fixtures = [self.packed(20137, 15, heads, dtype),
                        self.packed(133, 5, heads, dtype, all_hot=True),
                        self.packed(337, 1, heads, dtype, draft=True)]
            frames = [f[0] for f in fixtures]
            q = torch.cat([f[1] for f in fixtures])
            batch = PackedBatch(frames)
            offset = 0
            for layer in range(2):
                keys = torch.cat([f[2][layer, :, 0, 0] for f in fixtures])
                values = torch.cat([f[2][layer, :, 1, 0] for f in fixtures])
                batch.append_batch_kv_cache(keys, values, layer)
                output = batch.batch_attn_fwd(q, layer)
                offset = 0
                for frame, queries, tail, history, plan in fixtures:
                    draft = frame.mode == "draft"
                    prefix = (reference_gather(history, plan.hot_pages)[layer] if draft else history[layer])
                    expected, lse, scores = full_reference(queries, torch.cat((prefix, tail[layer])), len(prefix))
                    end = offset + len(queries)
                    torch.testing.assert_close(output[offset:end].double(), expected, rtol=.025, atol=.006)
                    torch.testing.assert_close(batch._gpu_attention.lse[offset:end].double(), lse,
                                               rtol=.002, atol=.005)
                    if not draft:
                        for head in range(heads):
                            hot_ids = torch.cat((plan.hot_pages[layer, head].long(),
                                                 torch.arange(len(history[layer]),
                                                              len(history[layer]) + len(queries), device="cuda")))
                            torch.testing.assert_close(frame.hot_scores[layer, head].double(), scores[head, hot_ids],
                                                       rtol=.04, atol=3e-5)
                            torch.testing.assert_close(frame.cold_scores[layer, head].double(),
                                                       scores[head, plan.cold_pages[layer, head].long()],
                                                       rtol=.04, atol=3e-5)
                    offset = end
            batch.finish()
            self.assertGreater(batch._gpu_attention.stats["maximum_splits"], 1)
            self.assertGreater(batch._gpu_attention.stats["TBs_per_SM"], 0)

    def test_chunk_prefill_and_verify_share_grid_without_reading_poisoned_capacity(self):
        torch.manual_seed(715)
        dtype, heads, dim = torch.bfloat16, 4, 128
        pool = KVBufferPool(16 * 1024**2, device="cuda")
        region = FullKVRegion(pool, layers=2, capacity=411, heads=heads, dim=dim, dtype=dtype)
        region.kv.fill_(float("nan"))
        region.kv[:, :137] = torch.randn((2, 137, 2, 1, heads, dim), device="cuda", dtype=dtype)
        region.length = 137
        full = FullForward(region, query_tokens=129)
        fixture = self.packed(2051, 15, heads, dtype)
        batch = PackedBatch([full, fixture[0]])
        q0 = torch.randn((129, heads * 4, dim), device="cuda", dtype=dtype)
        tail0 = torch.randn((2, 129, 2, 1, heads, dim), device="cuda", dtype=dtype)
        q = torch.cat((q0, fixture[1]))
        try:
            for layer in range(2):
                keys = torch.cat((tail0[layer, :, 0, 0], fixture[2][layer, :, 0, 0]))
                values = torch.cat((tail0[layer, :, 1, 0], fixture[2][layer, :, 1, 0]))
                batch.append_batch_kv_cache(keys, values, layer)
                output = batch.batch_attn_fwd(q, layer)
                expected, _, _ = full_reference(q0, torch.cat((region.kv[layer, :137], tail0[layer])), 137)
                torch.testing.assert_close(output[:129].double(), expected, rtol=.025, atol=.006)
                self.assertTrue(torch.isfinite(output).all())
            batch.finish()
            self.assertEqual(region.length, 266)
        finally:
            if not batch._finished:
                batch.abort()
            region.close()
            pool.close(wait=True)


if __name__ == "__main__":
    unittest.main()
