"""CPU partition checks and opt-in CUDA migration checks.

RUN_KV_PARTITION_GPU=1 enables CUDA tests once a GPU is available.
"""

import os
import unittest
from unittest.mock import Mock, patch

import torch

from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.kv_partition import PackedKVRegion, RestoredChunk, partition_pages
from serve.tests.test_kv_buffer_pool import CompletionEvent
from serve.tests.test_kv_pack import reference_pack


def make_selection(heads=2):
    full = torch.tensor([8, 1, 6, 2, 10, 4, 0], dtype=torch.int32)
    rows = []
    for layer in range(2):
        rows.append(torch.stack([
            torch.tensor([full[(layer + head) % 5], full[(layer + head + 2) % 5], -1])
            for head in range(heads)
        ]))
    return full, torch.stack(rows).int()


class PagePartitionTest(unittest.TestCase):
    def test_per_head_partition_and_recent_suffix(self):
        full, selected = make_selection()
        plan = partition_pages(full, selected, 2, pool_capacity=13, recent_count=2, validate=True)
        self.assertEqual((plan.hot_count, plan.cold_count), (4, 3))
        for layer in range(2):
            for head in range(2):
                hot = selected[layer, head, :2].tolist() + full[-2:].tolist()
                cold = [page for page in full.tolist() if page not in hot]
                self.assertEqual(plan.hot_pages[layer, head].tolist(), hot)
                self.assertEqual(plan.cold_pages[layer, head].tolist(), cold)
                self.assertEqual(sorted(hot + cold), sorted(full.tolist()))
        # Ownership metadata is a snapshot, not a view into the old owner's list.
        full[0] = 12
        self.assertEqual(plan.full_pages[0].item(), 8)

    def test_empty_and_all_hot(self):
        full, selected = make_selection()
        empty = partition_pages(full, selected, 0, pool_capacity=13, validate=True)
        self.assertEqual(empty.hot_count, 0)
        self.assertEqual(empty.cold_pages[0, 0].tolist(), full.tolist())
        all_hot = full.expand(2, 2, -1)
        plan = partition_pages(full, all_hot, 7, pool_capacity=13, validate=True)
        self.assertEqual(plan.cold_pages.shape, (2, 2, 0))
        recent_only = partition_pages(full, selected, 0, pool_capacity=13,
                                      recent_count=7, validate=True)
        self.assertEqual(recent_only.hot_pages[0, 0].tolist(), full.tolist())
        zero = partition_pages(full[:0], selected, 0, pool_capacity=13, validate=True)
        self.assertEqual((zero.hot_count, zero.cold_count), (0, 0))

    def test_reject_invalid_selection(self):
        full, selected = make_selection()
        invalid = [selected.clone() for _ in range(3)]
        invalid[0][0, 0, 1] = invalid[0][0, 0, 0]  # Duplicate.
        invalid[1][0, 0, 0] = 12  # Another request's page.
        invalid[2][0, 0, 0] = 0  # Recent page would be included twice.
        for indices in invalid:
            with self.subTest(indices=indices.tolist()), self.assertRaises(ValueError):
                partition_pages(full, indices, 2, pool_capacity=13, recent_count=2, validate=True)
        with self.assertRaises(ValueError):
            partition_pages(torch.tensor([1, 1]), selected, 0, pool_capacity=13, validate=True)

    def test_tp_local_partitions_match_global_heads(self):
        full, selected = make_selection(heads=8)
        global_plan = partition_pages(full, selected, 2, pool_capacity=13, recent_count=2, validate=True)
        pieces = [partition_pages(full, selected[:, start:start + 4], 2,
                                  pool_capacity=13, recent_count=2, validate=True)
                  for start in (0, 4)]
        torch.testing.assert_close(torch.cat([p.hot_pages for p in pieces], dim=1), global_plan.hot_pages)
        torch.testing.assert_close(torch.cat([p.cold_pages for p in pieces], dim=1), global_plan.cold_pages)

    def test_migration_boundary_protects_source_retirement(self):
        region = PackedKVRegion()
        region.phase = "offloading"
        region.offload_ready = Mock()
        region._source_pages = torch.tensor([8, 1, 6])
        region._source = region._staging = region._pending_plan = object()
        region._leases = {}
        region._source_claimed = False
        with self.assertRaises(RuntimeError):
            region.source_pages_to_release()
        region.finish_offload()
        region.finish_offload()
        region.offload_ready.synchronize.assert_called_once()
        self.assertIsNone(region._source)
        self.assertIsNone(region._staging)
        self.assertEqual(region.source_pages_to_release().tolist(), [8, 1, 6])
        with self.assertRaises(RuntimeError):
            region.source_pages_to_release()

    def test_nonblocking_completion_keeps_sources_and_hot_lease_owned(self):
        pool = KVBufferPool(256, pin_memory=False)
        region = PackedKVRegion()
        region.phase = "offloading"
        region.offload_ready = CompletionEvent()
        region._leases = {"hot": pool.allocate((128,), torch.float16)}
        region._source = region._pending_plan = object()
        region._staging = None
        self.assertFalse(region.finish_offload(wait=False))
        self.assertIsNotNone(region._source)
        self.assertEqual(pool.stats()["in_use_bytes"], 256)
        self.assertEqual(region.phase, "offloading")
        region.offload_ready.ready = True
        self.assertTrue(region.finish_offload(wait=False))
        self.assertEqual(region.offload_ready.waits, 0)
        self.assertIsNone(region._source)
        self.assertEqual(pool.stats()["in_use_bytes"], 256)
        region._leases["hot"].release()
        pool.close()

    def test_retirement_waits_for_cpu_restore_and_gpu_consumer(self):
        cpu = KVBufferPool(256, pin_memory=False)
        gpu_stand_in = KVBufferPool(256, pin_memory=False)
        region = PackedKVRegion()
        region.phase = "packed"
        region.offload_ready = CompletionEvent(ready=True)
        restore_done, consumer_done = CompletionEvent(), CompletionEvent()
        region._restore_events = [restore_done]
        region._leases = {"cold": cpu.allocate((128,), torch.float16),
                          "hot": gpu_stand_in.allocate((128,), torch.float16)}
        region.close(consumer_done=consumer_done, wait=False)
        self.assertEqual(region.phase, "closed")
        self.assertEqual(cpu.stats()["pending_bytes"], 256)
        self.assertEqual(gpu_stand_in.stats()["pending_bytes"], 256)
        with self.assertRaises(RuntimeError):
            _ = region.cold_cpu
        restore_done.ready = True
        self.assertEqual(cpu.stats()["in_use_bytes"], 0)
        self.assertEqual(gpu_stand_in.stats()["in_use_bytes"], 256)
        consumer_done.ready = True
        self.assertEqual(gpu_stand_in.stats()["in_use_bytes"], 0)
        self.assertEqual((restore_done.waits, consumer_done.waits), (0, 0))
        cpu.close()
        gpu_stand_in.close()

    def test_restored_workspace_waits_for_copy_and_consumer(self):
        pool = KVBufferPool(256, pin_memory=False)
        lease = pool.allocate((128,), torch.float16)
        copied, consumed = CompletionEvent(), CompletionEvent()
        chunk = RestoredChunk(lease.tensor, 0, 1, copied, lease)
        with self.assertRaises(ValueError):
            chunk.release(consumer_done=None)
        self.assertEqual(pool.stats()["active_leases"], 1)
        chunk.release(consumer_done=consumed)
        self.assertEqual(pool.stats()["pending_bytes"], 256)
        with self.assertRaises(RuntimeError):
            chunk.release(consumer_done=consumed)
        copied.ready = True
        self.assertEqual(pool.stats()["in_use_bytes"], 256)
        consumed.ready = True
        self.assertEqual(pool.stats()["in_use_bytes"], 0)
        self.assertEqual((copied.waits, consumed.waits), (0, 0))
        pool.close()


@unittest.skipUnless(os.environ.get("RUN_KV_PARTITION_GPU") == "1" and torch.cuda.is_available(),
                     "CUDA migration tests await an available GPU")
class PackedKVRegionTest(unittest.TestCase):
    def make_region(self, recent=2, selected_count=2, *, cpu_pool=None, gpu_pool=None,
                    hot_capacity=None, cold_capacity=None):
        full, selected = make_selection()
        full, selected = full.cuda(), selected.cuda()
        plan = partition_pages(full, selected, selected_count, pool_capacity=13,
                               recent_count=recent, validate=True)
        source = torch.arange(2 * 13 * 2 * 2 * 8, device="cuda", dtype=torch.float32)
        source = source.reshape(2, 13, 2, 1, 2, 8).half()
        producer_ready = torch.cuda.current_stream().record_event()
        region = PackedKVRegion.start_offload(source, plan, tail_capacity=4, chunk_pages=2,
                                              source_ready=producer_ready, stream=torch.cuda.Stream(),
                                              cpu_pool=cpu_pool, gpu_pool=gpu_pool,
                                              hot_capacity=hot_capacity, cold_capacity=cold_capacity)
        return source, plan, region

    def test_reserved_capacity_keeps_live_cold_contiguous_without_padding_transfer(self):
        cpu = KVBufferPool(4096)
        gpu = KVBufferPool(8192, device="cuda")
        cpu._buffer.view(torch.float16).fill_(1234)
        source, plan, region = self.make_region(cpu_pool=cpu, gpu_pool=gpu,
                                              hot_capacity=12, cold_capacity=9)
        restored = None
        try:
            region.finish_offload()
            self.assertEqual(region.hot_kv.shape[1], plan.hot_count + 4)
            self.assertEqual(region._cold_cpu.shape[1], plan.cold_count)
            self.assertTrue(region._cold_cpu.is_contiguous())
            torch.testing.assert_close(region._cold_cpu, reference_pack(source, plan.cold_pages, plan.cold_count).cpu())
            live = region._cold_cpu.numel()
            self.assertTrue(torch.all(region._leases["cold"].tensor.reshape(-1)[live:] == 1234))
            original = PackedKVRegion._copy_contiguous_layers
            with patch.object(PackedKVRegion, "_copy_contiguous_layers", side_effect=original) as copy:
                restored = region.restore_chunk(0, region.cold_count)
                restored.ready.synchronize()
                self.assertEqual(copy.call_count, 1)
                self.assertEqual(copy.call_args.args[1].nbytes, region._cold_cpu.nbytes)
                self.assertLess(copy.call_args.args[1].nbytes, region._leases["cold"].tensor.nbytes)
            torch.testing.assert_close(restored.kv, reference_pack(source, plan.cold_pages, plan.cold_count))
        finally:
            if restored is not None:
                restored.release(consumer_done=restored.ready)
            region.close(consumer_done=torch.cuda.current_stream().record_event())
            gpu.close(wait=True); cpu.close(wait=True)

    def test_hot_and_cold_round_trip(self):
        source, plan, region = self.make_region()
        with self.assertRaises(RuntimeError):
            _ = region.hot_kv
        region.finish_offload()
        torch.testing.assert_close(region.hot_kv[:, :plan.hot_count],
                                   reference_pack(source, plan.hot_pages, plan.hot_count), rtol=0, atol=0)
        expected = reference_pack(source, plan.cold_pages, plan.cold_count)
        torch.testing.assert_close(region.cold_cpu, expected.cpu(), rtol=0, atol=0)
        self.assertTrue(region.cold_cpu.is_pinned())
        cold_before = region.cold_cpu.clone()
        chunk = region.restore_chunk(1, 2)
        torch.cuda.current_stream().wait_event(chunk.ready)
        torch.testing.assert_close(chunk.kv, expected[:, 1:3], rtol=0, atol=0)
        torch.testing.assert_close(region.cold_cpu, cold_before, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            region.restore_chunk(0, 1, destination=region.hot_kv)

    def test_reusable_restore_workspace(self):
        source, plan, region = self.make_region()
        region.finish_offload()
        workspace = torch.full((2, 3, 2, 1, 2, 8), -17, device="cuda", dtype=source.dtype)
        chunk = region.restore_chunk(0, 2, destination=workspace, stream=torch.cuda.Stream())
        current = torch.cuda.current_stream()
        current.wait_event(chunk.ready)
        snapshot = workspace[:, :2].clone()
        consumed = current.record_event()
        last = region.restore_chunk(2, 1, destination=workspace, stream=torch.cuda.Stream(),
                                    consumer_done=consumed)
        current.wait_event(last.ready)
        expected = reference_pack(source, plan.cold_pages, plan.cold_count)
        torch.testing.assert_close(snapshot, expected[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(workspace[:, :1], expected[:, 2:3], rtol=0, atol=0)
        self.assertTrue(torch.all(workspace[:, 2:] == -17).item())

    def test_tail_write_and_rejection_truncation(self):
        source, plan, region = self.make_region()
        region.finish_offload()
        keys = torch.full((2, 2, 8), 100, device="cuda", dtype=source.dtype)
        values = keys + 10
        for layer in range(2):
            region.write_tail(layer, keys, values, 0)
        region.set_tail_length(2)
        self.assertEqual(region.hot_length, plan.hot_count + 2)
        for layer in range(2):
            torch.testing.assert_close(region.hot_kv[layer, plan.hot_count:plan.hot_count + 2, 0, 0], keys)
            torch.testing.assert_close(region.hot_kv[layer, plan.hot_count:plan.hot_count + 2, 1, 0], values)
        region.set_tail_length(1)
        self.assertEqual(region.hot_length, plan.hot_count + 1)
        with self.assertRaises(ValueError):
            region.set_tail_length(5)

    def test_no_cold_pages(self):
        source, plan, region = self.make_region(recent=7, selected_count=0)
        region.finish_offload()
        self.assertEqual(region.cold_count, 0)
        chunk = region.restore_chunk(0, 0)
        torch.cuda.current_stream().wait_event(chunk.ready)
        self.assertEqual(chunk.kv.shape[1], 0)
        torch.testing.assert_close(region.hot_kv[:, :7], reference_pack(source, plan.hot_pages, 7))

    def test_pooled_round_trip_and_retirement(self):
        cpu_pool = KVBufferPool(4096)
        gpu_pool = KVBufferPool(8192, device="cuda")
        source, plan, region = self.make_region(cpu_pool=cpu_pool, gpu_pool=gpu_pool)
        self.assertIsNone(region._staging)
        self.assertEqual(set(region._leases), {"hot", "cold"})
        region.finish_offload()
        self.assertEqual(gpu_pool.stats()["active_leases"], 1)  # Only hot remains.
        expected = reference_pack(source, plan.cold_pages, plan.cold_count)
        chunk = region.restore_chunk(0, plan.cold_count, stream=torch.cuda.Stream())
        current = torch.cuda.current_stream()
        current.wait_event(chunk.ready)
        snapshot = chunk.kv.clone()
        consumed = current.record_event()
        chunk.release(consumer_done=consumed)
        torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
        torch.testing.assert_close(region.cold_cpu, expected.cpu(), rtol=0, atol=0)
        region.close(consumer_done=current.record_event())
        self.assertEqual(cpu_pool.stats()["in_use_bytes"], 0)
        self.assertEqual(gpu_pool.stats()["in_use_bytes"], 0)
        cpu_pool.close()
        gpu_pool.close()

    def test_pool_exhaustion_rolls_back_before_migration(self):
        cpu_pool = KVBufferPool(4096)
        gpu_pool = KVBufferPool(256, device="cuda")  # Too small for hot KV.
        with self.assertRaises(MemoryError):
            self.make_region(cpu_pool=cpu_pool, gpu_pool=gpu_pool)
        self.assertEqual(cpu_pool.stats()["in_use_bytes"], 0)
        self.assertEqual(gpu_pool.stats()["in_use_bytes"], 0)
        cpu_pool.close()
        gpu_pool.close()


if __name__ == "__main__":
    unittest.main()
