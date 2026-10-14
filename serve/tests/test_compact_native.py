"""Ownership/epoch tests on CPU, actual same-pool KV copies on CUDA."""

import array
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from serve.attention.kv_pack import gather_to_pool, restore_to_pool
from serve.attention.kv_partition import partition_pages
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request


class Pool:
    def __init__(self, device="cpu", capacity=64):
        self.device, self.capacity, self.page_size, self.vocab_size = device, capacity, 1, 32
        self._free_gpu = set(range(capacity))
        self._physical_mem_gpu = torch.randn(2, capacity, 2, 1, 2, 128, device=device)

    @property
    def num_free_pages(self):
        return len(self._free_gpu)

    def alloc_page(self):
        return self._free_gpu.pop()

    def free_page(self, page):
        assert page not in self._free_gpu
        self._free_gpu.add(page)


class LocalPtr(CompactCachePtr):
    NUM_LAYERS = NUM_KV_HEADS = 2


def selected_copy(source, selection):
    result = torch.empty((source.shape[0], selection.shape[-1], *source.shape[2:]),
                         dtype=source.dtype, device=source.device)
    for layer, head in itertools.product(range(source.shape[0]), range(source.shape[4])):
        result[layer, :, :, :, head] = source[layer, selection[layer, head].long(), :, :, head]
    return result


class CPUOwnership(unittest.TestCase):
    def test_append_capability_preserves_legacy_streaming(self):
        from serve.request.kv_cache_ptr.streaming import StreamingCachePtr
        self.assertFalse(FullCachePtr.APPEND_USES_ALL_KV)
        self.assertFalse(StreamingCachePtr.APPEND_USES_ALL_KV)
        self.assertTrue(PillarCachePtr.APPEND_USES_ALL_KV)
        self.assertTrue(CompactCachePtr.APPEND_USES_ALL_KV)

    def new_ptr(self, device="cpu"):
        pool = Pool(device)
        pages = (8, 2, 11, 17, 4, 24, 0, 9, 20, 15, 6, 30)
        pool._free_gpu.difference_update(pages)
        source = FullCachePtr(pool, array.array("i", pages), len(pages))
        ptr, _ = LocalPtr.from_kv_cache(source)
        self.assertEqual(len(source.indices), 0)
        del source
        return pool, ptr

    def select(self, ptr, accepted=1):
        ptr.freeze_verify_view()
        ptr.retract_tokens(0)
        metadata = ptr.get_select_metadata(num_min_budget=5, budget_ratio=.1, num_accepted_tokens=accepted)
        count = metadata["num_selected_tokens"]
        prefix = list(ptr.indices)[:-ptr.recent_count]
        choices = torch.tensor([[prefix[i:i + count] for i in range(2)],
                                [prefix[i + 2:i + 2 + count] for i in range(2)]],
                               dtype=torch.int32, device=ptr.kv_pool.device)
        ptr.selected_indices_buffer_gpu[..., :count] = choices
        return choices

    def test_full_adoption_append_tail_retract_and_exactly_once_close(self):
        pool, ptr = self.new_ptr()
        initial = ptr.owned_pages
        ptr.allocate_tokens(1)
        ptr.allocate_tokens(1)
        self.assertEqual((ptr.len_kv_cache, len(ptr.indices), ptr.len_loading_indices), (14, 14, 14))
        ptr.freeze_verify_view()
        ptr.retract_tokens(1)
        self.assertEqual(ptr.owned_pages - initial, set(ptr.tail_pages))
        self.assertEqual(len(ptr.verify_pages), 14)
        self.assertEqual(len(ptr.indices), 13)
        with self.assertRaises(RuntimeError):
            ptr.retract_tokens(2)
        ptr.close()
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_physical_append_length_does_not_replace_logical_rope_offset(self):
        pool, ptr = self.new_ptr()
        ptr.indices = array.array("i", ptr.indices[:5])
        ptr.len_loading_indices = 5
        request = Request(list(range(13)), 20)
        request.status = Request.ReqExecType.DRAFT
        request.kv_cache_ptr = ptr
        host = dict(token_ids=array.array("i", [12]), token_indptr=array.array("i", [0, 1]))
        tensors = {"pos_offsets": torch.tensor([12], dtype=torch.int32)}
        with patch.object(PillarCachePtr, "dispatch_prepare_fwd_metadata"):
            LocalPtr.dispatch_prepare_fwd_metadata(input_metadata=host, input_metadata_tensors=tensors,
                                                  active_requests=[request], device="cpu")
        self.assertEqual(tensors["all_kv_len_arr"].tolist(), [5])
        self.assertEqual(tensors["pos_offsets"].tolist(), [12])
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_recent_budget_and_epoch_guards(self):
        pool, ptr = self.new_ptr()
        with self.assertRaises(RuntimeError):
            ptr.get_select_metadata(num_accepted_tokens=1)
        ptr.freeze_verify_view()
        with self.assertRaises(ValueError):
            ptr.get_select_metadata(num_min_budget=2, budget_ratio=.1, num_accepted_tokens=4)
        with self.assertRaises(ValueError):
            ptr.get_select_metadata(num_min_budget=6144, budget_ratio=.1, num_accepted_tokens=1)
        ptr.epoch += 1
        with self.assertRaises(RuntimeError):
            ptr.retract_tokens(0)
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_partition_membership_duplicates_and_per_head_complement(self):
        pool, ptr = self.new_ptr()
        self.select(ptr)
        full = torch.tensor(ptr.indices, dtype=torch.int32)
        plan = partition_pages(full, ptr.selected_indices_buffer_gpu, 3,
                               pool_capacity=pool.capacity, recent_count=2, validate=True)
        self.assertEqual((plan.hot_count, plan.cold_count), (5, 7))
        for layer, head in itertools.product(range(2), repeat=2):
            values = torch.cat([plan.hot_pages[layer, head], plan.cold_pages[layer, head]])
            self.assertEqual(sorted(values.tolist()), sorted(full.tolist()))
        ptr.selected_indices_buffer_gpu[0, 0, 1] = ptr.selected_indices_buffer_gpu[0, 0, 0]
        with self.assertRaises(ValueError):
            partition_pages(full, ptr.selected_indices_buffer_gpu, 3,
                            pool_capacity=pool.capacity, recent_count=2, validate=True)
        ptr.selected_indices_buffer_gpu[0, 0, 1] = 63
        with self.assertRaises(ValueError):
            partition_pages(full, ptr.selected_indices_buffer_gpu, 3,
                            pool_capacity=pool.capacity, recent_count=2, validate=True)
        ptr.close()


class CPUCapacity(unittest.TestCase):
    @staticmethod
    def scheduler():
        # scheduler.py creates its legacy async stream at import time; no GPU
        # is required for these pure admission/queue tests.
        with patch("torch.cuda.Stream", return_value=SimpleNamespace()):
            from serve.scheduler.compact_scheduler import CompactSpecScheduler
        scheduler = CompactSpecScheduler.__new__(CompactSpecScheduler)
        scheduler.spec_stride = 4
        scheduler.spec_cache_args = dict(budget_ratio=.1, num_min_budget=128)
        scheduler.kv_pool = SimpleNamespace(capacity=24000, num_free_pages=700)
        scheduler.cpu_capacity = 40000
        scheduler.max_batch_size = 4
        scheduler.group, scheduler.rank = None, 0
        scheduler._last_served = {}
        scheduler._args = dict(chunk_size=256)
        scheduler.metrics = dict(admission_waits=0, capacity_waits=0, max_live_requests=0)
        scheduler.active_requests, scheduler.offload_requests = [], []
        return scheduler

    def test_long_request_peak_and_all_full_to_hot_transitions(self):
        scheduler = self.scheduler()
        requests = [Request([1] * n, 512) for n in (20000, 8000, 128, 29)]
        self.assertTrue(scheduler._fits(requests[:1]))
        initial = [scheduler._estimate(request) for request in requests]
        reserved = sum(x[0] for x in initial) + max(x[1] for x in initial)
        for flags in itertools.product((False, True), repeat=len(requests)):
            for request, packed in zip(requests, flags):
                if packed:
                    ptr = CompactCachePtr.__new__(CompactCachePtr)
                    ptr._closed, ptr.phase = True, "hot"
                    request.kv_cache_ptr = ptr
                else:
                    request.kv_cache_ptr = None
            now = [scheduler._estimate(request) for request in requests]
            self.assertLessEqual(sum(x[0] for x in now) + max(x[1] for x in now), reserved)
        scheduler.kv_pool.capacity = 21000
        self.assertFalse(scheduler._fits([Request([1] * 20000, 512)]))

    def test_ready_verify_waits_without_freeing_and_next_turn_has_priority(self):
        scheduler = self.scheduler()
        verify = Request.ReqExecType.VERIFY
        draft = Request.ReqExecType.DRAFT
        requests = [SimpleNamespace(request_id=i, status=status, len_kv_cache=700, initial_idx=0,
            kv_cache_ptr=SimpleNamespace(phase="hot", cold_count=500, len_loading_indices=200, len_kv_cache=700))
            for i, status in enumerate((verify, verify, draft))]
        scheduler.active_requests = requests[:]
        scheduler.admit_requests([])
        self.assertEqual([r.request_id for r in scheduler.active_requests], [0, 2])
        self.assertEqual([r.request_id for r in scheduler.offload_requests], [1])
        self.assertEqual(requests[1].len_kv_cache, 700)
        scheduler._last_served = {0: 0, 2: 0}
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests[0].request_id, 1)

    def test_tp_sampling_broadcast_dtype_for_greedy_and_stochastic(self):
        module = "serve.scheduler.compact_scheduler"
        for temperature, sampler_dtype in ((0., torch.int64), (.65, torch.int32)):
            for rank in (0, 1):
                scheduler = self.scheduler()
                scheduler.group, scheduler.rank = object(), rank
                scheduler.temperature, scheduler.sampling_metadata = temperature, None
                scheduler.sampler = lambda *args: (
                    torch.tensor([2, 3], dtype=sampler_dtype), torch.zeros(2, 5))
                calls = []

                def broadcast(tensor, src):
                    calls.append(tensor.dtype)
                    if tensor.ndim == 1:
                        self.assertEqual(tensor.dtype, torch.int32)
                        tensor.copy_(torch.tensor([2, 3], dtype=torch.int32))

                with patch(f"{module}.coordinated", side_effect=lambda group, label, op: op()), \
                     patch(f"{module}.dist.broadcast", side_effect=broadcast):
                    tokens, probs = scheduler._sample(torch.zeros(2, 5), 2)
                self.assertEqual(tokens.dtype, torch.int32)
                self.assertEqual(calls, [torch.int32] + ([torch.float32] if temperature else []))
                if rank == 1 and temperature == 0:
                    self.assertEqual(probs.sum().item(), 2.)
                    self.assertEqual(probs[0, 2].item(), 1.)

    def test_constructor_rejects_unsafe_recent_and_score_tile(self):
        scheduler = self.scheduler()
        cls = type(scheduler)
        pool = SimpleNamespace(num_qo_heads=32, num_kv_heads=8)
        for options in (dict(spec_stride=4, num_min_budget=5, budget_ratio=.1),
                        dict(spec_stride=4, num_min_budget=128, budget_ratio=float("nan")),
                        dict(spec_stride=16, num_min_budget=128, budget_ratio=.1)):
            with self.assertRaises(ValueError):
                cls(pool, None, **options)

    def test_native_metadata_limits_are_strict_and_checked_before_dispatch(self):
        self.assertEqual(CompactCachePtr.MAX_TOTAL_DRAFT_KV_LEN, 128*1024)
        self.assertEqual(CompactCachePtr.MAX_TOTAL_VERIFY_KV_LEN, 256*1024)

        def prepare(status, length, query=1, count=1):
            requests = [SimpleNamespace(status=status, len_loading_kv_cache=length,
                        kv_cache_ptr=SimpleNamespace(indices=range(length), freeze_verify_view=lambda: None))
                        for _ in range(count)]
            host = dict(token_ids=[0]*(query*count), token_indptr=list(range(0, query*(count+1), query)))
            LocalPtr.dispatch_prepare_fwd_metadata(input_metadata=host, input_metadata_tensors={},
                                                  active_requests=requests, device="cpu")

        with patch.object(PillarCachePtr, "dispatch_prepare_fwd_metadata") as original, \
             patch.multiple(LocalPtr, MAX_TOTAL_DRAFT_KV_LEN=16, MAX_TOTAL_VERIFY_KV_LEN=24,
                            _MAX_NNZ=12, _MAX_BATCH_SIZE=3, NUM_QO_HEADS=16, NUM_KV_HEADS=2):
            draft, verify = Request.ReqExecType.DRAFT, Request.ReqExecType.VERIFY
            prepare(draft, 15)
            prepare(verify, 23)
            self.assertEqual(original.call_count, 2)
            for status, length, query, count in ((draft,16,1,1), (verify,24,1,1),
                                               (draft,8,1,2), (verify,12,1,2),
                                               (draft,1,4,3), (draft,1,12,1), (verify,23,9,1)):
                with self.assertRaises(ValueError):
                    prepare(status, length, query, count)
            self.assertEqual(original.call_count, 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GPUContent(CPUOwnership):
    def test_zero_cold_boundary_restore_and_close(self):
        pool, ptr = self.new_ptr("cuda:0")
        ptr.freeze_verify_view()
        ptr.get_select_metadata(num_min_budget=128, budget_ratio=.1, num_accepted_tokens=1)
        ptr.selected_indices_buffer_gpu[..., :10] = torch.tensor(list(ptr.indices)[:10], device=pool.device)
        ptr.commit_pack(ptr.begin_pack(validate=True))
        self.assertEqual(ptr.cold_count, 0)
        ptr.allocate_tokens(1)
        ptr.commit_restore(ptr.begin_restore())
        self.assertEqual(len(ptr.indices), ptr.len_kv_cache)
        ptr.freeze_verify_view()
        ptr.retract_tokens(0)
        ptr.get_select_metadata(num_min_budget=128, budget_ratio=.1, num_accepted_tokens=1)
        ptr.selected_indices_buffer_gpu[..., :11] = torch.tensor(list(ptr.indices)[:11], device=pool.device)
        ptr.commit_pack(ptr.begin_pack(validate=True))
        self.assertEqual(ptr.cold_count, 0)
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)
    def test_pack_restore_repack_preserves_content_and_releases_sources(self):
        pool, ptr = self.new_ptr("cuda:0")
        self.select(ptr)
        full = torch.tensor(ptr.indices, device=pool.device, dtype=torch.int32)
        plan = partition_pages(full, ptr.selected_indices_buffer_gpu, 3,
                               pool_capacity=pool.capacity, recent_count=2, validate=True)
        expected_hot = selected_copy(pool._physical_mem_gpu, plan.hot_pages).clone()
        expected_cold = selected_copy(pool._physical_mem_gpu, plan.cold_pages).clone()
        old_pages = set(ptr.indices)
        transfer = ptr.begin_pack(validate=True)
        self.assertEqual(ptr.owned_pages, old_pages | set(transfer.pages))
        transfer.wait()
        ptr.commit_pack(transfer)
        self.assertEqual(ptr.owned_pages, set(ptr.hot_pages))
        self.assertEqual((ptr.len_kv_cache, len(ptr.indices), ptr.len_loading_indices), (12, 5, 5))
        torch.testing.assert_close(pool._physical_mem_gpu[:, list(ptr.hot_pages)], expected_hot, rtol=0, atol=0)
        torch.testing.assert_close(ptr.cold_cpu, expected_cold.cpu(), rtol=0, atol=0)
        for _ in range(3):
            ptr.allocate_tokens(1)
        tail = tuple(ptr.tail_pages)
        tail_values = pool._physical_mem_gpu[:, list(tail)].clone()
        transfer = ptr.begin_restore()
        transfer.wait()
        ptr.commit_restore(transfer)
        combined = torch.cat([expected_hot[:, :3], expected_cold, expected_hot[:, 3:], tail_values], dim=1)
        torch.testing.assert_close(pool._physical_mem_gpu[:, list(ptr.indices)], combined, rtol=0, atol=0)
        ptr.freeze_verify_view()
        ptr.retract_tokens(1)
        self.select(ptr, accepted=2)
        ptr.commit_pack(ptr.begin_pack(validate=True))
        self.assertEqual((ptr.len_kv_cache, len(ptr.indices), ptr.cold_count), (14, 5, 9))
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_scatter_rejects_overlap_duplicate_dest_and_unpinned_restore(self):
        pool, ptr = self.new_ptr("cuda:0")
        self.select(ptr)
        with self.assertRaises(ValueError):
            gather_to_pool(pool._physical_mem_gpu, ptr.selected_indices_buffer_gpu, [ptr.indices[0]],
                           source_page_ids=tuple(ptr.indices))
        with self.assertRaises(ValueError):
            gather_to_pool(pool._physical_mem_gpu, ptr.selected_indices_buffer_gpu, [1, 1],
                           source_page_ids=tuple(ptr.indices))
        with self.assertRaises(ValueError):
            restore_to_pool(torch.empty(2, 1, 2, 1, 2, 128), pool._physical_mem_gpu, [1])
        ptr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
