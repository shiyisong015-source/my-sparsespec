"""Ownership/epoch tests on CPU, actual same-pool KV copies on CUDA."""

import array
from collections import OrderedDict
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from serve.attention.kv_pack import gather_to_pool, restore_to_pool
from serve.attention.kv_partition import partition_pages
from serve.request.kv_cache_ptr.base import FullCachePtr
from serve.request.kv_cache_ptr.compact import CompactCachePtr, PageTransfer
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
        before_rejection = tuple(ptr.indices)
        ptr.freeze_verify_view()
        ptr.retract_tokens(1)
        self.assertEqual(ptr.owned_pages - initial, set(ptr.tail_pages))
        self.assertEqual(ptr.verify_epoch, ptr.epoch)
        self.assertEqual(tuple(ptr.indices), before_rejection[:-1])
        self.assertEqual(len(ptr.indices), 13)
        with self.assertRaises(RuntimeError):
            ptr.retract_tokens(2)
        ptr.close()
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_resident_verification_does_not_iterate_the_cpu_page_table(self):
        class NoIterationPages(array.array):
            def __iter__(self):
                raise AssertionError("Full CPU page-table traversal during verification")

        pool, ptr = self.new_ptr()
        ptr.indices = NoIterationPages("i", ptr.indices)
        ptr.freeze_verify_view()
        ptr.retract_tokens(0)
        ptr.get_select_metadata(num_min_budget=5, budget_ratio=.1, num_accepted_tokens=1)
        ptr.keep_full_resident()
        self.assertEqual(ptr.phase, "resident")
        self.assertIsNone(ptr.verify_epoch)
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_physical_append_length_does_not_replace_logical_rope_offset(self):
        # The legacy scheduler creates a CUDA stream at import time.
        with patch("torch.cuda.Stream", return_value=SimpleNamespace()):
            from serve.scheduler.spec_scheduler import SpecScheduler

        # Exercise the real collector for both a full resident view and a view
        # omitting cold pages, alongside an original FullCachePtr prefill.
        for physical_count in (12, 5):
            with self.subTest(physical_count=physical_count):
                pool, ptr = self.new_ptr()
                ptr.indices = array.array("i", ptr.indices[:physical_count])
                ptr.len_loading_indices = 3
                draft = Request(list(range(13)), 20)
                draft.status = Request.ReqExecType.DRAFT
                draft.kv_cache_ptr = ptr
                draft.draft_probs = torch.empty((8, pool.vocab_size))
                prefill = Request([21, 22, 23], 20)
                prefill.kv_cache_ptr = FullCachePtr(pool)
                scheduler = SpecScheduler.__new__(SpecScheduler)
                scheduler.active_requests = [draft, prefill]
                scheduler.model = SimpleNamespace(device="cpu")
                scheduler._args = {"chunk_size": 256}
                scheduler.spec_cache_class = LocalPtr
                scheduler.spec_cache_args = {}
                uploaded_lengths = []

                def native_dispatch(**kwargs):
                    host = kwargs["input_metadata"]
                    tensors = kwargs["input_metadata_tensors"]
                    uploaded_lengths.append(tensors["all_kv_len_arr"])
                    self.assertEqual(list(host["all_kv_len_arr"]), [physical_count + 1, 3])
                    # The one common upload already contains the right values.
                    self.assertEqual(tensors["all_kv_len_arr"].tolist(), [physical_count + 1, 3])

                try:
                    with patch("serve.scheduler.spec_scheduler.nvtx.range_push"), \
                         patch("serve.scheduler.spec_scheduler.nvtx.range_pop"), \
                         patch.object(PillarCachePtr, "dispatch_prepare_fwd_metadata", side_effect=native_dispatch), \
                         patch("serve.scheduler.spec_scheduler.torch.frombuffer", wraps=torch.frombuffer) as uploads, \
                         patch("serve.request.kv_cache_ptr.compact.torch.tensor",
                               side_effect=AssertionError("Redundant metadata tensor creation")):
                        host, tensors = scheduler.prepare_forward_inputs()
                    length_uploads = [call for call in uploads.call_args_list
                                      if call.args[0] is host["all_kv_len_arr"]]
                    self.assertEqual(len(length_uploads), 1)
                    self.assertIs(tensors["all_kv_len_arr"], uploaded_lengths[0])
                    self.assertEqual(tensors["pos_offsets"].tolist(), [12, 0])
                    self.assertEqual(tensors["kv_len_arr"].tolist(), [4, 3])
                    self.assertEqual((draft.len_kv_cache, prefill.len_kv_cache), (13, 3))
                finally:
                    ptr.close()
                    prefill.retire()
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

    def test_resident_selection_retains_full_pages_and_offload_includes_new_tail(self):
        pool, ptr = self.new_ptr()
        self.select(ptr)
        original = tuple(ptr.indices)
        ptr.keep_full_resident()
        self.assertEqual(ptr.phase, "resident")
        self.assertEqual(tuple(ptr.indices), original)
        self.assertEqual(ptr.len_loading_indices, 5)
        for _ in range(3):
            ptr.allocate_tokens(1)
        tail = tuple(ptr.tail_pages)
        ptr.prepare_resident_offload()
        self.assertEqual(ptr.recent_count, 5)
        plan = partition_pages(torch.tensor(ptr.indices, dtype=torch.int32),
                               ptr.selected_indices_buffer_gpu, ptr.len_selected_indices_gpu,
                               pool_capacity=pool.capacity, recent_count=ptr.recent_count, validate=True)
        self.assertEqual((plan.hot_count, plan.cold_count), (8, 7))
        self.assertEqual(plan.hot_pages[0, 0, -3:].tolist(), list(tail))
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_nonblocking_commit_rejects_unfinished_transfer_without_releasing_sources(self):
        pool, ptr = self.new_ptr()
        self.select(ptr)
        ptr.keep_full_resident()
        ptr.prepare_resident_offload()
        original = ptr.owned_pages
        pages = ptr._allocate(5)
        transfer = PageTransfer(pages, SimpleNamespace(query=lambda: False, synchronize=lambda: None),
                                cold=torch.empty(2, 7, 2, 1, 2, 128))
        ptr._pending = transfer
        with self.assertRaises(RuntimeError):
            ptr.commit_pack(transfer, nonblocking=True)
        self.assertTrue(original <= ptr.owned_pages)
        self.assertIs(ptr._pending, transfer)
        with self.assertRaises(RuntimeError):
            ptr.allocate_tokens(1)
        ptr.abort_transfer(transfer)
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)


class CPUGraphCapture(unittest.TestCase):
    def check_capture(self, cache_class, sparse):
        from serve.model.graph_model import CUDAGraphModelWrapper

        inputs = []
        pool = SimpleNamespace(page_size=1, capacity=4096,
                               kv_cache_class=cache_class, phase=0,
                               pre_attn_fwd=lambda **kwargs: inputs.append(kwargs))

        def step():
            pool.phase ^= 1

        pool.step = step

        class Model:
            kv_cache = pool
            device = "cpu"
            config = SimpleNamespace(vocab_size=32)

            def __call__(self, ids):
                return torch.zeros(ids.shape[0], 32)

        sentinel = torch.tensor([71], dtype=torch.int32)
        module = "serve.model.graph_model"
        with patch.object(cache_class, "draft_kv_indices_tensor_buf", sentinel, create=True), \
             patch(f"{module}.torch.cuda.empty_cache"), \
             patch(f"{module}.torch.cuda.memory_reserved", return_value=0), \
             patch(f"{module}._CUDAGraphModelRunner") as runner:
            wrapper = CUDAGraphModelWrapper(Model())
            wrapper.capture([1])
        self.assertEqual(runner.return_value.capture.call_count, 2)
        self.assertEqual(len(wrapper.graph_runners[1]), 2)
        self.assertEqual(pool.phase, 0)
        self.assertEqual(len(inputs), 2)
        for metadata in inputs:
            self.assertEqual("request_type" in metadata, sparse)
            if sparse:
                self.assertIs(metadata["flatten_indices"], sentinel)
                torch.testing.assert_close(metadata["all_kv_len_arr"], metadata["kv_len_arr"])
                torch.testing.assert_close(metadata["all_kv_indptr"], metadata["kv_indptr"])

    def test_compact_subclass_capture_uses_its_own_sparse_metadata(self):
        self.check_capture(LocalPtr, sparse=True)

    def test_full_cache_capture_does_not_require_sparse_metadata(self):
        self.check_capture(FullCachePtr, sparse=False)

    def test_release_waits_then_resets_both_phases_and_preserves_statistics(self):
        from serve.model.graph_model import CUDAGraphModelWrapper

        events = []
        model = SimpleNamespace(kv_cache=object(), device="cpu")
        wrapper = CUDAGraphModelWrapper(model)
        wrapper.graph_runners[8] = [
            SimpleNamespace(cuda_graph=SimpleNamespace(reset=lambda: events.append("phase0"))),
            SimpleNamespace(cuda_graph=SimpleNamespace(reset=lambda: events.append("phase1"))),
        ]
        wrapper.captured_bsz = [8]
        wrapper.num_total_tokens, wrapper.num_captured_tokens, wrapper.num_padded_tokens = 7, 7, 1
        with patch("torch.cuda.synchronize", side_effect=lambda device: events.append("wait")):
            wrapper.release_graphs()
            wrapper.release_graphs()
        self.assertEqual(events, ["wait", "phase0", "phase1"])
        self.assertFalse(wrapper.graph_runners)
        self.assertIsNone(wrapper.captured_bsz)
        self.assertEqual((wrapper.num_total_tokens, wrapper.num_captured_tokens,
                          wrapper.num_padded_tokens), (7, 7, 1))


class CPUDynamicAdmission(unittest.TestCase):
    @staticmethod
    def scheduler(capacity=128, budget=16):
        base = CPUAsyncPipeline.scheduler()
        from serve.scheduler.dynamic_compact_scheduler import DynamicCompactSpecScheduler
        scheduler = DynamicCompactSpecScheduler.__new__(DynamicCompactSpecScheduler)
        scheduler.__dict__.update(base.__dict__)
        scheduler.kv_pool = Pool(capacity=capacity)
        scheduler.token_budget, scheduler.early_verify_length = budget, None
        scheduler.cpu_admit_limit = scheduler.cpu_capacity
        scheduler._args.update(batch_policy="greedy", tp_ranks=2)
        scheduler._cold_lru, scheduler._preempt_requested = {}, {}
        scheduler._waiting_replays, scheduler._sample_rows = OrderedDict(), None
        scheduler.metrics.update(scheduled_token_peak=0, prefill_request_peak=0,
            prefill_query_tokens=0, replay_query_tokens=0, replay_starts=0,
            replay_completions=0, preemptions=0, cpu_evictions=0, cpu_evicted_pages=0,
            early_verifies=0, cpu_watermark_waits=0, token_budget_waits=0,
            metadata_waits=0, kv_capacity_waits=0, pure_replay_forwards=0,
            recovery_queue_peak=0, replay_admissions=0, recovery_memory_waits=0,
            recovery_preemption_waits=0, recovery_idle_reclaims=0,
            known_growth_reserve_peak=0)
        return scheduler

    def test_two_prefills_receive_an_exact_shared_token_plan(self):
        scheduler = self.scheduler(budget=10)
        scheduler._args["chunk_size"] = 8
        pending = [Request([1] * 20, 16) for _ in range(3)]
        for i, request in enumerate(pending):
            request.request_id = i
        scheduler.admit_requests(pending)
        self.assertEqual([r.scheduled_tokens for r in scheduler.active_requests], [8, 2])
        self.assertEqual(len(pending), 1)
        self.assertEqual(scheduler.metrics["prefill_request_peak"], 2)
        self.assertEqual(scheduler.metrics["known_growth_reserve_peak"], 30)
        for request in scheduler.active_requests:
            request.retire()

    def test_actual_cpu_occupancy_does_not_reserve_every_output_limit(self):
        scheduler = self.scheduler(budget=40)
        scheduler.max_batch_size = 40
        scheduler.cpu_capacity = scheduler.cpu_admit_limit = 1
        pending = [Request([1], 32767) for _ in range(40)]
        for i, request in enumerate(pending):
            request.request_id = i
        scheduler.admit_requests(pending)
        self.assertEqual(len(scheduler.active_requests), 40)
        self.assertEqual(scheduler.metrics["max_live_requests"], 40)
        for request in scheduler.active_requests:
            request.retire()

    def test_preemption_preserves_drafts_and_defers_old_page_release(self):
        scheduler = self.scheduler()
        request = Request([1, 2, 3], 10)
        request.request_id = 0
        request.attach_kv_pool(scheduler.kv_pool)
        request.kv_cache_ptr.allocate_tokens(5)
        request.token_ids.extend([4, 5, 6])
        request.status = Request.ReqExecType.DRAFT
        request.len_draft_probs = 2
        probs = request.draft_probs
        original_tokens = tuple(request.token_ids)
        scheduler._requests[0] = request
        scheduler._preempt_requested[0] = None
        free = scheduler.kv_pool.num_free_pages
        scheduler._apply_preemptions()
        self.assertEqual(tuple(request.token_ids), original_tokens)
        self.assertIs(request.draft_probs, probs)
        self.assertEqual((request.len_draft_probs, request.replay_target), (2, 3))
        self.assertEqual(scheduler._known_growth(request), 6)
        self.assertEqual(scheduler.kv_pool.num_free_pages, free)
        scheduler._retired[0].control_done = SimpleNamespace(query=lambda: True)
        scheduler._poll_retired()
        self.assertEqual(scheduler.kv_pool.num_free_pages, scheduler.kv_pool.capacity)
        request.kv_cache_ptr.allocate_tokens(3)
        scheduler._finish_replays()
        self.assertEqual(request.status, Request.ReqExecType.VERIFY)
        self.assertTrue(request.rebuild_verify)
        self.assertIs(request.draft_probs, probs)
        request.retire()

    def test_replay_does_not_sample_or_advance_rng(self):
        scheduler = self.scheduler()
        scheduler._sample_rows = []
        before = torch.random.get_rng_state().clone()
        tokens, probs = scheduler._sample(torch.ones(4, 32), 4)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(tuple(probs.shape), (4, 0))
        self.assertEqual(tokens.tolist(), [0] * 4)

    def test_waiting_replay_does_not_evict_older_prefill_when_space_is_tight(self):
        scheduler = self.scheduler(capacity=20, budget=16)
        older, younger = [Request([1] * 20, 16) for _ in range(2)]
        for identity, request in enumerate((older, younger)):
            request.request_id = identity
            request.attach_kv_pool(scheduler.kv_pool)
            request.kv_cache_ptr.allocate_tokens(12 if identity == 0 else 8)
            request.initial_idx = request.cur_spec_idx = 0
            scheduler._requests[identity] = request
        scheduler._preempt_requested[1] = None
        scheduler._apply_preemptions()
        scheduler._retired[0].control_done = SimpleNamespace(query=lambda: True)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [older])
        self.assertIn(1, scheduler._waiting_replays)
        self.assertEqual(scheduler._preempt_requested, {})
        scheduler._requests.pop(0)
        older.retire()
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [younger])
        self.assertNotIn(1, scheduler._waiting_replays)
        younger.retire()

    def test_decode_and_multiple_replays_share_the_residual_budget(self):
        scheduler = self.scheduler(capacity=64, budget=10)
        scheduler._args["chunk_size"] = 4
        decoding = CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        scheduler._requests[0] = decoding
        replays = []
        for identity in (1, 2):
            request = Request([1] * 20, 16)
            request.request_id = identity
            request.attach_kv_pool(scheduler.kv_pool)
            request.kv_cache_ptr.allocate_tokens(8)
            request.initial_idx = request.cur_spec_idx = 0
            scheduler._requests[identity] = request
            scheduler._preempt_requested[identity] = None
            replays.append(request)
        scheduler._apply_preemptions()
        for retired in scheduler._retired:
            retired.control_done = SimpleNamespace(query=lambda: True)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [decoding, *replays])
        self.assertEqual([r.scheduled_tokens for r in scheduler.active_requests], [1, 4, 4])
        self.assertEqual(scheduler._waiting_replays, {})
        self.assertEqual(scheduler.metrics["replay_admissions"], 2)
        # Admitted partial replays continue with running work next round.
        for request in replays:
            request.kv_cache_ptr.allocate_tokens(4)
        decoding.kv_cache_ptr.allocate_tokens(1)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [decoding, *replays])
        self.assertEqual(scheduler.metrics["replay_admissions"], 2)
        for request in scheduler.active_requests:
            request.retire()

    def test_round_that_preempts_running_work_does_not_admit_waiting_replay(self):
        scheduler = self.scheduler(capacity=32, budget=16)
        for identity, count in ((0, 18), (1, 14), (2, 0)):
            request = Request([1] * 20, 16)
            request.request_id = identity
            request.attach_kv_pool(scheduler.kv_pool)
            if count:
                request.kv_cache_ptr.allocate_tokens(count)
            request.initial_idx = request.cur_spec_idx = 0
            scheduler._requests[identity] = request
            if identity == 2:
                request.begin_replay(8)
                scheduler._waiting_replays[identity] = None
        scheduler.admit_requests([])
        self.assertEqual(set(scheduler._preempt_requested), {1})
        self.assertIn(2, scheduler._waiting_replays)
        self.assertEqual(scheduler.metrics["replay_admissions"], 0)
        self.assertEqual(scheduler.metrics["recovery_preemption_waits"], 1)
        for request in scheduler._requests.values():
            request.retire()

    def test_replay_admission_protects_known_prefix_not_only_its_first_chunk(self):
        scheduler = self.scheduler(capacity=40, budget=8)
        scheduler._args["chunk_size"] = 4
        older = Request([1] * 30, 16)
        older.request_id = 0
        older.attach_kv_pool(scheduler.kv_pool)
        older.kv_cache_ptr.allocate_tokens(20)
        younger = Request([1] * 20, 16)
        younger.request_id = 1
        younger.attach_kv_pool(scheduler.kv_pool)
        younger.begin_replay(12)
        scheduler._requests = {0: older, 1: younger}
        scheduler._waiting_replays[1] = None
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [older])
        self.assertIn(1, scheduler._waiting_replays)
        self.assertEqual(scheduler._preempt_requested, {})
        older.retire()
        younger.retire()

    def test_running_younger_replay_cannot_preempt_an_older_request(self):
        scheduler = self.scheduler(capacity=40, budget=4)
        scheduler.cpu_capacity = 1  # No cold copy fits; only preemption can reclaim.
        decoding = CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        older = CPUAsyncTransfers.request(scheduler.kv_pool, 1)
        older.status = Request.ReqExecType.VERIFY
        older.len_draft_probs = scheduler.spec_stride
        younger = Request([1] * 20, 16)
        younger.request_id = 2
        younger.attach_kv_pool(scheduler.kv_pool)
        younger.kv_cache_ptr.allocate_tokens(8)
        younger.begin_replay(16)
        scheduler._requests = {0: decoding, 1: older, 2: younger}
        # Decode spends one token. The older Verify needs four and waits for
        # next round; the younger replay must not evict it to use the last three.
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [decoding, younger])
        self.assertEqual([r.scheduled_tokens for r in scheduler.active_requests], [1, 3])
        self.assertEqual(scheduler._preempt_requested, {})
        decoding.retire()
        older.retire()
        younger.retire()

    def test_solo_resident_draft_can_use_real_free_page_below_workspace_reserve(self):
        scheduler = self.scheduler(capacity=13, budget=16)
        request = CPUAsyncTransfers.request(scheduler.kv_pool, 0, length=12)
        scheduler._requests[0] = request
        self.assertGreater(scheduler._runtime_reserve(), scheduler.kv_pool.num_free_pages)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [request])
        self.assertEqual(request.scheduled_tokens, 1)
        request.retire()

    def test_pressure_spill_is_selected_before_ready_batch_locks_its_candidates(self):
        from serve.scheduler.async_compact_scheduler import TransferJob
        scheduler = self.scheduler(capacity=44, budget=16)
        requests = [CPUAsyncTransfers.request(scheduler.kv_pool, i, length=12) for i in range(3)]
        scheduler._requests = {r.request_id: r for r in requests}

        def submit(request, direction):
            scheduler._transfers[request.request_id] = TransferJob(request, direction, None, 12, 7)

        with patch.object(scheduler, "_submit_transfer", side_effect=submit) as spill:
            scheduler.admit_requests([])
        self.assertEqual(spill.call_args.args[1], "repack")
        self.assertNotEqual(spill.call_args.args[0].request_id, 0)
        self.assertNotIn(spill.call_args.args[0], scheduler.active_requests)
        for request in requests:
            request.retire()

    def test_mixed_sampling_preserves_tp_rank_major_row_layout(self):
        scheduler = self.scheduler()
        scheduler._sample_rows = [1, 3]
        shards = torch.arange(24).reshape(2, 4, 3)
        expected = shards[:, [1, 3], :].contiguous().reshape(2, 6)
        sampled = torch.tensor([2, 4], dtype=torch.int32)
        distributions = torch.ones(2, 6)
        module = "serve.scheduler.async_cpu_compact_scheduler.AsyncCPUCompactSpecScheduler._sample"
        with patch(module, return_value=(sampled, distributions)) as sample:
            tokens, probs = scheduler._sample(shards.reshape(4, 6), 4)
        self.assertTrue(torch.equal(sample.call_args.args[0], expected))
        self.assertEqual(tokens.tolist(), [0, 2, 0, 4])
        self.assertEqual(probs.sum(dim=1).tolist(), [0, 6, 0, 6])

    def test_evicted_cpu_copy_is_missing_not_an_empty_restore(self):
        ptr = LocalPtr.__new__(LocalPtr)
        ptr._closed, ptr._pending, ptr.phase = True, None, "hot"
        ptr.cold_cpu = torch.empty(2, 7, 2, 1, 2, 128)
        ptr.cold_missing = False
        self.assertEqual(ptr.evict_cold(), 7)
        self.assertTrue(ptr.cold_missing)
        ptr.needs_pack = False
        with self.assertRaisesRegex(RuntimeError, "requires replay"):
            ptr.begin_restore()

    def test_cpu_lru_skips_inflight_copy(self):
        scheduler = self.scheduler()
        scheduler.cpu_capacity = 25
        for identity in (0, 1):
            ptr = LocalPtr.__new__(LocalPtr)
            ptr._closed, ptr._pending, ptr.phase = True, None, "hot"
            ptr.cold_cpu, ptr.cold_missing = torch.empty(2, 10, 2, 1, 2, 128), False
            scheduler._requests[identity] = SimpleNamespace(
                request_id=identity, status=Request.ReqExecType.DRAFT, kv_cache_ptr=ptr)
            scheduler._cold_lru[identity] = None
        scheduler._transfers[0] = object()
        self.assertTrue(scheduler._ensure_cpu_space(10))
        self.assertEqual(scheduler._requests[0].kv_cache_ptr.cold_count, 10)
        self.assertTrue(scheduler._requests[1].kv_cache_ptr.cold_missing)

    def test_early_verify_uses_retained_draft_count(self):
        scheduler = self.scheduler()
        scheduler.spec_stride, scheduler.early_verify_length = 8, 4
        request = Request([1], 16)
        request.status, request.len_draft_probs = Request.ReqExecType.DRAFT, 4
        scheduler._early_verify(request)
        self.assertEqual(request.status, Request.ReqExecType.VERIFY)
        self.assertEqual(scheduler.spec_stride + 1 - request.initial_idx, 5)
        self.assertEqual(request.cur_spec_idx - request.initial_idx, request.len_draft_probs)


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
        scheduler.debug_checks = False
        scheduler._last_served = {}
        scheduler._args = dict(chunk_size=256)
        scheduler.metrics = dict(admission_waits=0, capacity_waits=0, max_live_requests=0)
        scheduler.active_requests, scheduler.offload_requests = [], []
        return scheduler

    def test_transfer_timing_separates_event_wait_tp_exchange_and_commit(self):
        scheduler = self.scheduler()
        scheduler.group, scheduler.step_id = object(), 7
        clock = [0.]
        calls = []

        def wait():
            calls.append("wait")
            if len(calls) == 1:
                clock[0] += 3.  # Only the first event wait blocks on GPU work.

        transfer = SimpleNamespace(wait=wait)

        def begin():
            clock[0] += 2.
            return transfer

        def commit(result):
            self.assertIs(result, transfer)
            result.wait()  # Existing commit calls wait again on the same event.
            clock[0] += 1.

        def exchange(errors, error, group):
            clock[0] += 5.
            errors[:] = [None, None]

        module = "serve.scheduler.compact_scheduler"
        with patch(f"{module}.time.perf_counter", side_effect=lambda: clock[0]), \
             patch(f"{module}.dist.get_world_size", return_value=2), \
             patch(f"{module}.dist.all_gather_object", side_effect=exchange):
            self.assertIs(scheduler._timed_transfer("restore", begin, commit, None), transfer)
        report = scheduler.transfer_timing_snapshot(elapsed_seconds=20.)
        self.assertEqual(report["directions"]["restore"], dict(
            submit_wall_seconds=2., event_wait_wall_seconds=3.,
            tp_coordination_wall_seconds=5., commit_wall_seconds=1.))
        self.assertEqual(report["transfer_event_wait_fraction"], .15)
        self.assertEqual(report["transfer_section_wall_fraction"], .55)
        self.assertEqual(calls, ["wait", "wait"])
        self.assertEqual(report["directions"]["repack"]["event_wait_wall_seconds"], 0.)

    def test_transfer_timing_zero_interval_and_no_tp_exchange(self):
        scheduler = self.scheduler()
        scheduler.step_id, scheduler._run_start_perf_counter = 0, None
        transfer = SimpleNamespace(wait=lambda: None)
        module = "serve.scheduler.compact_scheduler"
        with patch(f"{module}.time.perf_counter", return_value=10.), \
             patch(f"{module}.dist.all_gather_object") as exchange:
            scheduler._timed_transfer("repack", lambda: transfer, lambda result: None, None)
        exchange.assert_not_called()
        report = scheduler.transfer_timing_snapshot()
        self.assertEqual(report["elapsed_seconds"], 0.)
        self.assertEqual(report["transfer_event_wait_fraction"], 0.)
        self.assertEqual(report["directions"]["repack"]["tp_coordination_wall_seconds"], 0.)

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

    def test_tp_sampling_uses_native_local_results_without_communication(self):
        module = "serve.scheduler.compact_scheduler"
        for temperature, sampler_dtype in ((0., torch.int64), (.65, torch.int32)):
            for rank in (0, 1):
                scheduler = self.scheduler()
                scheduler.group, scheduler.rank = object(), rank
                scheduler.temperature, scheduler.sampling_metadata = temperature, None
                local_probs = torch.zeros(2, 5)
                local_probs[0, 2], local_probs[1, 3] = .25, .75
                sampled = []

                def sampler(*args):
                    sampled.append(args)
                    return torch.tensor([rank, rank], dtype=sampler_dtype), local_probs

                scheduler.sampler = sampler
                logits = torch.zeros(2, 5)
                with patch(f"{module}.coordinated", side_effect=AssertionError("sampling coordination")), \
                     patch(f"{module}.dist.broadcast", side_effect=AssertionError("sampling broadcast")):
                    tokens, probs = scheduler._sample(logits, 2)
                self.assertEqual(tokens.dtype, torch.int32)
                self.assertEqual(tokens.tolist(), [rank, rank])
                self.assertEqual(len(sampled), 1)
                self.assertIs(sampled[0][0], logits)
                self.assertEqual(sampled[0][2], 2)
                self.assertIs(probs, local_probs)
                self.assertEqual(probs[0, 2].item(), .25)
                self.assertEqual(probs[1, 3].item(), .75)

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


class CPUDiagnostics(unittest.TestCase):
    def test_release_stages_do_not_exchange_errors_and_propagate_local_failure(self):
        scheduler = CPUCapacity.scheduler()
        scheduler.group = object()
        with patch("serve.scheduler.compact_scheduler.coordinated",
                   side_effect=AssertionError("unexpected stage collective")):
            self.assertEqual(scheduler._run_stage("metadata", lambda: 42), 42)
            def fail():
                raise ValueError("local metadata failure")
            with self.assertRaisesRegex(ValueError, "local metadata failure"):
                scheduler._run_stage("metadata", fail)

    def test_diagnostic_stage_detects_remote_error(self):
        scheduler = CPUCapacity.scheduler()
        scheduler.group, scheduler.debug_checks = object(), True
        def exchange(errors, local, group):
            self.assertIsNone(local)
            errors[:] = [None, "ValueError: remote metadata failure"]
        module = "serve.scheduler.compact_scheduler"
        with patch(f"{module}.dist.get_world_size", return_value=2), \
             patch(f"{module}.dist.all_gather_object", side_effect=exchange):
            with self.assertRaisesRegex(RuntimeError, "remote metadata failure"):
                scheduler._run_stage("metadata", lambda: 42)

    def test_ready_batch_check_is_optional_and_detects_divergence(self):
        scheduler = CPUCapacity.scheduler()
        scheduler.group = object()
        with patch.object(scheduler, "_broadcast", side_effect=AssertionError("unexpected broadcast")):
            scheduler._check_ready_batch([0, 1])
        scheduler.debug_checks = True
        with patch.object(scheduler, "_broadcast", return_value=[1, 0]):
            with self.assertRaisesRegex(RuntimeError, "TP ready batches diverged"):
                scheduler._check_ready_batch([0, 1])

    def test_tp_failure_prints_traceback_and_exits_without_cuda_cleanup(self):
        import io
        scheduler = CPUCapacity.scheduler()
        scheduler.group, scheduler.rank = object(), 1
        error_log = io.StringIO()
        module = "serve.scheduler.compact_scheduler"
        with patch(f"{module}.sys.stderr", error_log), \
             patch(f"{module}.os._exit", side_effect=SystemExit(1)) as abort, \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CUDA wait during failure")):
            try:
                raise ValueError("injected worker failure")
            except ValueError:
                with self.assertRaises(SystemExit):
                    scheduler._abort_failed_tp_worker()
        abort.assert_called_once_with(1)
        self.assertIn("TP rank 1", error_log.getvalue())
        self.assertIn("ValueError: injected worker failure", error_log.getvalue())
        scheduler.group = None
        with patch(f"{module}.os._exit") as abort:
            scheduler._abort_failed_tp_worker()
        abort.assert_not_called()


class CPUAsyncTransfers(unittest.TestCase):
    @staticmethod
    def scheduler():
        base = CPUCapacity.scheduler()
        from serve.scheduler.async_compact_scheduler import AsyncCompactSpecScheduler
        scheduler = AsyncCompactSpecScheduler.__new__(AsyncCompactSpecScheduler)
        scheduler.__dict__.update(base.__dict__)
        scheduler.spec_stride, scheduler.step_id, scheduler.page_bytes = 3, 0, 1
        scheduler.spec_cache_args = dict(budget_ratio=.1, num_min_budget=5)
        scheduler._requests, scheduler._transfers = {}, {}
        scheduler.metrics.update(restores=0, migrations=0, h2d_bytes=0, d2h_bytes=0,
                                 freed_source_pages=0, transfer_poll_tp_wall_seconds=0.,
                                 max_live_requests=0, transfer_idle_waits=0)
        return scheduler

    @staticmethod
    def request(pool, identity, length=12):
        pages = [pool.alloc_page() for _ in range(length)]
        full = FullCachePtr(pool, array.array("i", pages), length)
        ptr, _ = LocalPtr.from_kv_cache(full)
        CPUOwnership().select(ptr)
        ptr.keep_full_resident()
        request = Request(list(range(length + 1)), 8)
        request.request_id, request.status = identity, Request.ReqExecType.DRAFT
        request.initial_idx, request.cur_spec_idx = 0, 1
        request.kv_cache_ptr = ptr
        return request

    def test_low_pressure_draft_keeps_full_resident_without_any_transfer(self):
        scheduler = self.scheduler()
        pool = Pool(capacity=64)
        request = self.request(pool, 0)
        scheduler.kv_pool, scheduler._requests = pool, {0: request}
        pages = request.kv_cache_ptr.owned_pages
        with patch.object(scheduler, "_submit_transfer") as submit:
            scheduler.admit_requests([])
        submit.assert_not_called()
        self.assertEqual(scheduler.active_requests, [request])
        self.assertEqual(request.kv_cache_ptr.owned_pages, pages)
        request.retire()

    def test_unfinished_restore_waits_while_other_request_is_ready(self):
        from serve.scheduler.async_compact_scheduler import TransferJob
        scheduler = self.scheduler()
        pool = Pool(capacity=64)
        restoring, ready = [self.request(pool, identity) for identity in (0, 1)]
        restoring.status = Request.ReqExecType.VERIFY
        restoring.kv_cache_ptr.phase = "hot"
        transfer = SimpleNamespace(ready=SimpleNamespace(query=lambda: False),
                                   wait=lambda: self.fail("Scheduling blocked on a pending event"))
        scheduler.kv_pool = pool
        scheduler._requests = {r.request_id: r for r in (restoring, ready)}
        scheduler._transfers = {0: TransferJob(restoring, "restore", transfer, 5, 7)}
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [ready])
        self.assertEqual(scheduler.offload_requests, [restoring])
        self.assertIn(0, scheduler._transfers)
        restoring.retire()
        ready.retire()

    def test_tp_local_completion_does_not_commit_until_every_rank_is_ready(self):
        from serve.scheduler.async_compact_scheduler import TransferJob
        scheduler = self.scheduler()
        scheduler.group = object()
        commits = []
        ptr = SimpleNamespace(commit_restore=lambda *args, **kwargs: commits.append(kwargs))
        request = SimpleNamespace(kv_cache_ptr=ptr)
        transfer = SimpleNamespace(ready=SimpleNamespace(query=lambda: True))
        scheduler._transfers = {0: TransferJob(request, "restore", transfer, 5, 7)}
        remote_ready = [False]

        def exchange(observations, local, group):
            observations[:] = [local, [(0, "restore", remote_ready[0])]]

        module = "serve.scheduler.async_compact_scheduler"
        with patch(f"{module}.dist.get_world_size", return_value=2), \
             patch(f"{module}.dist.all_gather_object", side_effect=exchange):
            scheduler._poll_transfers()
            self.assertEqual(commits, [])
            self.assertIn(0, scheduler._transfers)
            remote_ready[0] = True
            scheduler._poll_transfers()
        self.assertEqual(commits, [dict(nonblocking=True)])
        self.assertEqual(scheduler._transfers, {})
        self.assertEqual(scheduler.metrics["h2d_bytes"], 7)

    def test_empty_ready_batch_waits_and_commits_without_forward(self):
        from serve.scheduler.async_compact_scheduler import TransferJob
        scheduler = self.scheduler()
        ready, commits, waits = [False], [], []

        def wait():
            waits.append(True)
            ready[0] = True

        ptr = SimpleNamespace(commit_restore=lambda *args, **kwargs: commits.append(kwargs))
        request = SimpleNamespace(kv_cache_ptr=ptr)
        transfer = SimpleNamespace(wait=wait, ready=SimpleNamespace(query=lambda: ready[0]))
        scheduler._transfers = {0: TransferJob(request, "restore", transfer, 5, 7)}
        scheduler.wait_for_ready_requests()
        self.assertEqual(waits, [True])
        self.assertEqual(commits, [dict(nonblocking=True)])
        self.assertEqual(scheduler._transfers, {})
        self.assertEqual(scheduler.metrics["transfer_idle_waits"], 1)
        self.assertGreaterEqual(scheduler.metrics["restore_event_wait_wall_seconds"], 0.)

    def test_offload_locks_source_pages_and_other_drafts_can_use_future_reserve(self):
        from serve.scheduler.async_compact_scheduler import TransferJob
        scheduler = self.scheduler()
        pool = Pool(capacity=48)
        requests = [self.request(pool, identity) for identity in range(3)]
        for request in requests:
            request.kv_cache_ptr.allocate_tokens(1)
        scheduler.kv_pool = pool
        scheduler._requests = {r.request_id: r for r in requests}
        original = requests[0].kv_cache_ptr.owned_pages

        def submit(request, direction):
            self.assertEqual(direction, "repack")
            ptr = request.kv_cache_ptr
            ptr.prepare_resident_offload()
            destinations = ptr._allocate(ptr.len_loading_indices)
            transfer = SimpleNamespace(pages=destinations, ready=SimpleNamespace(query=lambda: False))
            scheduler._transfers[request.request_id] = TransferJob(
                request, direction, transfer, len(ptr.indices), len(ptr.indices)-len(destinations))

        with patch.object(scheduler, "_submit_transfer", side_effect=submit):
            scheduler.admit_requests([])
        self.assertEqual([r.request_id for r in scheduler.active_requests], [1, 2])
        self.assertTrue(original <= requests[0].kv_cache_ptr.owned_pages)
        self.assertEqual(pool.num_free_pages, 3)
        for request in requests:
            request.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)


class CPUAsyncPipeline(unittest.TestCase):
    @staticmethod
    def scheduler():
        base = CPUAsyncTransfers.scheduler()
        from serve.scheduler.async_cpu_compact_scheduler import AsyncCPUCompactSpecScheduler
        scheduler = AsyncCPUCompactSpecScheduler.__new__(AsyncCPUCompactSpecScheduler)
        scheduler.__dict__.update(base.__dict__)
        scheduler.async_on = True
        scheduler.kv_pool.device = "cpu"
        scheduler._retired, scheduler._output_keepalive = [], []
        scheduler._compute_fences, scheduler._patched_requests = {}, []
        scheduler.metrics.update(deferred_retirements=0, max_deferred_retirements=0,
                                 cpu_result_readback_wall_seconds=0.,
                                 cpu_result_update_wall_seconds=0., cpu_updates_with_forward_inflight=0,
                                 retirement_idle_wait_seconds=0.)
        return scheduler

    def test_invalid_later_request_leaves_all_original_retirement_methods_intact(self):
        scheduler = self.scheduler()
        requests = [Request([1], 8), Request([1], 0)]
        with self.assertRaises(ValueError):
            scheduler.run(requests)
        self.assertTrue(all("retire" not in r.__dict__ for r in requests))
        self.assertEqual(scheduler._patched_requests, [])

    def test_retirement_keeps_pages_and_probs_until_compute_and_control_complete(self):
        scheduler = self.scheduler()
        pool = Pool(capacity=64)
        request = CPUAsyncTransfers.request(pool, 0)
        request.draft_probs = torch.ones(2, 3)
        compute, control = [False], [False]
        scheduler._compute_fences[0] = SimpleNamespace(query=lambda: compute[0])
        ptr, probs = request.kv_cache_ptr, request.draft_probs
        scheduler._defer_retire(request)
        scheduler._retired[0].control_done = SimpleNamespace(query=lambda: control[0])
        self.assertIsNone(request.kv_cache_ptr)
        self.assertIs(scheduler._retired[0].probs, probs)
        scheduler._poll_retired()
        self.assertEqual(pool.num_free_pages, 52)
        compute[0] = True
        scheduler._poll_retired()
        self.assertEqual(pool.num_free_pages, 52)
        control[0] = True
        scheduler._poll_retired()
        self.assertEqual(pool.num_free_pages, pool.capacity)
        self.assertEqual(scheduler._retired, [])
        self.assertTrue(ptr._closed)

    def test_tp_retirement_waits_for_remote_rank(self):
        scheduler = self.scheduler()
        pool = Pool(capacity=64)
        request = CPUAsyncTransfers.request(pool, 0)
        scheduler._defer_retire(request)
        scheduler._retired[0].control_done = SimpleNamespace(query=lambda: True)
        scheduler.group = object()
        remote = [False]

        def exchange(outputs, local, group):
            outputs[:] = [local, remote[:]]

        with patch("serve.scheduler.async_cpu_compact_scheduler.dist.get_world_size", return_value=2), \
             patch("serve.scheduler.async_cpu_compact_scheduler.dist.all_gather_object", side_effect=exchange):
            scheduler._poll_retired()
            self.assertEqual(pool.num_free_pages, 52)
            remote[0] = True
            scheduler._poll_retired()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_page_pressure_waits_for_retirement_instead_of_reporting_oom(self):
        scheduler = self.scheduler()
        pool = Pool(capacity=24)
        retired, live = [CPUAsyncTransfers.request(pool, identity) for identity in range(2)]
        compute, control = [False], [False]
        scheduler._compute_fences[0] = SimpleNamespace(
            query=lambda: compute[0], synchronize=lambda: compute.__setitem__(0, True))
        scheduler._defer_retire(retired)
        scheduler._retired[0].control_done = SimpleNamespace(
            query=lambda: control[0], synchronize=lambda: control.__setitem__(0, True))
        scheduler.kv_pool, scheduler._requests = pool, {1: live}
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [])
        self.assertEqual(pool.num_free_pages, 0)
        scheduler.wait_for_ready_requests()
        self.assertEqual(pool.num_free_pages, 12)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [live])
        live.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_all_stall_batch_has_zero_queries_but_current_selection_page_table(self):
        scheduler = self.scheduler()
        pool = Pool(capacity=64)
        requests = [CPUAsyncTransfers.request(pool, identity) for identity in range(2)]
        for index, request in enumerate(requests):
            request.status = Request.ReqExecType.STALL
            request.prev_idx = 1 - index
        scheduler.kv_pool = pool
        scheduler._requests = {r.request_id: r for r in requests}
        scheduler.admit_requests([])
        self.assertEqual([r.request_id for r in scheduler.active_requests], [1, 0])
        host, tensors = scheduler.prepare_forward_inputs()
        self.assertIsNone(host)
        self.assertIsNone(tensors)
        self.assertEqual(list(scheduler.spec_cache_args["all_kv_indptr_cpu"]), [0, 12, 24])
        expected = list(requests[1].kv_cache_ptr.indices) + list(requests[0].kv_cache_ptr.indices)
        self.assertEqual(scheduler.spec_cache_args["all_kv_indices_gpu"].tolist(), expected)
        for request in requests:
            request.retire()

    def test_result_consumption_waits_previous_event_without_waiting_current_forward(self):
        from serve.scheduler.async_cpu_compact_scheduler import ForwardOutput
        scheduler = self.scheduler()
        calls = []
        consumed = SimpleNamespace(query=lambda: True)
        control = SimpleNamespace(wait_event=lambda event: calls.append("previous dependency"),
                                  record_event=lambda: consumed)
        tensor = SimpleNamespace(record_stream=lambda stream: calls.append("retain on control"))
        previous = ForwardOutput({}, {}, tensor, tensor, object())
        scheduler.update_metadata_host = lambda *args: calls.append("previous readback")
        scheduler.update_requests = lambda *args: calls.append("previous result") or []
        with patch("torch.cuda.current_stream", return_value=control):
            completed, event = scheduler._consume_previous(
                previous, dict(token_indptr=[0, 1]), {}, SimpleNamespace(query=lambda: False))
        self.assertEqual(calls, ["previous dependency", "retain on control", "retain on control",
                                 "previous readback", "previous result"])
        self.assertEqual(completed, [])
        self.assertIs(event, consumed)
        self.assertEqual(scheduler.metrics["cpu_updates_with_forward_inflight"], 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GPUContent(CPUOwnership):
    def test_resident_to_offloaded_to_restored_preserves_selected_content(self):
        pool, ptr = self.new_ptr("cuda:0")
        self.select(ptr)
        ptr.keep_full_resident()
        for _ in range(3):
            ptr.allocate_tokens(1)
        ptr.prepare_resident_offload()
        plan = partition_pages(torch.tensor(ptr.indices, device=pool.device, dtype=torch.int32),
                               ptr.selected_indices_buffer_gpu, ptr.len_selected_indices_gpu,
                               pool_capacity=pool.capacity, recent_count=ptr.recent_count, validate=True)
        hot = selected_copy(pool._physical_mem_gpu, plan.hot_pages).clone()
        cold = selected_copy(pool._physical_mem_gpu, plan.cold_pages).clone()
        transfer = ptr.begin_pack(validate=True)
        transfer.wait()
        ptr.commit_pack(transfer, nonblocking=True)
        self.assertEqual(len(ptr.tail_pages), 3)
        self.assertEqual(ptr.recent_count, 2)
        torch.testing.assert_close(pool._physical_mem_gpu[:, list(ptr.indices)], hot, rtol=0, atol=0)
        torch.testing.assert_close(ptr.cold_cpu, cold.cpu(), rtol=0, atol=0)
        transfer = ptr.begin_restore()
        transfer.wait()
        ptr.commit_restore(transfer, nonblocking=True)
        selected = ptr.len_selected_indices_gpu
        expected = torch.cat([hot[:, :selected], cold, hot[:, selected:]], dim=1)
        torch.testing.assert_close(pool._physical_mem_gpu[:, list(ptr.indices)], expected, rtol=0, atol=0)
        # A rejection can cross the migration boundary: three tail slots were
        # moved before offload, then two more were appended after restoration.
        for _ in range(2):
            ptr.allocate_tokens(1)
        rejected = set(ptr.tail_pages[-4:])
        ptr.freeze_verify_view()
        ptr.retract_tokens(4)
        self.assertEqual(len(ptr.tail_pages), 1)
        self.assertTrue(rejected.isdisjoint(ptr.owned_pages))
        self.select(ptr)
        ptr.keep_full_resident()
        self.assertIsNone(ptr.cold_cpu)
        self.assertEqual(len(ptr.indices), ptr.len_kv_cache)
        ptr.close()
        self.assertEqual(pool.num_free_pages, pool.capacity)

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
