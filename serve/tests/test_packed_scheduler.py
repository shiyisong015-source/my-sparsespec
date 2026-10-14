"""Check verify planning against explicit arena capacity before any restore."""

import unittest
import itertools
import math
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.kv_partition import PackedKVRegion
from serve.scheduler.packed_scheduler import PackedSpecScheduler
from serve.request.request import Request


class MixedVerifySchedulingTest(unittest.TestCase):
    def test_capacity_deferred_ready_request_stays_parked_while_others_compute(self):
        scheduler = PackedSpecScheduler.__new__(PackedSpecScheduler)
        scheduler.k, scheduler.dense_reference, scheduler.rank = 4, False, 0
        scheduler.metrics = dict(verify_model_forwards=0, max_verify_batch=0, verify_batch_histogram={})
        states = [SimpleNamespace(
            request=SimpleNamespace(request_id=i, len_output=20, len_input=4),
            region=SimpleNamespace(length=2), output=[9] if i < 3 else [],
            drafts=[1] * n, draft_probs=[]) for i, n in enumerate((4, 4, 2, 0))]
        frames = [SimpleNamespace(_finished=True, _aborted=False) for _ in states]
        scheduler._prepare_verify = Mock(return_value=(frames[0], None, [9] + states[0].drafts))
        scheduler._nonverify_entries = Mock(return_value=[
            (states[2], frames[2], [1], "draft"), (states[3], frames[3], [2, 3], "prefill")])
        probs = torch.arange(24).reshape(8, 3)
        scheduler._forward = Mock(return_value=(list(range(100, 108)), probs))
        scheduler._commit_verify = Mock()
        scheduler._verify_many([states[0]], states, [])
        self.assertEqual([id(s) for s in scheduler._nonverify_entries.call_args.args[0]],
                         [id(states[2]), id(states[3])])
        entries = scheduler._forward.call_args.args[0]
        self.assertEqual([(s.request.request_id, phase) for s, _, _, phase in entries],
                         [(0, "verify"), (2, "draft"), (3, "prefill")])
        self.assertEqual(states[1].drafts, [1] * 4)
        self.assertEqual(states[2].drafts, [1, 1, 105])
        torch.testing.assert_close(states[2].draft_probs[0], probs[5:6])
        self.assertEqual(scheduler._commit_verify.call_args.args[-2], list(range(100, 105)))
        torch.testing.assert_close(scheduler._commit_verify.call_args.args[-1], probs[:5])


class AdmissionPeakTest(unittest.TestCase):
    def scheduler(self):
        scheduler = PackedSpecScheduler.__new__(PackedSpecScheduler)
        scheduler.k, scheduler.r, scheduler.minimum, scheduler.token_bytes = 4, .1, 128, 256
        scheduler.dense_reference = scheduler.retain_full_reference = False
        return scheduler

    def test_long_request_fits_initial_and_future_peak_without_two_full_histories(self):
        scheduler = self.scheduler()
        request = SimpleNamespace(len_input=20000, len_output=512)
        scheduler.gpu_pool = SimpleNamespace(capacity_bytes=24000 * 256, alignment=256)
        scheduler.cpu_pool = SimpleNamespace(capacity_bytes=40000 * 256)
        self.assertTrue(scheduler._fits([], request))
        base, extra, _, _ = scheduler._estimate(request, packed=False)
        packed_base, packed_extra, _, _ = scheduler._estimate(request, packed=True)
        full_allocation = (request.len_input + scheduler.k + 1) * 256
        self.assertGreaterEqual(base + extra, full_allocation + packed_base)
        self.assertGreaterEqual(base + extra, packed_base + packed_extra)
        self.assertLess(base + extra, 2 * request.len_input * 256)

    def test_all_full_to_packed_transitions_preserve_one_verify_headroom(self):
        scheduler = self.scheduler()
        requests = [SimpleNamespace(len_input=n, len_output=512) for n in (20000, 8000, 128, 29)]
        estimates = [scheduler._estimate(r, packed=False) for r in requests]
        reserved = sum(e[0] for e in estimates) + max(e[1] for e in estimates)
        # Exhaust every subset of requests that has completed its first pack.
        # A single remaining request's first pack or later restore/repack must
        # fit under the original reservation, including final history growth.
        for packed in itertools.product((False, True), repeat=len(requests)):
            current = [scheduler._estimate(r, packed=p) for r, p in zip(requests, packed)]
            self.assertLessEqual(sum(e[0] for e in current) + max(e[1] for e in current), reserved)
            actual_bases, migration_peaks = [], []
            for request, is_packed in zip(requests, packed):
                history = request.len_input + request.len_output - 1
                hot = min(history, max(128, math.ceil(history * .1), 6))
                cold = history - min(history, max(128, math.ceil(history * .1), 2))
                packed_allocation = hot + 5
                actual_bases.append((packed_allocation if is_packed else request.len_input + 5) * 256)
                migration_peaks.append((packed_allocation + (cold if is_packed else 0)) * 256)
            self.assertLessEqual(sum(actual_bases) + max(migration_peaks), reserved)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA arenas require an available GPU")
class VerifyCapacityTest(unittest.TestCase):
    def test_real_kv_pipeline_under_pressure_and_growing_cold_history(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros((), device="cuda", dtype=torch.bfloat16))
                self.config = SimpleNamespace(num_layers=2, num_kv_heads=2, head_dim=128,
                    vocab_size=128, scaling_factor=1., rope_base=10000.,
                    high_freq_factor=None, low_freq_factor=None)

            def attach_kv_pool(self, pool):
                self.pool = pool

            def forward(self, inputs):
                for layer in range(2):
                    q = torch.zeros((len(inputs), 8, 128), device=inputs.device, dtype=self.weight.dtype)
                    k = torch.zeros((len(inputs), 2, 128), device=inputs.device, dtype=self.weight.dtype)
                    q, k = self.pool.apply_batch_rope(q, k, self.config)
                    self.pool.append_batch_kv_cache(k, k, layer)
                    self.pool.batch_attn_fwd(q, layer)
                logits = torch.zeros((len(inputs), 128), device=inputs.device, dtype=self.weight.dtype)
                logits[:, 1] = 20
                return logits

        model, token_bytes = TinyModel(), 2048
        for prompt_length, output_length, requests, gpu_tokens, cpu_tokens in (
                (250, 24, 4, 1000, 2048), (20000, 512, 1, 24000, 40000)):
            gpu = KVBufferPool(gpu_tokens * token_bytes, device="cuda")
            cpu = KVBufferPool(cpu_tokens * token_bytes)
            try:
                scheduler = PackedSpecScheduler(model, gpu, cpu, chunk_size=512,
                    num_min_budget=16, max_batch_size=4)
                with torch.inference_mode():
                    completed = scheduler.run([Request([1] * prompt_length, output_length) for _ in range(requests)])
                self.assertEqual([len(r.token_ids) - r.len_input for r in completed], [output_length] * requests)
                self.assertGreater(scheduler.metrics["migrations"], 3)
                self.assertEqual(gpu.stats()["in_use_bytes"], 0)
                self.assertEqual(cpu.stats()["in_use_bytes"], 0)
                if requests == 4:
                    self.assertGreater(scheduler.metrics["admission_waits"], 0)
                    self.assertEqual(scheduler.metrics["max_actual_batch"], 4)
            finally:
                gpu.close(wait=True); cpu.close(wait=True)

    def fixture(self, free_gpu_tokens, free_cpu_tokens):
        # Four current histories each own 16 hot + 5 tail slots on GPU and
        # 84 cold slots on CPU. A verify needs 84 restored slots plus 21 new
        # hot/tail slots and 89 new CPU slots. Use 256 bytes per token so each
        # capacity is exact after arena alignment.
        gpu = KVBufferPool((84 + free_gpu_tokens) * 256, device="cuda")
        cpu = KVBufferPool((336 + free_cpu_tokens) * 256)
        leases, states = [], []
        for i in range(4):
            leases.extend((gpu.allocate((21 * 256,), torch.uint8),
                           cpu.allocate((84 * 256,), torch.uint8)))
            region = PackedKVRegion()
            region.hot_count, region.cold_count = 16, 84
            states.append(SimpleNamespace(region=region, round_start=100, drafts=[1] * 4,
                                          output=[0], request=SimpleNamespace(request_id=i, len_input=100, len_output=6)))
        scheduler = PackedSpecScheduler.__new__(PackedSpecScheduler)
        scheduler.gpu_pool, scheduler.cpu_pool = gpu, cpu
        scheduler.token_bytes, scheduler.k, scheduler.minimum, scheduler.r = 256, 4, 16, .1
        scheduler.group, scheduler.rank, scheduler.retain_full_reference = None, 0, False
        scheduler.dense_reference = False
        scheduler._log = lambda *args, **kwargs: None
        return scheduler, states, leases

    def check(self, gpu_free, cpu_free, expected):
        scheduler, states, leases = self.fixture(gpu_free, cpu_free)
        try:
            before = (scheduler.gpu_pool.stats()["in_use_bytes"], scheduler.cpu_pool.stats()["in_use_bytes"])
            if expected:
                selected = scheduler._verify_group(states)
                self.assertEqual([s.request.request_id for s in selected], list(range(expected)))
            else:
                with self.assertRaisesRegex(RuntimeError, "No verify request fits"):
                    scheduler._verify_group(states)
            self.assertEqual(before, (scheduler.gpu_pool.stats()["in_use_bytes"],
                                      scheduler.cpu_pool.stats()["in_use_bytes"]))
        finally:
            for lease in leases:
                lease.release()
            scheduler.gpu_pool.close(wait=True)
            scheduler.cpu_pool.close(wait=True)

    def test_gpu_headroom_limits_group_to_two_without_allocations(self):
        self.check(210, 356, 2)

    def test_cpu_double_backing_limits_group_to_one(self):
        self.check(210, 89, 1)

    def test_infeasible_configuration_fails_before_restore(self):
        self.check(104, 356, 0)

    def test_final_round_needs_restore_but_no_replacement(self):
        scheduler, states, leases = self.fixture(84, 0)
        states[0].request.len_output = 2
        states[0].drafts = []
        try:
            self.assertEqual([s.request.request_id for s in scheduler._verify_group(states)], [0])
        finally:
            for lease in leases:
                lease.release()
            scheduler.gpu_pool.close(wait=True)
            scheduler.cpu_pool.close(wait=True)


if __name__ == "__main__":
    unittest.main()
