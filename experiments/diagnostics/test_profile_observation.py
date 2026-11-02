"""Accounting and early-stop regressions without model execution or CUDA."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

with patch('torch.cuda.Stream', return_value=SimpleNamespace()):
    from component_profile import ObservationMixin, ProfileWindowsComplete, Request


class NativeStub:
    def verify_outputs(self, request, output_token_ids, **kwargs):
        return {'native_result': True}

    def is_request_complete(self, request, check_idx=-1):
        if self.trim:
            del request.token_ids[-self.trim:]
            return True
        return False

    def run(self, pending):
        self._run_start_perf_counter = time.perf_counter()
        raise ProfileWindowsComplete()

    def transfer_timing_snapshot(self, elapsed_seconds=None):
        self.metrics['runtime_peak_reserved_bytes'] = 123
        return {}


class Worker(ObservationMixin, NativeStub):
    pass


class Tokens:
    def __init__(self):
        self.token_ids = list(range(10))
        self.request_id = 0
        self.status = Request.ReqExecType.VERIFY
        self.len_kv_cache = 50
        self.len_oracle_ctx = 100
        self.kv_cache_ptr = SimpleNamespace(len_loading_indices=5)

    @property
    def len_token_ids(self):
        return len(self.token_ids)


class ObservationTests(unittest.TestCase):
    def worker(self):
        worker = Worker.__new__(Worker)
        worker.metrics = {'profile_verified_output_tokens': 0}
        worker.acceptance_log = None
        worker.draft_start = {}
        worker.partial_acceptance_rounds_skipped = 0
        worker.trim = 0
        return worker

    def test_carry_in_rounds_count_outputs_even_when_acceptance_log_is_skipped(self):
        worker = self.worker()
        log = object()
        worker.acceptance_log = log
        request = Tokens()
        result = worker.verify_outputs(request, [11, 12, 13])
        self.assertEqual(result, {'native_result': True})
        self.assertEqual(worker.metrics['profile_verified_output_tokens'], 3)
        self.assertEqual(worker.partial_acceptance_rounds_skipped, 1)
        self.assertIs(worker.acceptance_log, log)
        worker.verify_outputs(request, [14, 15])
        self.assertEqual(worker.metrics['profile_verified_output_tokens'], 5)
        self.assertEqual(worker.partial_acceptance_rounds_skipped, 1)

    def test_eos_trim_is_removed_from_verified_window_output(self):
        worker = self.worker()
        request = Tokens()
        worker.verify_outputs(request, [11, 12, 13, 14])
        worker.trim = 2
        self.assertTrue(worker.is_request_complete(request))
        self.assertEqual(worker.metrics['profile_verified_output_tokens'], 2)
        self.assertEqual(request.profile_stop_reason, 'eos')

    def prepare_stop(self, worker, directory):
        worker.capture_active = False
        worker.kv_pool = SimpleNamespace(num_free_pages=8, capacity=8)
        worker.idle_recorder = SimpleNamespace(write=lambda: None)
        worker.window_contexts = []
        worker.observation = {'record_windows': 3}
        worker.case_output = Path(directory)
        worker.rank = 0

    def test_windows_only_returns_only_native_completions_and_preserves_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker()
            self.prepare_stop(worker, directory)
            finished, unfinished = Tokens(), Tokens()
            finished.profile_stop_reason = 'eos'
            result = worker.run([finished, unfinished])
            self.assertEqual(result, [finished])
            self.assertFalse(hasattr(unfinished, 'profile_stop_reason'))
            self.assertTrue(worker.metrics['profile_windows_only'])
            self.assertEqual(worker.metrics['runtime_peak_reserved_bytes'], 123)
            self.assertEqual(worker.metrics['gpu_final_free_pages'], 8)

    def test_windows_only_detects_unreleased_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker()
            self.prepare_stop(worker, directory)
            worker.kv_pool.num_free_pages = 7
            with self.assertRaisesRegex(RuntimeError, 'leaked GPU pages'):
                worker.run([Tokens()])


if __name__ == '__main__':
    unittest.main()
