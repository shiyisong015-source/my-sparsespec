"""Regression checks for partial/all completion of a verify batch."""
import unittest
from unittest.mock import patch
import torch

from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request


class SelectionCompletionTest(unittest.TestCase):
    def test_partial_completion_uses_surviving_task_count(self):
        calls = []

        class TopK:
            def run(self, **kwargs):
                calls.append(kwargs)
                return None, torch.zeros((1, 1, 6144), dtype=torch.int32)

        class Kernel:
            def __getitem__(self, grid):
                def launch(*args):
                    calls.append((grid, args))
                return launch

        # Previous batch had two VERIFY requests. Only the second survives.
        metadata = {'idx': 0, 'prev_idx': 1, 'ptr': 1234,
                    'num_selected_tokens': 3, 'num_effective_kv_len': 6}
        inputs = {'request_type': [Request.ReqExecType.VERIFY.value] * 2,
                  'verify_kv_indptr': [0, 8, 16], 'dump_logits_len': 16,
                  'dump_logits': torch.zeros(1)}
        current = {'all_kv_indptr_cpu': [0, 8],
                   'all_kv_indices_gpu': torch.arange(8, dtype=torch.int32)}
        with patch.object(PillarCachePtr, 'update_selected_indices_kernel', Kernel()):
            PillarCachePtr.dispatch_update_selected_indices(
                wrappers=[None, TopK()], selected_indices_metadatas=[metadata],
                spec_cache_args=current, input_metadata=inputs, device='cpu')
        self.assertEqual(calls[0]['num_requests'], 1)
        self.assertEqual(calls[0]['ks'].tolist(), [3])
        self.assertEqual(calls[0]['in_lens'].tolist(), [6])
        self.assertEqual(calls[0]['in_val_start_pos'].tolist(), [8])
        self.assertEqual(calls[0]['in_idx_start_pos'].tolist(), [0])
        self.assertEqual(calls[1][0][1], 1)
        self.assertEqual(calls[1][1][2].tolist(), [1234])

    def test_all_verify_requests_completed_with_normal_request_remaining(self):
        class TopK:
            def run(self, **kwargs):
                raise AssertionError('No surviving VERIFY request should launch Top-K')

        PillarCachePtr.dispatch_update_selected_indices(
            wrappers=[None, TopK()],
            selected_indices_metadatas=[{'idx': 0, 'prev_idx': -1,
                                        'num_selected_tokens': 0}],
            spec_cache_args={'all_kv_indptr_cpu': [0, 4],
                             'all_kv_indices_gpu': torch.arange(4)},
            input_metadata={'request_type': [Request.ReqExecType.VERIFY.value],
                            'verify_kv_indptr': [0, 8], 'dump_logits_len': 8,
                            'dump_logits': torch.zeros(1)}, device='cpu')


if __name__ == '__main__':
    unittest.main()
