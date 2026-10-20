"""Exercise expanded selection storage, TopK workspace, and per-head strides."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from serve.attention.backend import BatchTopKWrapper
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request


class SelectedCapacityAdmission(unittest.TestCase):
    def test_sync_and_async_cpu_validate_the_configured_capacity(self):
        from serve.request.kv_cache_ptr.compact import CompactCachePtr
        from serve.scheduler.compact_scheduler import CompactSpecScheduler
        from serve.scheduler.async_cpu_compact_scheduler import AsyncCPUCompactSpecScheduler

        class ValidationReached(Exception):
            pass

        # Stop at the clock immediately after the real request validation;
        # no model, KV pool, GPU allocation, or scheduler loop is required.
        for scheduler in [CompactSpecScheduler, AsyncCPUCompactSpecScheduler]:
            for ratio, capacity, valid in [(.1, 6144, True), (.15, 6144, True),
                    (.2, 6144, False), (.25, 6144, False),
                    (.2, 9216, True), (.25, 9216, True)]:
                with self.subTest(scheduler=scheduler.__name__, ratio=ratio, capacity=capacity):
                    owner = SimpleNamespace(spec_stride=32,
                        spec_cache_args=dict(num_min_budget=192, budget_ratio=ratio))
                    request = SimpleNamespace(len_oracle_ctx=32768, len_output=32000, len_input=768)
                    with patch.object(CompactCachePtr, "MAX_LEN_SELECTED_KV_CACHE", capacity), \
                         patch("time.perf_counter", side_effect=ValidationReached):
                        with self.assertRaises(ValidationReached if valid else ValueError):
                            scheduler.run(owner, [request])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SelectedCapacity(unittest.TestCase):
    def test_topk_9216_and_graph_replay(self):
        # TP2 Qwen3 geometry: 36 layers * 4 local KV heads. Eight 32K
        # verify requests reach the existing aggregate score-buffer boundary.
        heads, requests, length, capacity = 144, 8, 32768, 9216
        wrapper = BatchTopKWrapper(MAX_TOTAL_VERIFY_KV_LEN=262144,
                                  MAX_DRAFT_KV_LEN=capacity)
        torch.manual_seed(330)
        values = torch.rand(heads, requests * length, device="cuda", dtype=torch.float16)
        indices = torch.arange(requests * length, device="cuda", dtype=torch.int32)
        starts = torch.arange(requests, device="cuda", dtype=torch.int32) * length
        lengths = torch.full_like(starts, length)
        ks = torch.tensor([8201, 6561] * 4, device="cuda", dtype=torch.int32)
        out_values = torch.empty(requests, heads, capacity, device="cuda", dtype=values.dtype)
        out_indices = torch.empty(requests, heads, capacity, device="cuda", dtype=torch.int32)

        def run():
            wrapper.run(values, indices, starts, starts, lengths, ks, requests,
                        out_val=out_values, out_idx=out_indices)

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        ks.fill_(8210)
        graph.replay()
        torch.cuda.synchronize()
        for request in range(requests):
            k = int(ks[request])
            picked = out_indices[request, :, :k].long()
            self.assertTrue(bool(((picked >= request * length)
                                  & (picked < (request + 1) * length)).all()))
            sorted_indices = picked.sort(dim=-1).values
            self.assertTrue(bool((sorted_indices[:, 1:] > sorted_indices[:, :-1]).all()))
            actual = torch.gather(values, 1, picked)
            self.assertTrue(torch.equal(actual, out_values[request, :, :k]))
            for head in [0, heads - 1]:
                expected = values[head, request * length:(request + 1) * length].topk(k).values
                self.assertTrue(torch.equal(actual[head].sort(descending=True).values, expected))

    def test_concat_uses_subclass_capacity(self):
        class ExpandedPtr(PillarCachePtr):
            MAX_LEN_SELECTED_KV_CACHE = 9216

        layers, heads = 2, 4
        counts, recent = [8201, 6561], [3, 5]
        selected, requests = [], []
        for request, (count, tail) in enumerate(zip(counts, recent)):
            tensor = torch.arange(layers * heads * 9216, device="cuda", dtype=torch.int32)
            tensor = tensor.reshape(layers, heads, 9216) + request * 1000000
            selected.append(tensor)
            ptr = SimpleNamespace(len_selected_indices_cpu=0,
                selected_indices_ptr=tensor.data_ptr(), len_selected_indices_gpu=count,
                len_loading_indices=count + tail)
            requests.append(SimpleNamespace(status=Request.ReqExecType.DRAFT, kv_cache_ptr=ptr))
        offsets = np.array([0, counts[0] + recent[0], sum(counts) + sum(recent)], dtype=np.int32)
        all_indices = torch.arange(65536, device="cuda", dtype=torch.int32)
        output = torch.full((layers, heads, ExpandedPtr.MAX_TOTAL_DRAFT_KV_LEN), -1,
                            device="cuda", dtype=torch.int32)
        ExpandedPtr.concat_selected_indices(
            dict(kv_indptr=offsets, all_kv_indptr=np.array([0, 32768, 65536], dtype=np.int32)),
            dict(all_kv_indices=all_indices), requests, output, layers, heads)
        torch.cuda.synchronize()
        for request, (count, tail) in enumerate(zip(counts, recent)):
            start = int(offsets[request])
            self.assertTrue(torch.equal(output[..., start:start + count], selected[request][..., :count]))
            expected_tail = all_indices[(request + 1) * 32768 - tail:(request + 1) * 32768]
            self.assertTrue(torch.equal(output[..., start + count:start + count + tail],
                                        expected_tail.expand(layers, heads, tail)))
        self.assertTrue(bool((output[..., int(offsets[-1]):] == -1).all()))


if __name__ == "__main__":
    unittest.main()
