"""Dense first-round correctness and explicit lease ownership checks."""

import unittest

import torch

from serve.attention.full_forward import FullForward, FullKVRegion
from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.packed_attention import attention_segment


class FullForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.pool = KVBufferPool(65536, pin_memory=False)
        self.region = FullKVRegion(self.pool, layers=2, capacity=15, heads=2, dim=4,
                                   dtype=torch.float64)

    def tearDown(self):
        if self.region._forward_active:
            self.frame.abort()
        self.region.close()
        self.pool.close()

    def forward(self, q, keys, values, **kwargs):
        self.frame = FullForward(self.region, query_tokens=q.shape[0], backend="torch", **kwargs)
        output = [self.frame.layer(layer, q, keys, values) for layer in range(2)]
        self.frame.finish()
        return output

    def test_chunk_prefill_matches_full_causal_attention(self):
        q, keys, values = torch.randn(3, 9, 2, 4, dtype=torch.float64)
        q = q.repeat_interleave(2, dim=1)
        outputs = []
        for start, end in ((0, 3), (3, 7), (7, 9)):
            outputs.append(self.forward(q[start:end], keys[start:end], values[start:end])[0])
            self.assertEqual(self.frame.position_start, start)
        expected = attention_segment(q, self.region.kv[0, :9], causal=True, backend="torch")
        torch.testing.assert_close(torch.cat(outputs), expected.output)
        self.assertEqual(self.region.length, 9)

    def test_verify_rewrites_tail_and_truncates_rejected_kv(self):
        q, k, v = torch.randn(3, 6, 2, 4, dtype=torch.float64)
        self.forward(q, k, v)
        q, k, v = torch.randn(3, 3, 2, 4, dtype=torch.float64)
        self.forward(q, k, v, write_offset=5, verify=True)
        self.assertEqual(self.frame.position_start, 5)
        expected = attention_segment(q, self.region.kv[0, :8], causal=True, backend="torch")
        self.assertEqual(self.frame.scores.shape, (2, 2, 8))
        self.assertEqual(self.frame.accept(2), 7)
        torch.testing.assert_close(self.region.kv[0, 5:7, 0, 0], k[:2])
        with self.assertRaises(ValueError):
            self.frame.accept(2)
        self.assertTrue(torch.isfinite(expected.lse).all())

    def test_partial_forward_invalidates_region(self):
        self.frame = FullForward(self.region, query_tokens=1, backend="torch")
        q, k, v = torch.randn(3, 1, 2, 4, dtype=torch.float64)
        self.frame.layer(0, q, k, v)
        self.frame.abort()
        with self.assertRaises(RuntimeError):
            FullForward(self.region, query_tokens=1, backend="torch")

    def test_full_lease_returns_to_shared_arena(self):
        self.assertGreater(self.pool.stats()["in_use_bytes"], 0)
        self.region.close()
        self.assertEqual(self.pool.stats()["in_use_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
