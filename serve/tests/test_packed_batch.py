"""Batched model call contract and absolute-position checks without a GPU."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from serve.attention.packed_attention import PackedAttention
from serve.attention.packed_batch import PackedBatch
from serve.attention.packed_forward import PackedForward
from serve.tests.test_packed_attention import full_reference, make_region


CONFIG = SimpleNamespace(scaling_factor=1., rope_base=10000.,
                         high_freq_factor=None, low_freq_factor=None)


def batch_fixture():
    q1, _, tail1, draft_region, _, _ = make_region(1)
    q2, history2, tail2, verify_region, restored2, _ = make_region(3)
    verify_region.tail_length = 2
    frames = [PackedForward(draft_region, mode="draft", query_tokens=1,
                            attention=PackedAttention(backend="torch")),
              PackedForward(verify_region, mode="verify", query_tokens=3, restored=restored2,
                            attention=PackedAttention(backend="torch"))]
    generator = torch.Generator().manual_seed(332)
    new_tail1 = torch.randn(tail1.shape, generator=generator, dtype=tail1.dtype)
    calls = []

    def position_oracle(**kwargs):
        calls.append((kwargs["indptr"].tolist(), kwargs["offsets"].tolist(),
                      kwargs["rope_scale"], kwargs["rope_theta"]))
        # Fixture Q/K are already in their final positional representation.
        return kwargs["q"], kwargs["k"]

    batch = PackedBatch(frames, rope_fn=position_oracle)
    return batch, torch.cat((q1, q2)), new_tail1, history2, tail2, calls


class PackedBatchTest(unittest.TestCase):
    def test_query_head_geometry_cannot_change_after_rope_metadata_is_bound(self):
        batch, q, new_tail1, _, tail2, _ = batch_fixture()
        keys = torch.cat((new_tail1[0, :, 0, 0], tail2[0, :, 0, 0]))
        batch.apply_batch_rope(q, keys, CONFIG)
        with self.assertRaisesRegex(ValueError, "head count must remain fixed"):
            batch.apply_batch_rope(q.repeat_interleave(2, dim=1), keys, CONFIG)
        batch.abort()
        for frame in batch.frames:
            self.assertFalse(getattr(frame.region, "_forward_failed", False))

    def test_mixed_draft_verify_model_calls_preserve_request_outputs(self):
        batch, q, new_tail1, history2, tail2, calls = batch_fixture()
        self.assertEqual(batch.num_tokens, 4)
        self.assertEqual([frame.position_start for frame in batch.frames], [12, 11])
        for layer in range(2):
            keys = torch.cat((new_tail1[layer, :, 0, 0], tail2[layer, :, 0, 0]))
            values = torch.cat((new_tail1[layer, :, 1, 0], tail2[layer, :, 1, 0]))
            q_rotated, k_rotated = batch.apply_batch_rope(q, keys, CONFIG)
            draft_region = batch.frames[0].region
            draft_kv = torch.cat((draft_region.hot_kv[layer, :draft_region.hot_count + 1],
                                  new_tail1[layer]))
            expected_draft, _, _ = full_reference(q[:1], draft_kv, draft_kv.shape[0] - 1)
            expected_verify, _, _ = full_reference(q[1:], torch.cat((history2[layer], tail2[layer])), 11)
            batch.append_batch_kv_cache(k_rotated, values, layer)
            output = batch.batch_attn_fwd(q_rotated, layer)
            torch.testing.assert_close(output, torch.cat((expected_draft, expected_verify)),
                                       rtol=1e-11, atol=1e-11)
            self.assertEqual([frame.region.tail_length for frame in batch.frames], [1, 2])
        self.assertEqual(calls, [([0, 1, 4], [12, 11], 1., 10000.)] * 2)
        scores = batch.finish()
        self.assertEqual(scores[0], (None, None))
        self.assertEqual(scores[1][0].shape, (2, 2, 7))
        self.assertEqual([frame.region.tail_length for frame in batch.frames], [2, 3])
        batch.frames[1].accept(2)
        with self.assertRaises(RuntimeError):
            batch.finish()

    def test_layer_order_and_metadata_rejected_before_writes(self):
        batch, q, new_tail1, _, tail2, _ = batch_fixture()
        keys = torch.cat((new_tail1[0, :, 0, 0], tail2[0, :, 0, 0]))
        values = torch.cat((new_tail1[0, :, 1, 0], tail2[0, :, 1, 0]))
        with self.assertRaises(RuntimeError):
            batch.batch_attn_fwd(q, 0)
        with self.assertRaises(ValueError):
            batch.apply_batch_rope(q[:1], keys, CONFIG)
        with self.assertRaises(ValueError):
            batch.append_batch_kv_cache(keys.float(), values.float(), 0)
        with self.assertRaises(RuntimeError):
            batch.finish()
        batch.append_batch_kv_cache(keys, values, 0)
        with self.assertRaises(RuntimeError):
            batch.append_batch_kv_cache(keys, values, 1)
        with self.assertRaises(RuntimeError):
            batch.batch_attn_fwd(q, 1)
        batch.abort()  # No frame has written KV yet; both histories are usable.
        for frame in batch.frames:
            self.assertFalse(frame.region._forward_active)
            _ = frame.region.hot_kv

    def test_later_request_failure_invalidates_partially_written_batch(self):
        batch, q, new_tail1, _, tail2, _ = batch_fixture()
        keys = torch.cat((new_tail1[0, :, 0, 0], tail2[0, :, 0, 0]))
        values = torch.cat((new_tail1[0, :, 1, 0], tail2[0, :, 1, 0]))
        batch.append_batch_kv_cache(keys, values, 0)
        with patch.object(batch.frames[1].attention, "verify_layer", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                batch.batch_attn_fwd(q, 0)
        for frame in batch.frames:
            self.assertFalse(frame.region._forward_active)
            with self.assertRaises(RuntimeError):
                _ = frame.region.hot_kv
        batch.abort()  # Repeat cleanup returns the same events without altering owners.
        with self.assertRaises(RuntimeError):
            batch.finish()

    def test_empty_or_duplicate_request_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            PackedBatch([])
        batch, _, _, _, _, _ = batch_fixture()
        with self.assertRaises(ValueError):
            PackedBatch([batch.frames[0], batch.frames[0]])
        batch.abort()


@unittest.skipUnless(os.environ.get("RUN_PACKED_KV_GPU") == "1" and torch.cuda.is_available(),
                     "CUDA RoPE and mixed packed batch await an available GPU")
class PackedBatchCudaTest(unittest.TestCase):
    def test_real_rope_with_mixed_draft_verify(self):
        def reference_rope(tensor, position):
            dim = tensor.shape[-1]
            theta = CONFIG.rope_base ** (-torch.arange(0, dim, 2, device="cuda", dtype=torch.float64) / dim)
            angle = torch.arange(position, position + tensor.shape[0], device="cuda")[:, None] * theta
            first, second = tensor.double().chunk(2, dim=-1)
            cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
            return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(tensor.dtype)

        q1, _, tail1, draft_region, _, _ = make_region(1, heads=4, dim=128, device="cuda", dtype=torch.float16)
        q2, history2, tail2, verify_region, restored2, _ = make_region(3, heads=4, dim=128,
                                                                   device="cuda", dtype=torch.float16)
        verify_region.tail_length = 2
        frames = [PackedForward(draft_region, mode="draft", query_tokens=1),
                  PackedForward(verify_region, mode="verify", query_tokens=3, restored=restored2)]
        batch = PackedBatch(frames)
        q = torch.cat((q1, q2))
        expected_q = torch.cat((reference_rope(q1, 12), reference_rope(q2, 11)))
        for layer in range(2):
            keys = torch.cat((tail1[layer, :, 0, 0], tail2[layer, :, 0, 0]))
            values = torch.cat((tail1[layer, :, 1, 0], tail2[layer, :, 1, 0]))
            q_rotated, keys_rotated = batch.apply_batch_rope(q, keys, CONFIG)
            expected_keys = torch.cat((reference_rope(keys[:1], 12), reference_rope(keys[1:], 11)))
            torch.testing.assert_close(q_rotated, expected_q, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(keys_rotated, expected_keys, rtol=2e-3, atol=2e-3)
            draft_tail = tail1[layer].clone()
            draft_tail[:, 0, 0] = expected_keys[:1]
            verify_tail = tail2[layer].clone()
            verify_tail[:, 0, 0] = expected_keys[1:]
            draft_kv = torch.cat((draft_region.hot_kv[layer, :draft_region.hot_count + 1], draft_tail))
            expected_draft, _, _ = full_reference(expected_q[:1], draft_kv, draft_kv.shape[0] - 1)
            expected_verify, _, _ = full_reference(expected_q[1:], torch.cat((history2[layer], verify_tail)), 11)
            batch.append_batch_kv_cache(keys_rotated, values, layer)
            output = batch.batch_attn_fwd(q_rotated, layer)
            torch.testing.assert_close(output.double(), torch.cat((expected_draft, expected_verify)),
                                       rtol=1e-2, atol=3e-3)
        batch.finish()


if __name__ == "__main__":
    unittest.main()
