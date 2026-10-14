"""CPU lifecycle/reference checks; real transfer integration is CUDA opt-in."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from serve.attention.kv_buffer_pool import KVBufferPool
from serve.attention.kv_partition import PackedKVRegion
from serve.attention.packed_attention import PackedAttention, attention_segment
from serve.attention.packed_forward import PackedForward
from serve.tests.test_kv_buffer_pool import CompletionEvent
from serve.tests.test_packed_attention import full_reference, make_region, reference_gather


def verify_fixture():
    q, history, tail, region, restored, plan = make_region(3)
    region.tail_length = 2  # The final verify query has not yet written KV.
    frame = PackedForward(region, mode="verify", query_tokens=3, restored=restored,
                          attention=PackedAttention(backend="torch"))
    return q, history, tail, region, restored, plan, frame


def complete_verify(fixture):
    q, _, tail, region, _, _, frame = fixture
    for layer in range(region.layers):
        frame.layer(layer, q, tail[layer, :, 0, 0], tail[layer, :, 1, 0])
    frame.finish()
    return frame


class PackedForwardTest(unittest.TestCase):
    def test_draft_publishes_only_after_all_layers(self):
        q, _, tail, region, _, _ = make_region(1)
        region.tail_length = 0
        frame = PackedForward(region, mode="draft", query_tokens=1,
                              attention=PackedAttention(backend="torch"))
        output = frame.layer(0, q, tail[0, :, 0, 0], tail[0, :, 1, 0])
        expected, _, _ = full_reference(q, region._hot[0, :region.hot_count + 1], region.hot_count)
        torch.testing.assert_close(output, expected, rtol=1e-11, atol=1e-11)
        self.assertEqual(region.tail_length, 0)
        with self.assertRaises(RuntimeError):
            frame.finish()
        with self.assertRaises(RuntimeError):
            PackedForward(region, mode="draft", query_tokens=1)
        with self.assertRaises(RuntimeError):
            frame.layer(0, q, tail[0, :, 0, 0], tail[0, :, 1, 0])
        frame.layer(1, q, tail[1, :, 0, 0], tail[1, :, 1, 0])
        self.assertEqual(frame.finish(), (None, None))
        self.assertEqual(region.tail_length, 1)
        with self.assertRaises(RuntimeError):
            frame.finish()

    def test_verify_commit_requires_completion_and_then_repacking(self):
        fixture = verify_fixture()
        q, history, tail, region, restored, _, frame = fixture
        with self.assertRaises(RuntimeError):
            frame.accept(2)
        for layer in range(region.layers):
            output = frame.layer(layer, q, tail[layer, :, 0, 0], tail[layer, :, 1, 0])
            expected, _, _ = full_reference(q, torch.cat((history[layer], tail[layer])), 11)
            torch.testing.assert_close(output, expected, rtol=1e-11, atol=1e-11)
        frame.finish()
        self.assertEqual(region.tail_length, 3)
        with self.assertRaises(RuntimeError):
            PackedForward(region, mode="draft", query_tokens=1)
        for invalid in (0, 4, True):
            with self.assertRaises(ValueError):
                frame.accept(invalid)
        self.assertEqual(frame.accept(2), 13)
        self.assertEqual(region.tail_length, 2)
        self.assertTrue(region._needs_repack)
        with self.assertRaises(RuntimeError):
            frame.accept(2)
        with self.assertRaises(RuntimeError):
            PackedForward(region, mode="draft", query_tokens=1)
        self.assertIsNotNone(restored.kv)

    def test_invalid_query_does_not_overwrite_history(self):
        q, _, tail, region, restored, _ = make_region(3)
        original = region.hot_kv.clone()
        with self.assertRaises(ValueError):
            PackedForward(region, mode="verify", query_tokens=3, restored=restored)
        self.assertFalse(getattr(region, "_forward_active", False))
        region.tail_length = 2
        restored.count -= 1
        with self.assertRaises(ValueError):
            PackedForward(region, mode="verify", query_tokens=3, restored=restored)
        restored.count += 1
        frame = PackedForward(region, mode="verify", query_tokens=3, restored=restored,
                              attention=PackedAttention(backend="torch"))
        with self.assertRaises(ValueError):
            frame.layer(0, q, tail[0, :, 0, 0].float(), tail[0, :, 1, 0].float())
        torch.testing.assert_close(region.hot_kv, original, rtol=0, atol=0)
        frame.abort()
        next_frame = PackedForward(region, mode="draft", query_tokens=1)
        frame.abort()  # Repeated cancellation must not clear the next owner.
        self.assertTrue(region._forward_active)
        next_frame.abort()

    def test_partial_failure_invalidates_the_region(self):
        fixture = verify_fixture()
        q, _, tail, region, _, _, frame = fixture
        with patch.object(frame.attention, "verify_layer", side_effect=RuntimeError("kernel failure")):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                frame.layer(0, q, tail[0, :, 0, 0], tail[0, :, 1, 0])
        frame.abort()
        self.assertFalse(region._forward_active)
        with self.assertRaises(RuntimeError):
            _ = region.hot_kv
        with self.assertRaises(RuntimeError):
            frame.finish()

    def test_partial_cancellation_requires_region_retirement(self):
        fixture = verify_fixture()
        q, _, tail, region, _, _, frame = fixture
        frame.layer(0, q, tail[0, :, 0, 0], tail[0, :, 1, 0])
        frame.abort()
        with self.assertRaises(RuntimeError):
            PackedForward(region, mode="draft", query_tokens=1)

    def test_reselection_matches_original_token_score_ranking(self):
        fixture = verify_fixture()
        q, history, tail, region, _, initial, _ = fixture
        frame = complete_verify(fixture)
        frame.accept(2)
        plan = frame.selection_plan(budget_ratio=.5, min_tokens=1, validate=True)
        self.assertEqual((plan.hot_count, plan.cold_count, plan.recent_count), (7, 6, 3))
        for layer in range(region.layers):
            _, _, scores = full_reference(q, torch.cat((history[layer], tail[layer])), 11)
            for head in range(region.heads):
                identities = torch.cat((initial.hot_pages[layer, head], torch.arange(11, 13),
                                        initial.cold_pages[layer, head]))
                actual_hot = identities[plan.hot_pages[layer, head].long()]
                expected_selected = torch.topk(scores[head, :10].half(), 4).indices
                self.assertEqual(set(actual_hot[:4].tolist()), set(expected_selected.tolist()))
                self.assertEqual(actual_hot[-3:].tolist(), [10, 11, 12])
                actual_cold = identities[plan.cold_pages[layer, head].long()]
                self.assertEqual(sorted(torch.cat((actual_hot, actual_cold)).tolist()), list(range(13)))
        tiny = frame.selection_plan(budget_ratio=.01, min_tokens=1, validate=True)
        self.assertEqual((tiny.hot_count, tiny.selected_count), (3, 0))
        full = frame.selection_plan(budget_ratio=1, min_tokens=1, validate=True)
        self.assertEqual((full.hot_count, full.cold_count), (13, 0))

    def test_next_round_attention_preserves_canonical_full_history(self):
        fixture = verify_fixture()
        _, history, tail, region, restored, _, _ = fixture
        frame = complete_verify(fixture)
        frame.accept(2)
        plan = frame.selection_plan(budget_ratio=.5, min_tokens=1, validate=True)
        virtual_kv = torch.cat((region.hot_kv[:, :region.hot_length], restored.kv), dim=1)
        next_region = PackedKVRegion()
        next_region.__dict__.update(region.__dict__)
        next_region.history_length = 13
        next_region.hot_count, next_region.cold_count = plan.hot_count, plan.cold_count
        next_region.selected_count, next_region.recent_count = plan.selected_count, plan.recent_count
        next_region.tail_length = 0
        next_region._needs_repack = next_region._verification_pending = False
        next_region._hot = torch.empty((2, plan.hot_count + 5, 2, 1, 2, 8), dtype=region.dtype)
        next_region._hot[:, :plan.hot_count] = reference_gather(virtual_kv, plan.hot_pages)
        next_cold = type(restored)(reference_gather(virtual_kv, plan.cold_pages), 0, plan.cold_count,
                                  CompletionEvent(True))
        generator = torch.Generator().manual_seed(331)
        q = torch.randn((2, 8, 8), generator=generator, dtype=region.dtype)
        target_tail = torch.randn((2, 2, 2, 1, 2, 8), generator=generator, dtype=region.dtype)
        draft = PackedForward(next_region, mode="draft", query_tokens=1,
                              attention=PackedAttention(backend="torch"))
        for layer in range(2):
            draft.layer(layer, q[:1], target_tail[layer, :1, 0, 0], target_tail[layer, :1, 1, 0])
        draft.finish()
        verify = PackedForward(next_region, mode="verify", query_tokens=2, restored=next_cold,
                               attention=PackedAttention(backend="torch"))
        for layer in range(2):
            output = verify.layer(layer, q, target_tail[layer, :, 0, 0], target_tail[layer, :, 1, 0])
            canonical = torch.cat((history[layer], tail[layer, :2], target_tail[layer]))
            expected, _, _ = full_reference(q, canonical, 13)
            torch.testing.assert_close(output, expected, rtol=1e-11, atol=1e-11)
        verify.finish()
        verify.accept(1)

    def test_failed_reservation_keeps_old_owner_and_adoption_waits(self):
        fixture = verify_fixture()
        _, _, _, region, restored, _, _ = fixture
        frame = complete_verify(fixture)
        frame.accept(2)
        with patch.object(PackedKVRegion, "start_repartition", side_effect=MemoryError("arena full")):
            with self.assertRaises(MemoryError):
                frame.start_repack(budget_ratio=.5, min_tokens=1)
        self.assertEqual(region.phase, "packed")
        self.assertIsNotNone(restored.kv)
        replacement = PackedKVRegion()
        replacement.phase = "offloading"
        replacement._previous_region, replacement._restored_source = region, restored
        replacement.offload_ready = CompletionEvent(True)
        with patch.object(PackedKVRegion, "start_repartition", return_value=replacement):
            frame.start_repack(budget_ratio=.5, min_tokens=1)
        with self.assertRaises(RuntimeError):
            frame.adopt(replacement)
        self.assertEqual(region.phase, "packed")
        region.offload_ready = CompletionEvent(True)
        region._restore_events, region._leases = [], {}
        replacement.phase = "packed"
        self.assertIs(frame.adopt(replacement), replacement)
        self.assertEqual(region.phase, "closed")
        self.assertIsNone(restored.kv)
        self.assertIsNone(replacement._previous_region)
        with self.assertRaises(RuntimeError):
            frame.adopt(replacement)


@unittest.skipUnless(os.environ.get("RUN_PACKED_KV_GPU") == "1" and torch.cuda.is_available(),
                     "Full CUDA round awaits an available GPU")
class PackedForwardCudaTest(unittest.TestCase):
    def test_offload_verify_repack_restore_lifecycle(self):
        q, history, tail, _, _, plan = make_region(3, heads=4, dim=128, device="cuda", dtype=torch.float16)
        plan = type(plan)(*(tensor.cuda() for tensor in (plan.full_pages, plan.hot_pages, plan.cold_pages)),
                          plan.selected_count, plan.recent_count)
        cpu_pool = KVBufferPool(1 << 20)
        gpu_pool = KVBufferPool(1 << 20, device=history.device)
        copy_stream = torch.cuda.Stream(device=history.device)
        region = PackedKVRegion.start_offload(history, plan, tail_capacity=5, chunk_pages=3,
                                              cpu_pool=cpu_pool, gpu_pool=gpu_pool, stream=copy_stream)
        region.finish_offload()
        torch.testing.assert_close(region.source_pages_to_release(), plan.full_pages)
        # Two draft forwards leave the third query's KV absent.
        for position in range(2):
            draft = PackedForward(region, mode="draft", query_tokens=1)
            for layer in range(region.layers):
                draft.layer(layer, q[position:position + 1], tail[layer, position:position + 1, 0, 0],
                            tail[layer, position:position + 1, 1, 0])
            draft.finish()
        restored = region.restore_chunk(0, region.cold_count, stream=copy_stream)
        verify = PackedForward(region, mode="verify", query_tokens=3, restored=restored)
        for layer in range(region.layers):
            output = verify.layer(layer, q, tail[layer, :, 0, 0], tail[layer, :, 1, 0])
            expected, _, _ = full_reference(q, torch.cat((history[layer], tail[layer])), 11)
            torch.testing.assert_close(output.double(), expected, rtol=1e-2, atol=3e-3)
        verify.finish()
        verify.accept(2)
        replacement = verify.start_repack(budget_ratio=.5, min_tokens=1, stream=copy_stream, chunk_pages=3,
                                         cpu_pool=cpu_pool, gpu_pool=gpu_pool, validate=True)
        replacement.finish_offload()
        region = verify.adopt(replacement)
        with self.assertRaises(RuntimeError):
            region.source_pages_to_release()  # Repacking slots are not original pool pages.
        self.assertEqual(region.tail_length, 0)
        self.assertEqual(region.history_length, 13)
        restored = region.restore_chunk(0, region.cold_count, stream=copy_stream)
        torch.cuda.current_stream(history.device).wait_event(restored.ready)
        for layer in range(region.layers):
            actual = torch.cat((region.hot_kv[layer, :region.hot_count], restored.kv[layer]), dim=0)
            # All actual entries are history; append one identical new zero-KV query to each order.
            zero = torch.zeros_like(tail[layer, :1])
            state = attention_segment(q[:1], torch.cat((actual, zero)), causal=True)
            expected, _, _ = full_reference(q[:1], torch.cat((history[layer], tail[layer, :2], zero)), 13)
            torch.testing.assert_close(state.output.double(), expected, rtol=1e-2, atol=3e-3)
        consumed = torch.cuda.current_stream(history.device).record_event()
        restored.release(consumer_done=consumed)
        region.close(consumer_done=consumed)
        cpu_pool.close()
        gpu_pool.close(wait=True)

    def test_tp_local_heads_through_two_rounds(self):
        if not dist.is_initialized():
            self.skipTest("Run with torchrun for the two-rank packed-forward check")
        rank, world = dist.get_rank(), dist.get_world_size()
        self.assertEqual(world, 2)
        global_q, history, tail, _, _, plan = make_region(5, heads=8, dim=128,
                                                        device="cuda", dtype=torch.float16)
        local_history = history[..., rank * 4:(rank + 1) * 4, :].contiguous()
        local_plan = type(plan)(plan.full_pages.cuda(),
                                plan.hot_pages[:, rank * 4:(rank + 1) * 4].cuda().contiguous(),
                                plan.cold_pages[:, rank * 4:(rank + 1) * 4].cuda().contiguous(),
                                plan.selected_count, plan.recent_count)
        cpu_pool = KVBufferPool(1 << 20)
        gpu_pool = KVBufferPool(1 << 20, device=local_history.device)
        copy_stream = torch.cuda.Stream(device=local_history.device)
        draft_stream = torch.cuda.Stream(device=local_history.device)
        region = PackedKVRegion.start_offload(local_history, local_plan, tail_capacity=6, chunk_pages=3,
                                              cpu_pool=cpu_pool, gpu_pool=gpu_pool, stream=copy_stream)
        region.finish_offload()
        region.source_pages_to_release()
        canonical = history
        generator = torch.Generator().manual_seed(333)
        for round_index, accepted in enumerate((3, 2)):
            queries = global_q.shape[0]
            local_q = global_q[:, rank * 16:(rank + 1) * 16].contiguous()
            local_tail = tail[..., rank * 4:(rank + 1) * 4, :].contiguous()
            for position in range(queries - 1):
                draft = PackedForward(region, mode="draft", query_tokens=1)
                with torch.cuda.stream(draft_stream):
                    for layer in range(region.layers):
                        draft.layer(layer, local_q[position:position + 1],
                                    local_tail[layer, position:position + 1, 0, 0],
                                    local_tail[layer, position:position + 1, 1, 0])
                draft.finish()
            restored = region.restore_chunk(0, region.cold_count, stream=copy_stream)
            # Verify changes to the default compute stream and must wait for draft.
            verify = PackedForward(region, mode="verify", query_tokens=queries, restored=restored)
            for layer in range(region.layers):
                output = verify.layer(layer, local_q, local_tail[layer, :, 0, 0], local_tail[layer, :, 1, 0])
                pieces = [torch.empty_like(output) for _ in range(world)]
                dist.all_gather(pieces, output)  # Test outputs only; production KV stays rank-local.
                combined = torch.cat(pieces, dim=1)
                expected, _, _ = full_reference(global_q, torch.cat((canonical[layer], tail[layer])),
                                                canonical.shape[1])
                torch.testing.assert_close(combined.double(), expected, rtol=1e-2, atol=3e-3)
            verify.finish()
            verify.accept(accepted)
            canonical = torch.cat((canonical, tail[:, :accepted]), dim=1)
            if round_index == 0:
                replacement = verify.start_repack(budget_ratio=.5, min_tokens=1, chunk_pages=3,
                                                 cpu_pool=cpu_pool, gpu_pool=gpu_pool, stream=copy_stream,
                                                 validate=True)
                replacement.finish_offload()
                dist.barrier()  # Both local migrations succeeded; this is test coordination.
                region = verify.adopt(replacement)
                self.assertEqual(region.history_length, canonical.shape[1])
                global_q = torch.randn((3, 32, 128), generator=generator).cuda().half()
                tail = torch.randn((2, 3, 2, 1, 8, 128), generator=generator).cuda().half()
            else:
                restored.release(consumer_done=verify.ready)
                region.close(consumer_done=verify.ready)
        self.assertEqual(cpu_pool.stats()["in_use_bytes"], 0)
        self.assertEqual(gpu_pool.stats()["in_use_bytes"], 0)
        cpu_pool.close()
        gpu_pool.close(wait=True)


if __name__ == "__main__":
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        result = unittest.main(exit=False).result
    finally:
        if distributed:
            dist.destroy_process_group()
    raise SystemExit(0 if result.wasSuccessful() else 1)
