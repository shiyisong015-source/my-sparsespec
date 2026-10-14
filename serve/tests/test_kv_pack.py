"""GPU correctness checks, also runnable with torchrun --nproc-per-node=2.

These tests exercise packing only; they do not load a model or change serving.
"""

import os
import unittest

import torch
import torch.distributed as dist

from serve.attention.kv_pack import pack_selected_kv


def reference_pack(source, indices, count):
    # Independent reference using PyTorch index_select for each local head.
    result = torch.empty(
        (source.shape[0], count, *source.shape[2:]),
        device=source.device, dtype=source.dtype,
    )
    for layer in range(source.shape[0]):
        for head in range(source.shape[4]):
            result[layer, :, :, 0, head, :] = source[layer, :, :, 0, head, :].index_select(
                0, indices[layer, head, :count].long(),
            )
    return result


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KVPackTest(unittest.TestCase):
    def make_inputs(self, count=9, heads=4, dtype=torch.float16, dim=128):
        generator = torch.Generator(device="cpu").manual_seed(330)
        source = torch.randn((3, 41, 2, 1, heads, dim), generator=generator).cuda().to(dtype)
        indices = torch.randint(41, (3, heads, count + 5), generator=generator, dtype=torch.int32).cuda()
        indices[..., count:] = -1  # Padding must never be read.
        return source, indices

    def test_data_and_boundaries(self):
        cases = [
            (0, 4, torch.float16, 128, 2, 8),
            (1, 1, torch.float16, 128, 1, 8),
            (7, 4, torch.float16, 128, 2, 8),
            (8, 4, torch.bfloat16, 128, 2, 8),
            (9, 4, torch.float16, 128, 2, 8),
            (33, 4, torch.float32, 128, 4, 16),
            (9, 4, torch.float16, 80, 2, 8),
        ]
        for count, heads, dtype, dim, programs, tile in cases:
            with self.subTest(count=count, heads=heads, dtype=dtype, dim=dim):
                source, indices = self.make_inputs(count, heads, dtype, dim)
                original = source.clone()
                destination = torch.full(
                    (3, count + 7, 2, 1, heads, dim), -17, device="cuda", dtype=dtype,
                )
                pack_selected_kv(source, indices, destination, count, destination_offset=3,
                                 num_programs=programs, tokens_per_tile=tile)
                expected = reference_pack(source, indices, count)
                torch.testing.assert_close(destination[:, 3:3 + count], expected, rtol=0, atol=0)
                self.assertTrue(torch.all(destination[:, :3] == -17).item())
                self.assertTrue(torch.all(destination[:, 3 + count:] == -17).item())
                torch.testing.assert_close(source, original, rtol=0, atol=0)

    def test_noncontiguous_selection_and_int64(self):
        source, indices = self.make_inputs()
        storage = torch.full((*indices.shape[:2], indices.shape[2] * 2), -1,
                             dtype=torch.int64, device="cuda")
        storage[..., ::2] = indices
        indices = storage[..., ::2]
        destination = torch.empty((3, 9, 2, 1, 4, 128), device="cuda", dtype=source.dtype)
        pack_selected_kv(source, indices, destination, 9)
        torch.testing.assert_close(destination, reference_pack(source, indices, 9), rtol=0, atol=0)

    def test_direct_pinned_cpu_destination_preserves_padding(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            source, indices = self.make_inputs(count=33, dtype=dtype)
            destination = torch.full((3, 40, 2, 1, 4, 128), -17,
                                     dtype=dtype, device="cpu", pin_memory=True)
            pack_selected_kv(source, indices, destination, 33, destination_offset=3)
            torch.cuda.current_stream().record_event().synchronize()
            torch.testing.assert_close(destination[:, 3:36], reference_pack(source, indices, 33).cpu(),
                                       rtol=0, atol=0)
            self.assertTrue(torch.all(destination[:, :3] == -17))
            self.assertTrue(torch.all(destination[:, 36:] == -17))
            with self.assertRaisesRegex(ValueError, "pinned CPU"):
                pack_selected_kv(source, indices, torch.empty(destination.shape, dtype=dtype, device="cpu"), 33)

    def test_disjoint_regions_in_one_arena(self):
        source, indices = self.make_inputs()
        destination_shape = (3, 9, 2, 1, 4, 128)
        destination_elements = 3 * 9 * 2 * 4 * 128
        arena = torch.empty(source.numel() + destination_elements, device="cuda", dtype=source.dtype)
        pooled_source = arena[:source.numel()].view_as(source)
        pooled_source.copy_(source)
        destination = arena[source.numel():].view(destination_shape)
        pack_selected_kv(pooled_source, indices, destination, 9)
        torch.testing.assert_close(destination, reference_pack(source, indices, 9), rtol=0, atol=0)
        # A shifted region with overlapping bytes must still be rejected.
        overlapping = arena[1:1 + destination_elements].view(destination_shape)
        with self.assertRaises(ValueError):
            pack_selected_kv(pooled_source, indices, overlapping, 9)

    def test_repack_from_hot_and_cold_without_reading_hot_padding(self):
        for dtype in (torch.float16, torch.bfloat16):
            layers, heads, dim, hot_live, cold_count, count = 2, 4, 128, 5, 7, 7
            shapes = [(layers, pages, 2, 1, heads, dim) for pages in (10, cold_count, count + 2)]
            sizes = [layers * shape[1] * 2 * heads * dim for shape in shapes]
            arena = torch.empty(sum(sizes), device="cuda", dtype=dtype)
            hot = arena[:sizes[0]].view(shapes[0])
            cold = arena[sizes[0]:sizes[0] + sizes[1]].view(shapes[1])
            destination = arena[sizes[0] + sizes[1]:].view(shapes[2])
            generator = torch.Generator().manual_seed(330)
            hot.copy_(torch.randn(shapes[0], generator=generator))
            cold.copy_(torch.randn(shapes[1], generator=generator))
            hot[:, hot_live:].fill_(10000)
            destination.fill_(-17)
            ids = torch.tensor([0, 4, 5, 11, 1, 6, 10], dtype=torch.int32)
            indices = torch.stack([torch.stack([ids.roll(layer + head) for head in range(heads)])
                                   for layer in range(layers)]).cuda()
            expected = reference_pack(torch.cat((hot[:, :hot_live], cold), dim=1), indices, count)
            pack_selected_kv(hot, indices, destination, count, destination_offset=1,
                             cold_source=cold, source_hot_length=hot_live)
            torch.testing.assert_close(destination[:, 1:1 + count], expected, rtol=0, atol=0)
            self.assertTrue(torch.all(destination[:, :1] == -17).item())
            self.assertTrue(torch.all(destination[:, 1 + count:] == -17).item())
            self.assertTrue(torch.all(hot[:, hot_live:] == 10000).item())

    def test_stream_dependencies_and_next_consumer(self):
        source, indices = self.make_inputs()
        destination = torch.empty((3, 9, 2, 1, 4, 128), device="cuda", dtype=source.dtype)
        current = torch.cuda.current_stream()
        ready = current.record_event()
        packing = torch.cuda.Stream()
        with torch.cuda.stream(packing):
            packing.wait_event(ready)
            pack_selected_kv(source, indices, destination, 9)
            packed_ready = packing.record_event()
        current.wait_event(packed_ready)
        consumed = destination + 1
        torch.testing.assert_close(consumed, reference_pack(source, indices, 9) + 1, rtol=0, atol=0)

    def test_fixed_metadata_cuda_graph(self):
        source, indices = self.make_inputs()
        destination = torch.empty((3, 9, 2, 1, 4, 128), device="cuda", dtype=source.dtype)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            pack_selected_kv(source, indices, destination, 9)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_selected_kv(source, indices, destination, 9)
        source.add_(1)
        indices[..., :9] = indices[..., :9].flip(-1)
        graph.replay()
        torch.testing.assert_close(destination, reference_pack(source, indices, 9), rtol=0, atol=0)

    def test_invalid_metadata(self):
        source, indices = self.make_inputs()
        destination = torch.empty((3, 9, 2, 1, 4, 128), device="cuda", dtype=source.dtype)
        for kwargs in [dict(destination_offset=1), dict(num_programs=0),
                       dict(tokens_per_tile=3), dict(num_warps=3)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pack_selected_kv(source, indices, destination, 9, **kwargs)
        with self.assertRaises(ValueError):
            pack_selected_kv(source, indices, source, 9)

    def test_tp_local_heads_match_full_reference(self):
        if not dist.is_initialized():
            self.skipTest("Run with torchrun for the two-rank check")
        rank, world = dist.get_rank(), dist.get_world_size()
        self.assertEqual(world, 2)
        generator = torch.Generator(device="cpu").manual_seed(331)
        # All ranks construct the same eight-head reference, then use their
        # own four-head source/selection tensors. No model KV is all-gathered.
        full = torch.randn((3, 41, 2, 1, 8, 128), generator=generator).cuda().half()
        indices = torch.stack([
            torch.stack([torch.randperm(41, generator=generator)[:9] for _ in range(8)])
            for _ in range(3)
        ]).cuda().int()
        start, end = rank * 4, (rank + 1) * 4
        local = full[..., start:end, :].contiguous()
        local_indices = indices[:, start:end].contiguous()
        output = torch.empty((3, 9, 2, 1, 4, 128), device="cuda", dtype=full.dtype)
        pack_selected_kv(local, local_indices, output, 9)
        # Gathering only in this test verifies the rank-local partitioning.
        pieces = [torch.empty_like(output) for _ in range(world)]
        dist.all_gather(pieces, output)
        combined = torch.cat(pieces, dim=4)
        torch.testing.assert_close(combined, reference_pack(full, indices, 9), rtol=0, atol=0)


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
