"""CPU checks for bounded arenas and device-completion lifetime contracts."""

import unittest

import torch

from serve.attention.kv_buffer_pool import KVBufferPool


class CompletionEvent:
    """Controllable stand-in for a recorded device event; no CUDA is needed."""

    def __init__(self, ready=False):
        self.ready = ready
        self.waits = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True


class KVBufferPoolTest(unittest.TestCase):
    def make_pool(self, size=1024):
        return KVBufferPool(size, device="cpu", pin_memory=False)

    def test_end_allocations_are_aligned_with_an_unaligned_arena_boundary(self):
        pool = self.make_pool(1027)
        hot = pool.allocate((128,), torch.float16)
        restore = pool.allocate((128,), torch.float16, from_end=True)
        self.assertEqual((hot.offset, restore.offset), (0, 768))
        self.assertEqual(pool._free, [(256, 512), (1024, 3)])
        hot.tensor.fill_(3)
        restore.tensor.fill_(7)
        self.assertTrue(torch.all(hot.tensor == 3))
        self.assertEqual(pool.stats()["largest_free_bytes"], 512)
        restore.release(); hot.release()
        self.assertEqual(pool._free, [(0, 1027)])
        pool.close()

    def test_replacement_hot_leases_do_not_split_growing_restore_space(self):
        pool = self.make_pool(16 * 256)
        hot = pool.allocate((256,), torch.uint8)
        for cold_blocks in (8, 9, 10, 11):
            restore = pool.allocate((cold_blocks * 256,), torch.uint8, from_end=True)
            replacement = pool.allocate((256,), torch.uint8)
            restore.tensor.fill_(cold_blocks)
            replacement.tensor.fill_(23)
            self.assertTrue(torch.all(restore.tensor == cold_blocks))
            hot.release(); restore.release()
            hot = replacement
        hot.release()
        self.assertEqual(pool._free, [(0, 16 * 256)])
        pool.close()

    def test_typed_views_do_not_overlap(self):
        pool = self.make_pool()
        first = pool.allocate((2, 3, 8), torch.float16)
        second = pool.allocate((4, 8), torch.float32)
        first.tensor.fill_(7)
        second.tensor.fill_(19)
        self.assertTrue(first.tensor.is_contiguous())
        self.assertEqual(first.tensor.shape, (2, 3, 8))
        self.assertEqual(second.tensor.dtype, torch.float32)
        self.assertGreaterEqual(second.tensor.data_ptr(), first.tensor.data_ptr() + first.tensor.nbytes)
        self.assertTrue(torch.all(first.tensor == 7))
        self.assertTrue(torch.all(second.tensor == 19))
        self.assertEqual(pool.stats()["in_use_bytes"], 512)
        first.release()
        second.release()
        self.assertEqual(pool.stats()["free_bytes"], 1024)
        pool.close()

    def test_exhaustion_and_empty_tensor(self):
        pool = self.make_pool(256)
        full = pool.allocate((128,), torch.float16)
        before = pool.stats()
        with self.assertRaises(MemoryError):
            pool.allocate((1,), torch.float16)
        self.assertEqual(pool.stats(), before)
        empty = pool.allocate((2, 0, 3), torch.float16)
        self.assertEqual(empty.tensor.shape, (2, 0, 3))
        self.assertEqual(pool.stats()["in_use_bytes"], 256)
        empty.release()
        full.release()
        pool.close()

    def test_adjacent_frees_coalesce_without_touching_live_data(self):
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                pool = self.make_pool(768)
                pieces = [pool.allocate((128,), torch.float16) for _ in range(3)]
                pieces[2].tensor.fill_(31)
                for index in order:
                    pieces[index].release()
                joined = pool.allocate((256,), torch.float16)
                joined.tensor.fill_(4)
                self.assertTrue(torch.all(pieces[2].tensor == 31))
                self.assertEqual(pool.stats()["in_use_bytes"], 768)
                joined.release()
                pieces[2].release()
                pool.close()

    def test_fragmented_space_does_not_grow_or_fallback(self):
        pool = self.make_pool()
        pieces = [pool.allocate((128,), torch.float16) for _ in range(4)]
        pieces[0].release()
        pieces[2].release()
        before = pool.stats()
        self.assertEqual(before["free_bytes"], 512)
        with self.assertRaises(MemoryError):
            pool.allocate((256,), torch.float16)
        self.assertEqual(pool.stats(), before)
        pieces[1].release()
        pieces[3].release()
        whole = pool.allocate((512,), torch.float16)
        whole.release()
        pool.close()

    def test_reuse_waits_for_every_reader_without_host_wait(self):
        pool = self.make_pool(256)
        lease = pool.allocate((128,), torch.float16)
        pointer = lease.tensor.data_ptr()
        copy_done, consumer_done = CompletionEvent(), CompletionEvent()
        lease.release(after=(copy_done, consumer_done))
        self.assertEqual(pool.stats()["pending_bytes"], 256)
        copy_done.ready = True
        with self.assertRaises(MemoryError):
            pool.allocate((128,), torch.float16)
        consumer_done.ready = True
        reused = pool.allocate((128,), torch.float16)
        self.assertEqual(reused.tensor.data_ptr(), pointer)
        self.assertEqual((copy_done.waits, consumer_done.waits), (0, 0))
        reused.release()
        pool.close()

    def test_close_and_double_return_guards(self):
        pool = self.make_pool(256)
        lease = pool.allocate((128,), torch.float16)
        with self.assertRaises(RuntimeError):
            pool.close(wait=True)
        done = CompletionEvent()
        lease.release(after=(done,))
        with self.assertRaises(RuntimeError):
            lease.release()
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertEqual(done.waits, 0)
        pool.close(wait=True)
        self.assertEqual(done.waits, 1)
        pool.close()
        with self.assertRaises(RuntimeError):
            pool.allocate((1,), torch.float16)

    def test_invalid_metadata_preserves_capacity(self):
        for kwargs in (dict(capacity_bytes=0), dict(capacity_bytes=256, alignment=3)):
            with self.assertRaises(ValueError):
                KVBufferPool(**kwargs, pin_memory=False)
        pool = self.make_pool()
        for shape in ((-1,), (True,), (1.5,)):
            with self.assertRaises(ValueError):
                pool.allocate(shape, torch.float16)
        lease = pool.allocate((128,), torch.float16)
        with self.assertRaises(ValueError):
            lease.release(after=(object(),))
        self.assertEqual(pool.stats()["active_leases"], 1)
        lease.release()
        pool.close()


if __name__ == "__main__":
    unittest.main()
