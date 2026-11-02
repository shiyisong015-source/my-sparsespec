"""CPU policy/ownership tests using the existing native compact fixtures."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import serve.tests.test_compact_native as fixtures
from serve.request.kv_cache_ptr.compact import PageTransfer
from serve.request.request import Request
with patch("torch.cuda.Stream", return_value=SimpleNamespace()):
    from serve.scheduler.reservation_scheduler import ReservationCompactSpecScheduler
from serve.scheduler.reservation_table import ReservationTable


class PolicyTests(unittest.TestCase):
    def scheduler(self):
        base = fixtures.CPUDynamicAdmission.scheduler(capacity=256, budget=32)
        scheduler = ReservationCompactSpecScheduler.__new__(ReservationCompactSpecScheduler)
        scheduler.__dict__.update(base.__dict__)
        scheduler.reservation_table = ReservationTable(16, 100., 1.)
        scheduler.reservation_target, scheduler.step_id = 4, 1
        scheduler._reservation_wait_since, scheduler._reservation_selection = {}, {}
        scheduler._reservation_log = None
        for key in ("admissions", "waits", "deadline_misses", "replans", "reload_starts",
                    "cpu_waits", "memory_waits", "empty_windows", "repack_waits",
                    "release_plans", "release_commits", "release_delays", "dependency_waits",
                    "dependent_admissions", "dependency_replans", "capacity_replans"):
            scheduler.metrics["reservation_" + key] = 0
        return scheduler

    def test_due_verify_preserves_native_window_math(self):
        scheduler = self.scheduler()
        request = Request([1], 32)
        request.request_id = 0
        request.status, request.len_draft_probs = Request.ReqExecType.DRAFT, 2
        reservation = scheduler.reservation_table.reserve(
            0, transfer_bytes=0, memory_pages=0, anchor=1, resident_pages=0, capacity_pages=256)
        scheduler.step_id = reservation.deadline
        scheduler._early_verify(request)
        self.assertEqual(request.status, Request.ReqExecType.VERIFY)
        self.assertEqual(scheduler.spec_stride + 1 - request.initial_idx, 3)

    def test_reload_can_draft_but_cannot_verify_until_commit(self):
        scheduler = self.scheduler()
        request = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        scheduler.reservation_table.reserve(0, transfer_bytes=10, memory_pages=7,
                                            anchor=4, resident_pages=20, capacity_pages=256)
        scheduler._transfers[0] = SimpleNamespace(direction="restore")
        self.assertTrue(scheduler._eligible(request))
        request.status = Request.ReqExecType.VERIFY
        self.assertFalse(scheduler._eligible(request))
        scheduler._transfers[0].direction = "repack"
        request.status = Request.ReqExecType.DRAFT
        self.assertFalse(scheduler._eligible(request))
        request.retire()

    def test_full_resident_cycle_does_not_reset_its_deadline_each_iteration(self):
        scheduler = self.scheduler()
        request = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        request.kv_cache_ptr.len_loading_indices = len(request.kv_cache_ptr.indices)
        scheduler._requests[0] = request
        scheduler._place_reservations()
        # The first observed selection changes the cycle exactly once.
        scheduler.step_id += 1
        scheduler.reservation_table.advance(scheduler.step_id)
        scheduler._place_reservations()
        deadline = scheduler.reservation_table.reservations[0].deadline
        scheduler.step_id += 1
        scheduler.reservation_table.advance(scheduler.step_id)
        scheduler._place_reservations()
        self.assertEqual(scheduler.reservation_table.reservations[0].deadline, deadline)
        request.retire()

    def test_inflight_restore_append_is_included_in_committed_page_table(self):
        scheduler = self.scheduler()
        request = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        ptr = request.kv_cache_ptr
        # Construct a completed compact layout without running a GPU copy.
        original = tuple(ptr.indices)
        ptr._release(original[5:])
        ptr.indices = type(ptr.indices)("i", original[:5])
        ptr.hot_pages, ptr.tail_pages = original[:5], []
        ptr.phase, ptr.recent_count = "hot", 2
        ptr.cold_cpu = torch.empty(2, 7, 2, 1, 2, 128)
        restored = ptr._allocate(7)
        ptr._pending = PageTransfer(restored, SimpleNamespace(query=lambda: True))
        ptr.allocate_tokens(1)
        tail = tuple(ptr.tail_pages)
        ptr.commit_restore(ptr._pending, nonblocking=True)
        self.assertEqual(len(ptr.indices), ptr.len_kv_cache)
        self.assertEqual(tuple(ptr.indices[-1:]), tail)
        self.assertEqual(set(ptr.indices), ptr.owned_pages)
        ptr.close()
        request.retire()
        self.assertEqual(scheduler.kv_pool.num_free_pages, scheduler.kv_pool.capacity)

    def test_cpu_reservation_sources_are_never_silently_evicted(self):
        scheduler = self.scheduler()
        scheduler.cpu_capacity = 10
        with patch.object(scheduler, "_current_cpu", return_value=9):
            self.assertFalse(scheduler._ensure_cpu_space(2))

    def test_repack_reserves_same_link_and_waits_for_its_start(self):
        scheduler = self.scheduler()
        scheduler.reservation_table.advance(scheduler.step_id)
        request = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        scheduler._requests[0] = request
        with patch.object(scheduler, "_submit_transfer") as submit:
            scheduler._start_repack()
            submit.assert_not_called()
            booking = scheduler.reservation_table.reservations[-1]
            scheduler.step_id = booking.start
            scheduler.reservation_table.advance(scheduler.step_id)
            scheduler._start_reload()  # A link-only booking is not a Verify.
            submit.assert_not_called()
            scheduler._start_repack()
            submit.assert_called_once_with(request, "repack")
            self.assertTrue(booking.materialized)
        request.retire()

    def test_prefill_retains_workspace_for_its_future_first_partition(self):
        scheduler = self.scheduler()
        request = Request([1] * 100, 128)
        request.request_id = 0
        request.attach_kv_pool(scheduler.kv_pool)
        scheduler._requests[0] = request
        self.assertGreaterEqual(scheduler._pack_workspace(), 11)
        request.retire()

    def constrained_swap(self):
        scheduler = self.scheduler()
        scheduler.kv_pool = fixtures.Pool(capacity=22)
        a = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 0)
        ptr = a.kv_cache_ptr
        original = tuple(ptr.indices)
        ptr._release(original[5:])
        ptr.indices = type(ptr.indices)("i", original[:5])
        ptr.hot_pages, ptr.tail_pages = original[:5], []
        ptr.phase, ptr.recent_count = "hot", 2
        ptr.cold_cpu = torch.empty(2, 7, 2, 1, 2, 128)
        b = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, 1)
        scheduler._requests = {0: a, 1: b}
        scheduler.reservation_table.advance(scheduler.step_id)
        return scheduler, a, b

    def test_future_admission_waits_for_real_page_release_before_allocating(self):
        scheduler, a, b = self.constrained_swap()
        # Isolate the page-transfer dependency from token/workspace reserves.
        with patch.object(scheduler, "_growth_pages", return_value=0), \
             patch.object(scheduler, "_pack_workspace", return_value=0):
            scheduler._place_reservations()
            table = scheduler.reservation_table
            reservation = table.reservations[0]
            source = table.reservations[-2]
            self.assertLess(scheduler.kv_pool.num_free_pages, a.kv_cache_ptr.cold_count)
            self.assertEqual(reservation.dependencies, (source.release_event,))
            scheduler.step_id = reservation.start
            table.advance(scheduler.step_id)
            with patch.object(scheduler, "_submit_transfer") as submit:
                scheduler._start_reload()
                submit.assert_not_called()
            self.assertEqual(scheduler.metrics["reservation_dependency_waits"], 1)

            # Submit B's real ownership transaction, with a controllable fence.
            ready = [False]
            event = SimpleNamespace(query=lambda: ready[0], synchronize=lambda: None)
            ptr = b.kv_cache_ptr
            ptr.prepare_resident_offload()
            transfer = PageTransfer(ptr._allocate(5), event, cold=torch.empty(2, 7, 2, 1, 2, 128))
            ptr._pending = transfer
            scheduler._transfers[1] = SimpleNamespace(direction="repack", request=b, transfer=transfer,
                                                       source_pages=12, cold_pages=7)
            table.materialize(-2, source_pages=12)
            scheduler._poll_transfers()
            self.assertEqual(scheduler.kv_pool.num_free_pages, 0)
            self.assertFalse(table.dependencies_ready(0))
            ready[0] = True
            scheduler._poll_transfers()
            self.assertEqual(scheduler.kv_pool.num_free_pages, 12)
            self.assertTrue(table.dependencies_ready(0))
            self.assertEqual(scheduler.metrics["reservation_release_commits"], 1)

            def allocate_restore(request, direction):
                self.assertEqual((request, direction), (a, "restore"))
                cache = request.kv_cache_ptr
                transaction = PageTransfer(cache._allocate(7), event, keepalive=cache.cold_cpu)
                cache._pending = transaction
                scheduler._transfers[0] = SimpleNamespace(direction=direction, request=a, transfer=transaction,
                                                          source_pages=5, cold_pages=7)

            with patch.object(scheduler, "_submit_transfer", side_effect=allocate_restore):
                scheduler._start_reload()
            self.assertEqual(scheduler.kv_pool.num_free_pages, 5)
            scheduler._poll_transfers()
            self.assertEqual(a.kv_cache_ptr.phase, "verify")
            self.assertTrue(table.reservations[0].materialized)
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()
        scheduler._cancel_reservation(0, "test_done")
        scheduler._cancel_reservation(-2, "test_done")
        self.assertEqual(scheduler.kv_pool.num_free_pages, 22)
        self.assertFalse(table.releases)

    def test_late_repack_revokes_dependent_and_keeps_its_source(self):
        scheduler, a, b = self.constrained_swap()
        with patch.object(scheduler, "_growth_pages", return_value=0), \
             patch.object(scheduler, "_pack_workspace", return_value=0):
            scheduler._place_reservations()
            table = scheduler.reservation_table
            source = table.reservations[-2]
            event = table.releases[source.release_event]
            scheduler._transfers[1] = SimpleNamespace(direction="repack")
            table.materialize(-2)
            scheduler.step_id = event.available_at
            table.advance(scheduler.step_id)
            scheduler._reconcile_reservations()
            self.assertNotIn(0, table.reservations)
            self.assertIn(-2, table.reservations)
            self.assertEqual(event.available_at, scheduler.step_id + 1)
            self.assertEqual(scheduler.metrics["reservation_dependency_replans"], 1)
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()


if __name__ == "__main__":
    unittest.main()
