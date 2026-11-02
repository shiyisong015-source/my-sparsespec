"""Calendar safety and Algorithm 1 placement, without CUDA dependencies."""
import random
import unittest

from serve.scheduler.reservation_table import ReservationTable


class CalendarTests(unittest.TestCase):
    def book(self, table, identity, size=100, pages=20, anchor=4, baseline=50, capacity=100):
        return table.reserve(identity, transfer_bytes=size, memory_pages=pages,
                             anchor=anchor, resident_pages=baseline, capacity_pages=capacity)

    def test_algorithm_candidate_order_and_bandwidth_staggering(self):
        table = ReservationTable(8, 100., 1.)
        self.assertEqual(list(table.candidates(4, 1, 7)), [4, 3, 5, 2, 6, 1, 7])
        self.assertEqual(self.book(table, 0).deadline, 4)
        self.assertEqual(self.book(table, 1).deadline, 3)
        self.assertEqual(self.book(table, 2).deadline, 5)

    def test_multiframe_reload_reserves_full_target_from_start(self):
        table = ReservationTable(8, 100., 1.)
        reservation = self.book(table, 0, size=150)
        self.assertEqual((reservation.start, reservation.deadline), (3, 4))
        windows = table.snapshot()["windows"]
        self.assertEqual([windows[i]["bandwidth_seconds"] for i in (3, 4)], [1., .5])
        self.assertEqual([windows[i]["reload_pages"] for i in (3, 4)], [20, 20])

    def test_hbm_staggers_even_when_link_has_room(self):
        table = ReservationTable(8, 100., 1.)
        self.assertEqual(self.book(table, 0, size=10, pages=40).deadline, 4)
        self.assertEqual(self.book(table, 1, size=10, pages=40).deadline, 3)

    def test_materialization_transfers_accounting_and_cancel_is_idempotent(self):
        table = ReservationTable(8, 100., 1.)
        self.book(table, 0)
        table.materialize(0)
        table.materialize(0)
        self.assertEqual(table.peak_memory_pages, 0)
        self.assertEqual(table.snapshot()["windows"][4]["bandwidth_seconds"], 1.)
        table.cancel(0)
        table.cancel(0)
        self.assertTrue(all(w["bandwidth_seconds"] == w["reload_pages"] == 0
                            for w in table.snapshot()["windows"]))

    def test_overdue_cancel_cannot_clear_wrapped_new_booking(self):
        table = ReservationTable(8, 100., 1.)
        self.book(table, 0)
        table.advance(8)
        current = self.book(table, 1)
        self.assertEqual(current.deadline, 12)
        table.cancel(0)
        window = next(w for w in table.snapshot()["windows"] if w["iteration"] == 12)
        self.assertEqual((window["reload_pages"], window["bandwidth_seconds"]), (20, 1.))

    def test_infeasible_admission_does_not_leave_partial_reservation(self):
        table = ReservationTable(4, 100., 1.)
        self.assertIsNone(self.book(table, 0, size=400))
        self.assertIsNone(self.book(table, 0, pages=51))
        self.assertFalse(table.reservations)
        self.assertEqual(table.peak_memory_pages, 0)

    def test_randomized_ring_accounting_matches_live_reservations(self):
        rng = random.Random(330)
        table = ReservationTable(16, 100., 1.)
        for identity in range(500):
            if rng.random() < .3:
                table.advance(table.now + rng.randint(1, 4))
            if table.reservations and rng.random() < .4:
                table.cancel(rng.choice(list(table.reservations)))
            reservation = self.book(table, identity, size=rng.randint(1, 450),
                                    pages=rng.randint(1, 35), anchor=rng.randint(1, 20))
            if reservation is not None and rng.random() < .2:
                table.materialize(identity)
            for window in table.snapshot()["windows"]:
                tick = window["iteration"]
                expected_bw = sum(seconds for item in table.reservations.values()
                                  for position, seconds in item.bandwidth_slots if position == tick)
                expected_hbm = sum(item.memory_pages for item in table.reservations.values()
                                   if not item.materialized and item.start <= tick <= item.deadline)
                self.assertAlmostEqual(window["bandwidth_seconds"], expected_bw)
                self.assertEqual(window["reload_pages"], expected_hbm)
                self.assertLessEqual(expected_bw, 1. + 1e-12)
                self.assertLessEqual(expected_hbm, 50)


class FutureCapacityTests(unittest.TestCase):
    def prepare(self):
        table = ReservationTable(8, 100., 1.)
        table.reserve(-1, transfer_bytes=100, memory_pages=0, anchor=1,
                      resident_pages=0, capacity_pages=100)
        release = table.plan_release(-1, pages=50, available_at=2)
        return table, release

    def reload(self, table, identity=0, size=100, pages=30, anchor=1, baseline=90):
        return table.reserve(identity, transfer_bytes=size, memory_pages=pages,
                             anchor=anchor, resident_pages=baseline, capacity_pages=100)

    def test_now_does_not_fit_but_future_release_makes_admission_possible(self):
        table, event = self.prepare()
        reservation = self.reload(table)
        self.assertEqual((reservation.start, reservation.deadline), (2, 2))
        self.assertEqual(reservation.dependencies, (event.event_id,))
        self.assertFalse(table.dependencies_ready(0))
        projection = table.memory_profile(90, 100)
        self.assertEqual(projection[1]["projected_free_pages"], 10)
        self.assertEqual(projection[2]["projected_free_pages"], 30)

    def test_commit_switches_from_forecast_to_actual_without_double_credit(self):
        table, event = self.prepare()
        self.reload(table)
        with self.assertRaises(RuntimeError):
            table.materialize(0)
        self.assertTrue(table.confirm_release(event.event_id))
        self.assertFalse(table.confirm_release(event.event_id))
        self.assertTrue(table.dependencies_ready(0))
        # Baseline fell from 90 to 40 after the real net 50-page release.
        self.assertEqual(table.memory_profile(40, 100)[2]["projected_used_pages"], 70)
        table.materialize(0)
        # Actual allocation raises baseline back to 70; no forecast is left.
        self.assertEqual(table.memory_profile(70, 100)[2]["projected_used_pages"], 70)
        table.cancel(-1)
        self.assertTrue(table.dependencies_ready(0))
        table.cancel(0)
        self.assertFalse(table.releases)

    def test_source_cancellation_revokes_dependent_reload_atomically(self):
        table, event = self.prepare()
        self.reload(table)
        self.assertEqual(table.cancel(-1), (-1, 0))
        self.assertFalse(table.reservations)
        self.assertFalse(table.releases)
        self.assertEqual(table.peak_memory_pages, 0)
        self.assertTrue(all(w["bandwidth_seconds"] == 0 for w in table.snapshot()["windows"]))

    def test_delayed_release_rebooks_dependent_and_preserves_source(self):
        table, event = self.prepare()
        self.reload(table)
        table.materialize(-1)
        self.assertEqual(table.defer_release(event.event_id, 5), (0,))
        self.assertIn(-1, table.reservations)
        replacement = self.reload(table, anchor=2)
        self.assertGreaterEqual(replacement.start, 5)
        self.assertEqual(replacement.dependencies, (event.event_id,))

    def test_whole_restore_cannot_borrow_a_release_halfway_through_copy(self):
        table, event = self.prepare()
        reservation = self.reload(table, size=150, anchor=2)
        self.assertEqual((reservation.start, reservation.deadline), (2, 3))

    def test_repack_temporary_hot_allocation_updates_release_amount(self):
        table, event = self.prepare()
        self.reload(table)
        before = table.memory_profile(90, 100)[2]["projected_used_pages"]
        # Ten new hot destination pages were allocated before submission.
        table.materialize(-1, source_pages=60)
        after = table.memory_profile(100, 100)[2]["projected_used_pages"]
        self.assertEqual(before, after)
        self.assertEqual(event.pages, 60)

    def test_reusing_source_key_does_not_alias_an_older_committed_event(self):
        table, old = self.prepare()
        self.reload(table)
        table.confirm_release(old.event_id)
        table.cancel(-1)
        table.advance(2)
        source = table.reserve(-1, transfer_bytes=100, memory_pages=0, anchor=1,
                               resident_pages=0, capacity_pages=100)
        new = table.plan_release(-1, pages=20, available_at=source.deadline + 1)
        self.assertNotEqual(old.event_id, new.event_id)
        replacement = self.reload(table, identity=1, pages=50, anchor=2, baseline=70)
        self.assertEqual(replacement.dependencies, (new.event_id,))
        self.assertEqual(table.cancel(-1), (-1, 1))
        self.assertIn(0, table.reservations)
        self.assertTrue(table.dependencies_ready(0))


if __name__ == "__main__":
    unittest.main()
