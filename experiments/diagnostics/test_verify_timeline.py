"""Regression tests for accounting, prefetch and capture-boundary censoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_timeline import VerifyTimeline, analyze_verify_timeline


class VerifyTimelineTest(unittest.TestCase):
    def test_prefetch_and_disjoint_waits_with_censored_neighbor(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            clock = [0]
            def tick(ms):
                clock[0] = int(ms * 1e6)
            with patch("verify_timeline.time.perf_counter_ns", side_effect=lambda:clock[0]):
                t = VerifyTimeline(case/"verify-timeline.rank0.jsonl", 0)
                tick(100); t.start(clock[0])
                tick(110); t.observe(1, "DRAFT")
                tick(120); t.note("restore_submit_begin", 1)
                tick(130); t.note("restore_pages_allocated", 1)
                tick(140); t.note("restore_submitted", 1)
                tick(150); t.note("restore_ready_observed", 1)
                tick(160); t.note("restore_page_table_committed", 1)
                tick(200); t.observe(1, "VERIFY")
                tick(220); t.wait(1, "reservation_deadline")
                tick(270); t.wait(1, "batch_preparation")
                tick(300); t.submit_verify(1, 5)
                tick(310); t.emit("restore_device_ready", 1, cycle=0, gpu_ready_ms=25.)
                tick(320); t.result(1)
                tick(400); t.observe(2, "VERIFY", carry_in=True)
                tick(410); t.wait(2, "gpu_capacity")
                tick(500); t.stop(clock[0]); t.close()
            (case/"components.rank0.jsonl").write_text(json.dumps(dict(
                component="forward", producer_step=5, gpu_start_ms=40., gpu_end_ms=45.)) + "\n")
            summary = analyze_verify_timeline(case, plots=False)
            self.assertEqual(summary["complete_uncensored_cycles"], 1)
            self.assertEqual(summary["left_censored_cycles"], 1)
            self.assertEqual(summary["right_censored_cycles"], 1)
            self.assertEqual(summary["host_need_to_submit_ms"]["mean"], 100.)
            self.assertEqual(sum(summary["window_request_wait_ms_by_state"].values()), 200.)
            self.assertEqual(summary["prefetch_committed_before_need"], 1)
            self.assertEqual(summary["device_ready_to_forward_event_ms"]["mean"], 15.)
            rows = [json.loads(line) for line in (case/"verify-wait-cycles.rank0.jsonl").read_text().splitlines()]
            self.assertTrue(rows[0]["destination_allocated_before_need"])
            self.assertEqual(rows[0]["wait_ms"], dict(await_admission=20., reservation_deadline=50., batch_preparation=30.))

    def test_independent_request_waits_are_not_system_elapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            with patch("verify_timeline.time.perf_counter_ns", return_value=100_000_000):
                t = VerifyTimeline(case/"verify-timeline.rank0.jsonl", 0)
                t.start(0)
                t.observe(1, "VERIFY"); t.observe(2, "VERIFY")
                t.observe(3, "VERIFY")
                t.observe(4, "DRAFT")  # Prefetch-only is not a waiting cycle.
                t.stop(200_000_000); t.close()
            summary = analyze_verify_timeline(case, plots=False)
            self.assertEqual(summary["observed_verify_cycles"], 3)
            self.assertEqual(summary["window_seconds"], .2)
            self.assertEqual(sum(summary["window_request_wait_ms_by_state"].values()), 300.)

    def test_no_events_after_window_and_duplicate_submit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"verify-timeline.rank0.jsonl"
            t = VerifyTimeline(path, 0)
            t.start(0); t.observe(1, "VERIFY"); t.submit_verify(1, 1)
            with self.assertRaises(RuntimeError):
                t.submit_verify(1, 1)
            t.stop(100)
            t.observe(2, "VERIFY"); t.note("restore_submitted", 2); t.result(1)
            t.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "window_end")
            self.assertFalse(any(e.get("request_id") == 2 for e in events))


if __name__ == "__main__":
    unittest.main()
