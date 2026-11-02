"""CPU data/ownership tests with controllable GPU-like completion events."""
import array
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
import serve.tests.test_compact_native as fixtures
from serve.request.kv_cache_ptr.distance_swap import SwapStorage
from serve.scheduler.distance_policy import Candidate, FuturePages, distance_to_verify, may_replace
from serve.scheduler.distance_swap_engine import DistanceSwapEngine


class Event:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class Transport:
    def __init__(self, pool):
        self.pool, self.events = pool, []
        self.downloaded = threading.Event()

    def selection(self, ptr, history):
        chosen = ptr.selected_indices_buffer_gpu[..., :ptr.len_selected_indices_gpu].clone()
        inverse = {p: i for i, p in enumerate(history)}
        positions = chosen.clone()
        for value in chosen.unique().tolist():
            positions[chosen == value] = inverse[value]
        return chosen, positions, Event()

    def cpu_buffer(self, count):
        shape = self.pool._physical_mem_gpu.shape
        return torch.empty(shape[0], count, *shape[2:])

    def hot(self, storage, pages, from_host=False):
        if from_host:
            source = torch.cat([c.cpu for c in storage.chunks], dim=1)
            chosen = storage.positions
        else:
            source = self.pool._physical_mem_gpu.clone()
            chosen = storage.chosen
        out = self.pool._physical_mem_gpu
        for layer in range(out.shape[0]):
            for head in range(out.shape[4]):
                out[layer, list(pages), :, :, head] = source[layer, chosen[layer, head].long(), :, :, head]
        storage.ptr.selected_indices_buffer_gpu[..., :len(pages)] = torch.tensor(pages)
        return Event()

    def download(self, storage, chunk):
        pages = storage.history[chunk.offset:chunk.offset + chunk.count]
        chunk.cpu.copy_(self.pool._physical_mem_gpu[:, pages])
        event = Event(False)
        self.events.append(event)
        self.downloaded.set()
        return event, None

    def upload(self, storage, chunk, pages):
        self.pool._physical_mem_gpu[:, list(pages)] = chunk.cpu
        event = Event(False)
        self.events.append(event)
        return event

    def synchronize(self):
        for event in self.events:
            event.synchronize()


class PriorityTests(unittest.TestCase):
    def test_verify_distance_uses_progress_and_native_initial_offset(self):
        self.assertEqual(distance_to_verify(8, 0, verify=False), 8)
        self.assertEqual(distance_to_verify(8, 7, verify=False), 1)
        self.assertEqual(distance_to_verify(8, 8, verify=False), 0)
        self.assertEqual(distance_to_verify(8, 8, verify=True), 0)
        for stride, index in ((0, 0), (8, -1), (8, 9)):
            with self.assertRaises(ValueError):
                distance_to_verify(stride, index, verify=False)

    def test_native_numpy_bucket_index_produces_a_python_integer_distance(self):
        distance = distance_to_verify(np.int64(8), np.int64(6), verify=False)
        self.assertEqual(distance, 2)
        self.assertIs(type(distance), int)
        for index in (1.5, True):
            with self.assertRaises(ValueError):
                distance_to_verify(8, index, verify=False)

    def test_distance_precedes_verify_tie_break(self):
        near_verify = Candidate(0, 0, True, 0)
        far_draft = Candidate(1, 3, False, 0)
        self.assertEqual(sorted([near_verify, far_draft], key=lambda c: c.victim_key)[0], far_draft)
        self.assertTrue(may_replace(near_verify, far_draft))
        self.assertFalse(may_replace(far_draft, near_verify))

    def test_equal_distance_evicts_verify(self):
        draft = Candidate(0, 0, False, 9)
        verify = Candidate(1, 0, True, 0)
        self.assertEqual(sorted([draft, verify], key=lambda c: c.victim_key)[0], verify)
        self.assertTrue(may_replace(draft, verify))
        self.assertFalse(may_replace(verify, draft))
        self.assertFalse(may_replace(draft, draft))

    def test_restore_wait_age_and_no_equal_priority_ping_pong(self):
        older = Candidate(0, 0, True, 1)
        younger = Candidate(1, 0, True, 2)
        self.assertEqual(sorted([younger, older], key=lambda c: c.restore_key)[0], older)
        self.assertTrue(may_replace(older, younger))
        self.assertFalse(may_replace(younger, older))

    def test_future_release_is_unique_and_cancelable(self):
        ledger = FuturePages()
        producer = (1, 3, 0)
        ledger.claim(producer, 7, 4)
        self.assertEqual(ledger.pages(7), 4)
        with self.assertRaises(ValueError):
            ledger.claim(producer, 8, 4)
        with self.assertRaises(ValueError):
            ledger.consume(producer, 8)
        ledger.cancel(7)
        self.assertFalse(ledger.claims)


class StorageHelpers:
    def storage(self, capacity=64, identity=0, pool=None):
        pool = pool or fixtures.Pool(capacity=capacity)
        request = fixtures.CPUAsyncTransfers.request(pool, identity)
        transport = Transport(pool)
        ptr = request.kv_cache_ptr
        storage = SwapStorage(ptr, transport, 4, identity + 1)
        return request, storage, transport

    def back_up(self, storage, transport):
        for chunk in storage.chunks:
            chunk.cpu = transport.cpu_buffer(chunk.count)
            self.assertTrue(storage.start_down(chunk))
            chunk.job.ready.ready = True
            storage.poll()


class StorageTests(StorageHelpers, unittest.TestCase):
    def test_chunk_release_waits_event_and_keeps_draft_view(self):
        request, storage, transport = self.storage()
        ptr = storage.ptr
        original = set(storage.history)
        storage.start_hot()
        storage.poll()
        hot = storage.hot_pages
        chunk = storage.chunks[0]
        chunk.cpu = transport.cpu_buffer(chunk.count)
        free = ptr.kv_pool.num_free_pages
        storage.start_down(chunk)
        storage.poll()
        self.assertEqual(ptr.kv_pool.num_free_pages, free)
        chunk.job.ready.ready = True
        storage.poll()
        self.assertEqual(ptr.kv_pool.num_free_pages, free + chunk.count)
        self.assertEqual(storage.hot_pages, hot)
        self.assertFalse(storage.complete)
        self.assertTrue(set(hot).isdisjoint(original))
        self.assertTrue(set(ptr.indices).issubset(ptr.owned_pages))
        ptr.close()
        request.retire()
        self.assertEqual(ptr.kv_pool.num_free_pages, ptr.kv_pool.capacity)

    def test_restore_data_and_tail_append_survive_partial_exchange(self):
        request, storage, transport = self.storage()
        ptr = storage.ptr
        physical = ptr.kv_pool._physical_mem_gpu
        original = physical[:, list(storage.history) + list(storage.recent)].clone()
        storage.start_hot()
        storage.poll()
        self.back_up(storage, transport)
        ptr.allocate_tokens(1)
        tail = ptr.tail_pages[0]
        physical[:, tail] = 123
        storage.publish()
        with self.assertRaises(RuntimeError):
            ptr.freeze_verify_view()
        for chunk in storage.chunks:
            storage.start_up(chunk)
            self.assertFalse(storage.complete)
            chunk.job.ready.ready = True
            storage.poll()
        self.assertTrue(storage.complete)
        self.assertEqual(ptr.indices[-1], tail)
        torch.testing.assert_close(physical[:, list(ptr.indices)[:-1]], original)
        self.assertTrue((physical[:, tail] == 123).all())
        ptr.freeze_verify_view()
        ptr.close()
        request.retire()

    def test_zero_free_bootstrap_uses_saved_history_to_build_hot(self):
        request, storage, transport = self.storage(capacity=12)
        self.assertEqual(storage.ptr.kv_pool.num_free_pages, 0)
        self.assertFalse(storage.start_hot())
        self.back_up(storage, transport)
        self.assertTrue(storage.start_hot())
        storage.poll()
        self.assertTrue(storage.hot_ready)
        physical = storage.ptr.kv_pool._physical_mem_gpu
        source = torch.cat([c.cpu for c in storage.chunks], dim=1)
        for layer in range(2):
            for head in range(2):
                torch.testing.assert_close(physical[layer, list(storage.hot_pages), :, :, head],
                    source[layer, storage.positions[layer, head].long(), :, :, head])
        storage.ptr.close()
        request.retire()

    def test_saved_cpu_copy_allows_release_without_retransfer(self):
        request, storage, transport = self.storage()
        storage.start_hot()
        storage.poll()
        self.back_up(storage, transport)
        chunk = storage.chunks[0]
        storage.start_up(chunk)
        chunk.job.ready.ready = True
        storage.poll()
        transfers = len(transport.events)
        self.assertTrue(storage.start_down(chunk))
        self.assertIsNone(chunk.job)
        self.assertEqual(len(transport.events), transfers)
        storage.ptr.close()
        request.retire()


class EngineTests(StorageHelpers, unittest.TestCase):
    def engine(self, capacity=64):
        pool = fixtures.Pool(capacity=capacity)
        transport = Transport(pool)
        engine = DistanceSwapEngine(pool, transport, threading.RLock(), 128, 4)
        engine.reserve = 0
        return engine, pool, transport

    def add(self, engine, identity, length=12):
        request = fixtures.CPUAsyncTransfers.request(engine.pool, identity, length=length)
        storage = SwapStorage(request.kv_cache_ptr, engine.transport, 4, identity + 1)
        storage.start_hot()
        storage.poll()
        engine.storages[identity] = storage
        return request, storage

    def test_future_plan_exists_before_victim_finishes_and_uploads_per_chunk(self):
        engine, pool, transport = self.engine(capacity=40)
        a, sa = self.add(engine, 0)
        b, sb = self.add(engine, 1)
        self.back_up(sa, transport)
        # Consume all currently free pages to force an actual page exchange.
        held = tuple(pool.alloc_page() for _ in range(pool.num_free_pages))
        engine.publish([Candidate(0, 0, True, 0), Candidate(1, 3, False, 1)], 0)
        engine.tick()
        self.assertEqual(engine.target, 0)
        self.assertEqual(pool.num_free_pages, 0)
        self.assertGreater(engine.claims.pages(0), 0)
        self.assertTrue(any(c.job is not None and c.job.direction == "down" for c in sb.chunks))
        # One finished chunk is enough to start uploading, with the rest of
        # the victim still resident / in flight.
        for event in transport.events:
            event.ready = True
        engine.tick()
        self.assertTrue(any(c.job is not None and c.job.direction == "up" for c in sa.chunks))
        self.assertFalse(sa.complete)
        for _ in range(20):
            for event in transport.events:
                event.ready = True
            engine.tick()
            if sa.complete:
                break
        self.assertTrue(sa.complete)
        engine.stop()
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()
        for page in held:
            pool.free_page(page)
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_sealed_and_inflight_verify_are_not_victims(self):
        engine, pool, transport = self.engine()
        request, storage = self.add(engine, 0)
        engine.publish([Candidate(0, 0, True, 0)], 0)
        engine.sealed.add(0)
        self.assertFalse(list(engine._victims()))
        engine.sealed.clear()
        fence = Event(False)
        engine.compute[0] = fence, True
        self.assertFalse(list(engine._victims()))
        fence.ready = True
        engine.poll()
        self.assertFalse(list(engine._victims()))  # CPU still needs the Verify view.
        engine.compute.pop(0)  # Rejection/selection handoff has published a new epoch.
        self.assertTrue(list(engine._victims()))
        request.kv_cache_ptr.close()
        request.retire()

    def test_retire_cancels_future_credit_but_preserves_inflight_job(self):
        engine, pool, transport = self.engine()
        request, storage = self.add(engine, 0)
        engine.publish([Candidate(0, 0, True, 0)], 0)
        engine.pressure = pool.capacity
        engine.tick()
        job = next(c.job for c in storage.chunks if c.job is not None)
        engine.retire(0)
        self.assertFalse(engine.candidates)
        self.assertFalse(job.ready.query())
        engine.stop()
        self.assertFalse(storage.busy)
        request.kv_cache_ptr.close()
        request.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_background_worker_progresses_without_forward(self):
        engine, pool, transport = self.engine()
        request, storage = self.add(engine, 0)
        engine.publish([Candidate(0, 0, True, 0)], 0)
        engine.pressure = pool.capacity
        engine.start()
        self.assertTrue(transport.downloaded.wait(2))
        engine.stop()
        self.assertLess(len([p for p in storage.history if p is not None]), len(storage.history))
        self.assertFalse(engine.thread.is_alive())
        engine.stop()  # Cleanup is idempotent.
        request.kv_cache_ptr.close()
        request.retire()

    def test_worker_failure_reaches_compute_owner(self):
        engine, _, _ = self.engine()
        failed = threading.Event()
        def tick():
            failed.set()
            raise ValueError("injected transfer failure")
        engine.tick = tick
        engine.start()
        self.assertTrue(failed.wait(2))
        engine.stop()
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            engine.check()


    def test_cpu_capacity_is_a_real_limit(self):
        engine, pool, transport = self.engine()
        request, storage = self.add(engine, 0)
        engine.cpu_capacity = 0
        engine.publish([Candidate(0, 0, True, 0)], 0)
        engine.pressure = pool.capacity
        engine.tick()
        self.assertFalse(storage.busy)
        self.assertEqual(storage.cpu_pages, 0)
        self.assertGreater(engine.metrics["swap_cpu_waits"], 0)
        request.kv_cache_ptr.close()
        request.retire()

    def test_draft_lease_protects_hot_view_without_blocking_history_eviction(self):
        engine, _, _ = self.engine()
        request, storage = self.add(engine, 0)
        engine.publish([Candidate(0, 2, False, 0)], 0)
        engine.sealed.add(0)
        engine.compute[0] = Event(False), False
        self.assertTrue(list(engine._victims()))
        request.kv_cache_ptr.close()
        request.retire()

    def test_growth_extends_future_plan_and_draft_does_not_consume_restore_lease(self):
        engine, pool, transport = self.engine(capacity=60)
        a, sa = self.add(engine, 0)
        b, sb = self.add(engine, 1)
        c, sc = self.add(engine, 2)
        self.back_up(sa, transport)
        held = tuple(pool.alloc_page() for _ in range(pool.num_free_pages))
        engine.publish([Candidate(0, 1, False, 0), Candidate(1, 3, False, 1),
                        Candidate(2, 4, False, 2)], 0)
        engine.tick()
        initial = engine.claims.pages(0)
        self.assertEqual(initial, sa.missing)
        engine.compute[0] = Event(), False
        engine.reserve = 2
        engine.tick()
        self.assertTrue(sa.pinned)
        self.assertGreater(engine.claims.pages(0), initial)
        engine.stop()
        for request in (a, b, c):
            request.kv_cache_ptr.close()
            request.retire()
        for page in held:
            pool.free_page(page)
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_nearer_verify_preempts_prefetch_at_chunk_completion(self):
        engine, pool, transport = self.engine()
        a, sa = self.add(engine, 0)
        b, sb = self.add(engine, 1)
        self.back_up(sa, transport)
        self.back_up(sb, transport)
        engine.publish([Candidate(0, 4, False, 0)], 0)
        engine.tick()
        first = sa.chunks[0].job
        self.assertIsNotNone(first)
        engine.publish([Candidate(0, 4, False, 0), Candidate(1, 0, True, 1)], 0)
        engine.tick()
        self.assertEqual(engine.target, 0)  # Queued work is not freed/canceled.
        self.assertIs(sa.chunks[0].job, first)
        first.ready.ready = True
        engine.tick()
        self.assertEqual(engine.target, 1)
        self.assertTrue(any(c.job is not None and c.job.direction == "up" for c in sb.chunks))
        self.assertTrue(set(first.pages).issubset(sa.ptr.owned_pages))
        self.assertFalse(any(c.job is not None for c in sa.chunks))
        engine.stop()
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_infeasible_nearer_verify_does_not_repeatedly_cancel_prefetch(self):
        engine, pool, transport = self.engine(capacity=60)
        a, sa = self.add(engine, 0)
        b, sb = self.add(engine, 1, length=22)
        self.back_up(sa, transport)
        self.back_up(sb, transport)
        held = tuple(pool.alloc_page() for _ in range(pool.num_free_pages - 10))
        engine.publish([Candidate(0, 4, False, 0)], 0)
        engine.tick()
        sa.chunks[0].job.ready.ready = True
        engine.publish([Candidate(0, 4, False, 0), Candidate(1, 0, True, 1)], 0)
        engine.tick()
        self.assertEqual(engine.target, 0)
        self.assertEqual(engine.metrics["swap_plans"], 1)
        self.assertIsNotNone(sa.chunks[1].job)
        engine.stop()
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()
        for page in held:
            pool.free_page(page)
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_draft_progress_invalidates_an_unstarted_victim(self):
        engine, pool, transport = self.engine(capacity=40)
        a, sa = self.add(engine, 0)
        b, sb = self.add(engine, 1)
        self.back_up(sa, transport)
        held = tuple(pool.alloc_page() for _ in range(pool.num_free_pages))
        engine.publish([Candidate(0, 2, False, 0), Candidate(1, 6, False, 1)], 0)
        self.assertTrue(engine._new_plan())
        self.assertTrue(engine.producers)
        engine.publish([Candidate(0, 2, False, 0), Candidate(1, 1, False, 1)], 0)
        engine.tick()
        self.assertIsNone(engine.target)
        self.assertFalse(engine.claims.claims)
        self.assertFalse(engine.producers)
        self.assertFalse(any(c.job is not None for c in sb.chunks))
        engine.stop()
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()
        for page in held:
            pool.free_page(page)
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_replan_retains_queued_h2d_destinations_in_capacity_accounting(self):
        engine, pool, transport = self.engine(capacity=20)
        request, storage = self.add(engine, 0)
        self.back_up(storage, transport)
        held = tuple(pool.alloc_page() for _ in range(pool.num_free_pages - storage.missing))
        engine.publish([Candidate(0, 0, True, 0)], 0)
        engine.tick()
        first = storage.chunks[0].job
        engine._cancel_plan()
        engine.tick()
        self.assertEqual(engine.target, 0)
        self.assertIs(storage.chunks[0].job, first)
        self.assertTrue(set(first.pages).issubset(storage.ptr.owned_pages))
        self.assertFalse(engine.claims.claims)
        engine.stop()
        request.kv_cache_ptr.close()
        request.retire()
        for page in held:
            pool.free_page(page)
        self.assertEqual(pool.num_free_pages, pool.capacity)


# CPU scheduler integration, preserving the actual admission/retirement code.
with patch("torch.cuda.Stream", return_value=SimpleNamespace()):
    from serve.scheduler.distance_swap_scheduler import DistanceSwapSpecScheduler


class SchedulerTests(StorageHelpers, unittest.TestCase):
    def scheduler(self):
        base = fixtures.CPUDynamicAdmission.scheduler(capacity=128, budget=16)
        scheduler = DistanceSwapSpecScheduler.__new__(DistanceSwapSpecScheduler)
        scheduler.__dict__.update(base.__dict__)
        scheduler._residency_lock = threading.RLock()
        scheduler._swap = DistanceSwapEngine(scheduler.kv_pool, Transport(scheduler.kv_pool),
                                             scheduler._residency_lock, 128, 4)
        scheduler._service_order, scheduler._service_sequence, scheduler._swap_generation = {}, 0, 0
        scheduler._args["tp_ranks"] = 1
        return scheduler

    def add(self, scheduler, identity):
        request = fixtures.CPUAsyncTransfers.request(scheduler.kv_pool, identity)
        storage = SwapStorage(request.kv_cache_ptr, scheduler._swap.transport, 4, identity + 1)
        storage.start_hot()
        storage.poll()
        scheduler._requests[identity] = request
        scheduler._swap.storages[identity] = storage
        return request, storage

    def test_ready_verify_is_nearer_than_draft_even_when_its_history_is_missing(self):
        scheduler = self.scheduler()
        draft, sd = self.add(scheduler, 0)
        verify, sv = self.add(scheduler, 1)
        verify.status = fixtures.Request.ReqExecType.VERIFY
        verify.cur_spec_idx = scheduler.spec_stride
        scheduler._service_order[1] = 100
        self.back_up(sv, scheduler._swap.transport)
        scheduler._publish_candidates()
        candidates = scheduler._swap.candidates
        self.assertEqual(candidates[0].distance, scheduler.spec_stride - draft.cur_spec_idx)
        self.assertEqual(candidates[1].distance, 0)
        self.assertTrue(candidates[1].verify)
        self.assertLess(candidates[1].restore_key, candidates[0].restore_key)
        self.assertLess(candidates[0].victim_key, candidates[1].victim_key)
        for request in (draft, verify):
            request.kv_cache_ptr.close()
            request.retire()

    def test_query_budget_and_queue_order_do_not_change_verify_distance(self):
        scheduler = self.scheduler()
        a, _ = self.add(scheduler, 0)
        b, _ = self.add(scheduler, 1)
        for request in (a, b):
            request.status = fixtures.Request.ReqExecType.VERIFY
            request.cur_spec_idx = scheduler.spec_stride
            request.len_draft_probs = 2
        scheduler._publish_candidates()
        self.assertEqual([scheduler._swap.candidates[i].distance for i in (0, 1)], [0, 0])
        scheduler.token_budget = 4  # Only one Verify fits a real batch.
        scheduler._service_order.update({0: 100, 1: -1})
        scheduler._publish_candidates()
        self.assertEqual([scheduler._swap.candidates[i].distance for i in (0, 1)], [0, 0])
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()

    def test_same_draft_count_with_different_initial_offsets_has_different_distance(self):
        scheduler = self.scheduler()
        scheduler.spec_stride = 8
        a, _ = self.add(scheduler, 0)
        b, _ = self.add(scheduler, 1)
        a.initial_idx, a.cur_spec_idx, a.len_draft_probs = np.int64(6), np.int64(7), 1
        b.initial_idx, b.cur_spec_idx, b.len_draft_probs = np.int64(3), np.int64(4), 1
        scheduler._service_order.update({0: 100, 1: -1})
        scheduler._publish_candidates()
        candidates = scheduler._swap.candidates
        self.assertEqual((candidates[0].distance, candidates[1].distance), (1, 4))
        self.assertLess(candidates[0].restore_key, candidates[1].restore_key)
        self.assertLess(candidates[1].victim_key, candidates[0].victim_key)
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()

    def test_only_that_requests_draft_progress_reduces_its_distance(self):
        scheduler = self.scheduler()
        scheduler.spec_stride = 8
        a, _ = self.add(scheduler, 0)
        b, _ = self.add(scheduler, 1)
        scheduler._publish_candidates()
        before = {i: c.distance for i, c in scheduler._swap.candidates.items()}
        scheduler.step_id += 100
        a.cur_spec_idx += 1
        scheduler._publish_candidates()
        self.assertEqual(scheduler._swap.candidates[0].distance, before[0] - 1)
        self.assertEqual(scheduler._swap.candidates[1].distance, before[1])
        a.cur_spec_idx, a.status = 8, fixtures.Request.ReqExecType.VERIFY
        scheduler._publish_candidates()
        self.assertEqual(scheduler._swap.candidates[0].distance, 0)
        for request in (a, b):
            request.kv_cache_ptr.close()
            request.retire()

    def test_draft_computes_during_down_but_verify_waits_full_restore(self):
        scheduler = self.scheduler()
        request, storage = self.add(scheduler, 0)
        chunk = storage.chunks[0]
        chunk.cpu = scheduler._swap.transport.cpu_buffer(chunk.count)
        storage.start_down(chunk)
        scheduler.admit_requests([])
        self.assertEqual(scheduler.active_requests, [request])
        self.assertIn(0, scheduler._swap.sealed)
        fence = Event(False)
        scheduler._on_compute_submitted([request], [request.status], fence)
        self.assertNotIn(0, scheduler._swap.sealed)
        self.assertFalse(scheduler._swap.compute[0][1])
        request.status = fixtures.Request.ReqExecType.VERIFY
        self.assertFalse(scheduler._ready(request))
        scheduler._swap.stop()
        request.kv_cache_ptr.close()
        request.retire()

    def test_retirement_waits_for_transfer_even_after_compute_finished(self):
        scheduler = self.scheduler()
        request, storage = self.add(scheduler, 0)
        chunk = storage.chunks[0]
        chunk.cpu = scheduler._swap.transport.cpu_buffer(chunk.count)
        storage.start_down(chunk)
        scheduler._compute_fences[0] = Event()
        scheduler._defer_retire(request)
        scheduler._retired[0].control_done = Event()
        scheduler._poll_retired()
        self.assertFalse(storage.ptr._closed)
        self.assertTrue(scheduler._retired)
        chunk.job.ready.ready = True
        scheduler._poll_retired()
        self.assertFalse(scheduler._retired)
        self.assertFalse(scheduler._swap.storages)
        self.assertTrue(storage.ptr._closed)
        self.assertEqual(scheduler.kv_pool.num_free_pages, scheduler.kv_pool.capacity)

    def test_next_selection_retires_old_hot_only_after_control_event(self):
        scheduler = self.scheduler()
        request, old = self.add(scheduler, 0)
        ptr = request.kv_cache_ptr
        old_hot = old.hot_pages
        # Simulate the real native rejection/selection boundary, with actual
        # per-head selected metadata. It is generated from the full raw view.
        fixtures.CPUOwnership().select(ptr)
        ready = Event(False)
        original_selection = scheduler._swap.transport.selection
        def selection(*args):
            chosen, positions, _ = original_selection(*args)
            return chosen, positions, ready
        scheduler._swap.transport.selection = selection
        scheduler.active_requests = [request]
        with patch("serve.scheduler.distance_swap_scheduler.SpecScheduler.update_requests", return_value=[]):
            scheduler.update_requests()
        self.assertTrue(set(old_hot).issubset(ptr.owned_pages))
        scheduler._swap.poll()
        self.assertTrue(set(old_hot).issubset(ptr.owned_pages))
        ready.ready = True
        scheduler._swap.poll()
        self.assertTrue(set(old_hot).isdisjoint(ptr.owned_pages))
        scheduler._swap.stop()
        ptr.close()
        request.retire()
        self.assertEqual(scheduler.kv_pool.num_free_pages, scheduler.kv_pool.capacity)

    def test_initial_dense_draft_cannot_be_evicted_as_cold_history(self):
        scheduler = self.scheduler()
        request = fixtures.Request([1] * 13, 8)
        request.request_id = 0
        full = fixtures.FullCachePtr(scheduler.kv_pool,
            array.array("i", [scheduler.kv_pool.alloc_page() for _ in range(12)]), 12)
        ptr, _ = fixtures.LocalPtr.from_kv_cache(full)
        request.kv_cache_ptr, request.status = ptr, fixtures.Request.ReqExecType.DRAFT
        scheduler._requests[0] = request
        scheduler.active_requests = [request]
        with patch("serve.scheduler.distance_swap_scheduler.SpecScheduler.update_requests", return_value=[]):
            scheduler.update_requests()
        storage = scheduler._swap.storages[0]
        self.assertEqual(storage.history, [])
        self.assertEqual(storage.recent, tuple(ptr.indices))
        self.assertEqual(ptr.len_loading_indices, 12)
        ptr.close()
        request.retire()

    def test_prefill_admission_uses_current_chunks_without_final_context_reserve(self):
        scheduler = self.scheduler()
        scheduler._args["chunk_size"] = 8
        pending = [fixtures.Request([1] * 54, 32000) for _ in range(2)]
        for i, request in enumerate(pending):
            request.request_id = i
        scheduler.admit_requests(pending)
        self.assertFalse(pending)
        self.assertEqual(len(scheduler.active_requests), 2)
        self.assertEqual([r.scheduled_tokens for r in scheduler.active_requests], [8, 8])
        for request in scheduler.active_requests:
            request.retire()

    def test_prefill_admission_still_requires_real_current_chunk_pages(self):
        scheduler = self.scheduler()
        scheduler._args["chunk_size"] = 8
        occupied=[scheduler.kv_pool.alloc_page() for _ in range(scheduler.kv_pool.capacity-8)]
        pending=[fixtures.Request([1] * 54, 32000) for _ in range(2)]
        for identity,request in enumerate(pending):
            request.request_id=identity
        scheduler.admit_requests(pending)
        self.assertEqual(len(scheduler.active_requests),1)
        self.assertEqual(len(pending),1)
        self.assertEqual(scheduler.active_requests[0].scheduled_tokens,8)
        self.assertGreater(scheduler.metrics['admission_waits'],0)
        for request in scheduler.active_requests:
            request.retire()
        for page in occupied:
            scheduler.kv_pool.free_page(page)

    def test_single_prefill_is_not_double_counted_against_its_verify(self):
        scheduler = self.scheduler()
        pending = [fixtures.Request([1] * 70, 8)]
        pending[0].request_id = 0
        scheduler.admit_requests(pending)
        self.assertFalse(pending)
        self.assertEqual(len(scheduler.active_requests), 1)
        scheduler.active_requests[0].retire()

    def test_tp_and_conflicting_policies_fail_before_cuda_allocation(self):
        for options in (dict(tp_ranks=2), dict(tp_ranks=1, compact_reservation=True),
                        dict(tp_ranks=1, distance_swap_chunk_pages=0)):
            with self.assertRaises(ValueError):
                DistanceSwapSpecScheduler(None, None, **options)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GPUStorageTests(unittest.TestCase):
    def test_real_chunked_data_roundtrip_with_draft_tail(self):
        from serve.request.kv_cache_ptr.distance_swap import CUDASwapTransport
        pool = fixtures.Pool("cuda:0", capacity=64)
        request = fixtures.CPUAsyncTransfers.request(pool, 0)
        ptr = request.kv_cache_ptr
        source = pool._physical_mem_gpu[:, list(ptr.indices)].clone()
        transport = CUDASwapTransport(pool)
        storage = SwapStorage(ptr, transport, 4, 1)
        try:
            storage.start_hot()
            transport.synchronize()
            storage.poll()
            for chunk in storage.chunks:
                chunk.cpu = transport.cpu_buffer(chunk.count)
                storage.start_down(chunk)
                chunk.job.ready.synchronize()
                storage.poll()
            ptr.allocate_tokens(1)
            pool._physical_mem_gpu[:, ptr.tail_pages[-1]] = 17
            for chunk in storage.chunks:
                storage.start_up(chunk)
                chunk.job.ready.synchronize()
                storage.poll()
            torch.testing.assert_close(pool._physical_mem_gpu[:, list(ptr.indices)[:-1]], source)
            self.assertTrue((pool._physical_mem_gpu[:, ptr.indices[-1]] == 17).all())
            ptr.freeze_verify_view()
        finally:
            transport.synchronize()
            ptr.close()
            request.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)

    def test_real_hot_bootstrap_from_multiple_host_chunks(self):
        from serve.request.kv_cache_ptr.distance_swap import CUDASwapTransport
        pool = fixtures.Pool("cuda:0", capacity=12)
        request = fixtures.CPUAsyncTransfers.request(pool, 0)
        transport = CUDASwapTransport(pool)
        ptr = request.kv_cache_ptr
        expected = fixtures.selected_copy(pool._physical_mem_gpu, ptr.selected_indices_buffer_gpu[
            ..., :ptr.len_selected_indices_gpu]).clone()
        storage = SwapStorage(ptr, transport, 4, 1)
        try:
            self.assertFalse(storage.start_hot())
            for chunk in storage.chunks:
                chunk.cpu = transport.cpu_buffer(chunk.count)
                storage.start_down(chunk)
                chunk.job.ready.synchronize()
                storage.poll()
            self.assertTrue(storage.start_hot())
            transport.synchronize()
            storage.poll()
            torch.testing.assert_close(pool._physical_mem_gpu[:, list(storage.hot_pages)], expected)
        finally:
            transport.synchronize()
            ptr.close()
            request.retire()
        self.assertEqual(pool.num_free_pages, pool.capacity)


if __name__ == "__main__":
    unittest.main()
