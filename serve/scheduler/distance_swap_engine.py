"""TP1 event-driven residency owner. No model/sampling state is mutated here."""
from collections import deque
from contextlib import nullcontext
import threading
import time

from serve.scheduler.distance_policy import DISTANCE_SCOPE, FuturePages, may_replace


class DistanceSwapEngine:
    def __init__(self, pool, transport, lock, cpu_capacity, chunk_pages,
                 device_context=nullcontext, trace=None):
        self.pool, self.transport, self.lock = pool, transport, lock
        self.cpu_capacity, self.chunk_pages = cpu_capacity, chunk_pages
        self.device_context = device_context
        self.trace = trace
        self.sequence = 0
        self.storages, self.candidates = {}, {}
        self.sealed, self.compute = set(), {}
        self.garbage = []
        self.retired = set()
        self.target = None
        self.producers = deque()
        self.claims = FuturePages()
        self.reserve = 1
        self.pressure = 0
        self.error = None
        self.wake, self.stopping = threading.Event(), threading.Event()
        self.thread = None
        self._stopped = False
        self.metrics = dict(swap_plans=0, swap_down_chunks=0, swap_up_chunks=0,
                            swap_d2h_pages=0, swap_h2d_pages=0, swap_future_pages_peak=0,
                            swap_cpu_waits=0, swap_worker_ticks=0)

    def _event(self, event, **fields):
        if self.trace is not None:
            self.sequence += 1
            self.trace(dict(event=event, sequence=self.sequence, cpu_ns=time.perf_counter_ns(),
                            distance_scope=DISTANCE_SCOPE, **fields))

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Transfer engine already started")
        self.thread = threading.Thread(target=self._run, name="distance-swap", daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Distance-swap transfer worker failed") from self.error

    def _run(self):
        try:
            with self.device_context():
                while not self.stopping.is_set():
                    with self.lock:
                        active = self.tick()
                    # Event.query has no blocking completion primitive. Poll bounded
                    # in-flight work; idle workers wake on new compute snapshots.
                    self.wake.wait(.001 if active else .02)
                    self.wake.clear()
        except BaseException as exc:
            self.error = exc
            self.wake.set()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join()
        # Join first: no transfer can be submitted after synchronization.
        self.transport.synchronize()
        with self.lock:
            self.poll()
            self.producers.clear()
            self.claims.claims.clear()
            self.target = None

    @property
    def cpu_pages(self):
        return sum(s.cpu_pages for s in self.storages.values())

    def publish(self, candidates, reserve):
        self.candidates = {c.identity: c for c in candidates}
        self.reserve = reserve
        self.wake.set()

    def retire(self, identity):
        self.retired.add(identity)
        self.candidates.pop(identity, None)
        self.sealed.discard(identity)
        if self.target == identity:
            self._cancel_plan()

    def forget(self, identity):
        storage = self.storages.pop(identity, None)
        if storage is not None and storage.busy:
            raise RuntimeError("Cannot forget in-flight storage")
        self.retired.discard(identity)
        self.compute.pop(identity, None)
        self.candidates.pop(identity, None)

    def _protected(self, identity):
        if identity in self.retired or identity not in self.candidates:
            return True
        if identity in self.sealed and self.candidates[identity].verify:
            return True
        state = self.compute.get(identity)
        # A completed Verify still feeds CPU rejection/selection. Its full
        # view is leased until that control handoff publishes the next epoch.
        return state is not None and state[1]

    def poll(self):
        for identity, storage in list(self.storages.items()):
            events = {c.offset: c.job.ready for c in storage.chunks if c.job is not None}
            hot_event = storage.hot_event
            for direction, chunk in storage.poll():
                timing = getattr(self.transport, "pop_timing", lambda event: None)(events[chunk.offset])
                measured = dict(gpu_ms=timing) if timing is not None else {}
                self._event("chunk_complete", request_id=identity, generation=storage.generation,
                            direction=direction, offset=chunk.offset, pages=chunk.count, **measured)
                if timing is not None:
                    metric = f"swap_{direction}_gpu_ms"
                    self.metrics[metric] = self.metrics.get(metric, 0.) + timing
                self.metrics["swap_" + ("d2h" if direction == "down" else "h2d") + "_pages"] += chunk.count
                key = identity, storage.generation, chunk.offset
                if direction == "down" and key in self.claims.claims:
                    owner = self.claims.claims[key][0]
                    self.claims.consume(key, owner)
            if hot_event is not None and storage.hot_event is None:
                timing = getattr(self.transport, "pop_timing", lambda event: None)(hot_event)
                if timing is not None:
                    self._event("hot_complete", request_id=identity, generation=storage.generation,
                                gpu_ms=timing, pages=len(storage.hot_pages))
                    self.metrics["swap_hot_gpu_ms"] = self.metrics.get("swap_hot_gpu_ms", 0.) + timing
            state = self.compute.get(identity)
            if state is not None and not state[1] and state[0].query():
                self.compute.pop(identity)
                if identity != self.target:
                    storage.pinned = False
        keep = []
        for ptr, pages, ready in self.garbage:
            if ready.query():
                ptr._release(pages)
            else:
                keep.append((ptr, pages, ready))
        self.garbage = keep

    def _cancel_plan(self):
        if self.target is not None:
            storage = self.storages.get(self.target)
            if storage is not None and not storage.complete:
                storage.pinned = False
            self._event("plan_end", request_id=self.target, complete=storage.complete if storage else False)
            self.claims.cancel(self.target)
        self.target = None
        self.producers.clear()  # Submitted jobs remain owned by their storage.

    def _victims(self, receiver=None, *, release_plan=False):
        entries = sorted(self.candidates.values(), key=lambda c: c.victim_key)
        for candidate in entries:
            identity = candidate.identity
            storage = self.storages.get(identity)
            released_target = release_plan and identity == self.target
            if (storage is None or self._protected(identity)
                    or (storage.pinned and not released_target)
                    or (identity == self.target and not released_target)):
                continue
            if receiver is not None and not may_replace(receiver, candidate):
                continue
            for chunk in storage.chunks:
                if chunk.job is None and all(p is not None for p in
                        storage.history[chunk.offset:chunk.offset + chunk.count]):
                    key = identity, storage.generation, chunk.offset
                    claim = self.claims.claims.get(key)
                    if claim is None or (release_plan and claim[0] == self.target):
                        yield identity, storage, chunk

    def _prepare_backing(self, storage, chunk):
        if chunk.cpu is None:
            if self.cpu_pages + chunk.count > self.cpu_capacity:
                self.metrics["swap_cpu_waits"] += 1
                return False
            chunk.cpu = self.transport.cpu_buffer(chunk.count)
        return True

    def _restore_candidates(self):
        return sorted((c for c in self.candidates.values()
                            if c.identity in self.storages and c.identity not in self.retired
                            and self.storages[c.identity].hot_ready
                            and self.storages[c.identity].missing
                            and not self._protected(c.identity)
                            and not any(chunk.job is not None and chunk.job.direction == "down"
                                        for chunk in self.storages[c.identity].chunks)),
                      key=lambda c: c.restore_key)

    @staticmethod
    def _unallocated_missing(storage):
        # A canceled/replanned request can still own queued H2D destinations.
        allocated = sum(c.count for c in storage.chunks
                        if c.job is not None and c.job.direction == "up")
        return storage.missing - allocated

    def _select_sources(self, receiver, *, release_plan=False):
        storage = self.storages[receiver.identity]
        need = max(0, self._unallocated_missing(storage) + self.reserve - self.pool.num_free_pages)
        selected, credit, cpu = [], 0, self.cpu_pages
        for identity, victim, chunk in self._victims(receiver, release_plan=release_plan):
            if credit >= need:
                break
            if chunk.cpu is None:
                if cpu + chunk.count > self.cpu_capacity:
                    continue
                cpu += chunk.count
            selected.append((identity, victim, chunk))
            credit += chunk.count
        return selected if credit >= need else None

    def _preempt_for_nearer_verify(self):
        current = self.candidates.get(self.target)
        if current is None:
            self._cancel_plan()
            return
        for receiver in self._restore_candidates():
            if receiver.restore_key >= current.restore_key:
                break
            # Do not abandon a feasible prefetch for an infeasible restore.
            # Simulate releasing ONLY this plan's unsubmitted claims/pin.
            if self._select_sources(receiver, release_plan=True) is not None:
                self._event("plan_preempt", request_id=self.target,
                            receiver=receiver.identity, old_distance=current.distance,
                            new_distance=receiver.distance)
                self._cancel_plan()
                return

    def _new_plan(self):
        for receiver in self._restore_candidates():
            storage = self.storages[receiver.identity]
            selected = self._select_sources(receiver)
            if selected is None:
                continue
            # Reserve CPU buffers before publishing claims; allocation failure
            # never authorizes future GPU writes.
            for _, victim, chunk in selected:
                if not self._prepare_backing(victim, chunk):
                    return False
            self.target = receiver.identity
            storage.pinned = True
            for identity, victim, chunk in selected:
                key = identity, victim.generation, chunk.offset
                self.claims.claim(key, self.target, chunk.count)
                self.producers.append((identity, victim.generation, chunk.offset))
            self._event("plan", request_id=self.target, distance=receiver.distance,
                        verify=receiver.verify, missing_pages=storage.missing,
                        free_pages=self.pool.num_free_pages, reserve=self.reserve,
                        victims=[dict(request_id=i, generation=s.generation, offset=c.offset, pages=c.count,
                                      distance=self.candidates[i].distance, verify=self.candidates[i].verify)
                                 for i, s, c in selected])
            self.metrics["swap_plans"] += 1
            self.metrics["swap_future_pages_peak"] = max(
                self.metrics["swap_future_pages_peak"], self.claims.pages(self.target))
            return True
        return False

    def _extend_plan(self):
        """Reconcile new compute/hot-view allocations against future claims."""
        storage = self.storages[self.target]
        deficit = (self._unallocated_missing(storage) + self.reserve
                   - self.pool.num_free_pages - self.claims.pages(self.target))
        if deficit <= 0:
            return
        receiver = self.candidates.get(self.target)
        if receiver is None:
            self._cancel_plan()
            return
        selected, credit, cpu = [], 0, self.cpu_pages
        for identity, victim, chunk in self._victims(receiver):
            if chunk.cpu is None:
                if cpu + chunk.count > self.cpu_capacity:
                    continue
                cpu += chunk.count
            selected.append((identity, victim, chunk))
            credit += chunk.count
            if credit >= deficit:
                break
        if credit < deficit:
            self._cancel_plan()  # In-flight chunks still retain all their pages.
            return
        for identity, victim, chunk in selected:
            if not self._prepare_backing(victim, chunk):
                self._cancel_plan()
                return
            key = identity, victim.generation, chunk.offset
            self.claims.claim(key, self.target, chunk.count)
            self.producers.append(key)
        self._event("plan_extend", request_id=self.target, additional_pages=credit)

    def tick(self):
        self.metrics["swap_worker_ticks"] += 1
        self.poll()
        down_busy = any(c.job is not None and c.job.direction == "down"
                        for s in self.storages.values() for c in s.chunks)
        up_busy = any(s.hot_event is not None or any(c.job is not None and c.job.direction == "up"
                       for c in s.chunks) for s in self.storages.values())
        for identity, storage in self.storages.items():
            if identity not in self.retired and not up_busy and storage.start_hot():
                up_busy = True
        if self.target is not None:
            target = self.storages.get(self.target)
            if target is None or self.target in self.retired or target.complete:
                self._cancel_plan()
        if self.target is not None and not up_busy:
            # Reconsider priorities at an H2D chunk boundary. New Draft
            # progress can make a different request's Verify more urgent.
            self._preempt_for_nearer_verify()
        if self.target is None:
            self._new_plan()
        if self.target is not None:
            self._extend_plan()
        if self.target is not None and not up_busy:
            storage = self.storages[self.target]
            for chunk in storage.chunks:
                if chunk.job is None and chunk.saved and all(p is None for p in
                        storage.history[chunk.offset:chunk.offset + chunk.count]):
                    if chunk.count + self.reserve <= self.pool.num_free_pages:
                        storage.start_up(chunk)
                        self._event("chunk_submit", request_id=self.target, generation=storage.generation,
                                    direction="up", offset=chunk.offset, pages=chunk.count)
                        self.metrics["swap_up_chunks"] += 1
                        up_busy = True
                    break
        if not down_busy and self.producers:
            identity, generation, offset = self.producers[0]
            storage = self.storages.get(identity)
            receiver = self.candidates.get(self.target)
            victim = self.candidates.get(identity)
            if (storage is None or storage.generation != generation or self._protected(identity)
                    or receiver is None or victim is None or not may_replace(receiver, victim)):
                self._cancel_plan()
            else:
                chunk = next(c for c in storage.chunks if c.offset == offset)
                if storage.start_down(chunk):
                    self._event("chunk_submit", request_id=identity, generation=storage.generation,
                                direction="down" if chunk.job else "release_backed",
                                offset=chunk.offset, pages=chunk.count)
                    self.producers.popleft()
                    key = identity, generation, offset
                    if chunk.job is None:  # Already backed: release without D2H.
                        self.claims.consume(key, self.target)
                    self.metrics["swap_down_chunks"] += 1
                    down_busy = chunk.job is not None
        # Admission/append pressure and no-hot-workspace bootstrap. These
        # release chunks without waiting for a receiver's complete footprint.
        bootstrap = any(not s.hot_ready and s.hot_event is None for i, s in self.storages.items()
                        if i not in self.retired)
        if not down_busy and not self.producers and (self.pool.num_free_pages < self.pressure or bootstrap):
            for identity, storage, chunk in self._victims():
                if self._prepare_backing(storage, chunk) and storage.start_down(chunk):
                    self._event("chunk_submit", request_id=identity, generation=storage.generation,
                                direction="down" if chunk.job else "release_backed",
                                offset=chunk.offset, pages=chunk.count, reason="bootstrap" if bootstrap else "pressure")
                    self.metrics["swap_down_chunks"] += 1
                    down_busy = chunk.job is not None
                    break
        self.metrics["swap_peak_cpu_pages"] = max(self.metrics.get("swap_peak_cpu_pages", 0), self.cpu_pages)
        self.metrics["swap_peak_gpu_pages"] = max(self.metrics.get("swap_peak_gpu_pages", 0),
                                                 self.pool.capacity - self.pool.num_free_pages)
        return (down_busy or up_busy or bool(self.producers) or bool(self.garbage))
