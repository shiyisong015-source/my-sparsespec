"""CPU-only BW/HBM lookahead calendar, following VeriCache Algorithm 1.

Time is measured in scheduler iterations. HBM is in original-pool pages;
bandwidth is bytes/second PER TP RANK. A reload allocates its entire target
at submission, so its full footprint is reserved throughout the transfer span.
"""
from dataclasses import dataclass
import math


@dataclass
class Reservation:
    request_id: int
    start: int
    deadline: int
    transfer_bytes: int
    memory_pages: int
    bandwidth_slots: tuple
    materialized: bool = False
    missed: bool = False
    dependencies: tuple = ()
    release_event: object = None


@dataclass
class MemoryRelease:
    event_id: int
    source_key: int
    available_at: int
    pages: int
    committed: bool = False


class ReservationTable:
    def __init__(self, horizon, bandwidth_bytes_per_second, iteration_seconds):
        if type(horizon) is not int or horizon < 2:
            raise ValueError("Reservation horizon must be an integer >= 2")
        if (not math.isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0
                or not math.isfinite(iteration_seconds) or iteration_seconds <= 0):
            raise ValueError("Reservation bandwidth and iteration time must be finite and positive")
        self.horizon = horizon
        self.bandwidth = bandwidth_bytes_per_second
        self.iteration_seconds = iteration_seconds
        self.now = 0
        self.reservations = {}
        self._tags = [-1] * horizon
        self._bw = [0.] * horizon
        self._hbm = [0] * horizon
        self.releases = {}
        self._next_release_event = 0

    def _slot(self, tick):
        slot = tick % self.horizon
        if self._tags[slot] != tick:
            self._tags[slot], self._bw[slot], self._hbm[slot] = tick, 0., 0
        return slot

    def advance(self, now):
        if type(now) is not int or now < self.now:
            raise ValueError("Reservation time must advance monotonically")
        self.now = now
        # Reservations use absolute tags: a wrapped slot never inherits usage
        # from an old window. Keep overdue records until explicit reconciliation.

    @staticmethod
    def candidates(anchor, earliest, latest):
        if earliest > latest:
            return
        anchor = max(earliest, min(anchor, latest))
        yield anchor
        for distance in range(1, max(anchor - earliest, latest - anchor) + 1):
            if anchor - distance >= earliest:
                yield anchor - distance
            if anchor + distance <= latest:
                yield anchor + distance

    def reserve(self, request_id, *, transfer_bytes, memory_pages, anchor,
                resident_pages, capacity_pages):
        if request_id in self.reservations:
            raise ValueError("Cancel a request's old reservation before admitting it again")
        values = (transfer_bytes, memory_pages, resident_pages, capacity_pages)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Bytes and page counts must be nonnegative integers")
        if type(anchor) is not int:
            raise ValueError("Reservation anchor must be an integer")
        duration = transfer_bytes / self.bandwidth
        span = max(1, math.ceil(duration / self.iteration_seconds))
        if span >= self.horizon or resident_pages > capacity_pages:
            return None
        for offset in self.candidates(anchor, span, self.horizon - 1):
            deadline = self.now + offset
            start = deadline - span + 1
            windows = tuple(range(start, deadline + 1))
            charges = tuple(min(self.iteration_seconds, max(
                0., duration - index * self.iteration_seconds)) for index in range(span))
            # Whole-target allocation: only a release predicted BEFORE start
            # can support this reload. Do not borrow a release mid-transfer.
            release_need = max(0, max(resident_pages + self._hbm[self._slot(tick)]
                                     + memory_pages - capacity_pages for tick in windows))
            dependencies, credit = [], 0
            for event in sorted(self.releases.values(), key=lambda e: (e.available_at, e.event_id)):
                if credit >= release_need:
                    break
                if not event.committed and event.available_at <= start:
                    dependencies.append(event.event_id)
                    credit += event.pages
            if credit < release_need:
                continue
            if any(resident_pages + self._hbm[self._slot(tick)] > capacity_pages
                   + self.release_pages(tick)
                   for tick in range(self.now, self.now + self.horizon)
                   if not start <= tick <= deadline):
                continue
            if any(self._bw[self._slot(tick)] + seconds > self.iteration_seconds + 1e-12
                   for tick, seconds in zip(windows, charges)):
                continue
            reservation = Reservation(request_id, start, deadline, transfer_bytes,
                                      memory_pages, tuple(zip(windows, charges)),
                                      dependencies=tuple(dependencies))
            for tick, seconds in reservation.bandwidth_slots:
                slot = self._slot(tick)
                self._bw[slot] += seconds
                self._hbm[slot] += memory_pages
            self.reservations[request_id] = reservation
            return reservation
        return None

    def materialize(self, request_id, *, source_pages=None):
        """Allocated reload pages now belong to actual resident accounting.

        This must be called only AFTER a successful real allocation. Do not
        count the same pages in both baseline residency and the HBM ring.
        BW reservations remain until their original windows expire.
        """
        reservation = self.reservations[request_id]
        if reservation.materialized:
            return
        if not self.dependencies_ready(request_id):
            raise RuntimeError("Reload allocation depends on an uncommitted memory release")
        if source_pages is not None:
            if (reservation.request_id >= 0 or reservation.release_event is None
                    or type(source_pages) is not int
                    or source_pages < self.releases[reservation.release_event].pages):
                raise ValueError("Repack source pages must cover its forecast net release")
            # The new hot destinations are NOW part of actual residency.
            # At commit the entire old source is freed, not just its cold
            # subset. Updating this avoids counting the hot copy twice.
            self.releases[reservation.release_event].pages = source_pages
        for tick, _ in reservation.bandwidth_slots:
            slot = tick % self.horizon
            if tick >= self.now and self._tags[slot] == tick:
                self._hbm[slot] -= reservation.memory_pages
        reservation.materialized = True

    def cancel(self, request_id):
        reservation = self.reservations.pop(request_id, None)
        if reservation is None:
            return ()
        for tick, seconds in reservation.bandwidth_slots:
            slot = tick % self.horizon
            if tick < self.now or self._tags[slot] != tick:
                continue
            self._bw[slot] = max(0., self._bw[slot] - seconds)
            if not reservation.materialized:
                self._hbm[slot] -= reservation.memory_pages
        invalidated = [request_id]
        if reservation.release_event is not None:
            event = self.releases.get(reservation.release_event)
            if event is not None and not event.committed:
                invalidated.extend(self._invalidate_dependents(event.event_id))
                self.releases.pop(event.event_id, None)
        self._collect_releases()
        return tuple(invalidated)

    def plan_release(self, source_key, *, pages, available_at):
        source = self.reservations[source_key]
        if source_key >= 0 or source.memory_pages or source.release_event is not None:
            raise ValueError("A release must belong to one unique link-only repack booking")
        if (type(pages) is not int or pages <= 0 or type(available_at) is not int
                or available_at <= source.deadline):
            raise ValueError("Positive release pages must become available after repack's deadline")
        self._next_release_event += 1
        event = MemoryRelease(self._next_release_event, source_key, available_at, pages)
        self.releases[event.event_id] = event
        source.release_event = event.event_id
        return event

    def release_pages(self, tick):
        # Once committed, real free-page accounting already includes it.
        return sum(event.pages for event in self.releases.values()
                   if not event.committed and event.available_at <= tick)

    def dependencies_ready(self, request_id):
        return all(event_id in self.releases and self.releases[event_id].committed
                   for event_id in self.reservations[request_id].dependencies)

    def confirm_release(self, event_id):
        event = self.releases[event_id]
        if event.committed:
            return False
        event.committed = True
        self._collect_releases()
        return True

    def _invalidate_dependents(self, event_id):
        invalidated = []
        for key, reservation in list(self.reservations.items()):
            if not reservation.materialized and event_id in reservation.dependencies:
                invalidated.extend(self.cancel(key))
        return invalidated

    def defer_release(self, event_id, available_at):
        event = self.releases[event_id]
        if event.committed or available_at <= event.available_at:
            return ()
        event.available_at = available_at
        # Rebook affected unstarted reloads using the new timing. In-flight
        # allocations are never undone by a forecast correction.
        return tuple(self._invalidate_dependents(event_id))

    def _collect_releases(self):
        referenced = {event_id for r in self.reservations.values() for event_id in r.dependencies}
        referenced.update(r.release_event for r in self.reservations.values()
                          if r.release_event is not None)
        for event_id in list(self.releases):
            if self.releases[event_id].committed and event_id not in referenced:
                del self.releases[event_id]

    def memory_profile(self, resident_pages, capacity_pages):
        result = []
        for tick in range(self.now, self.now + self.horizon):
            reloads = self._hbm[self._slot(tick)]
            release = self.release_pages(tick)
            occupancy = resident_pages + reloads - release
            result.append(dict(iteration=tick, resident_and_reserve_pages=resident_pages,
                               reload_pages=reloads, planned_release_pages=release,
                               projected_used_pages=occupancy,
                               projected_free_pages=capacity_pages - occupancy))
        return result

    def memory_feasible(self, resident_pages, capacity_pages):
        return all(window["projected_free_pages"] >= 0
                   for window in self.memory_profile(resident_pages, capacity_pages))

    @property
    def peak_memory_pages(self):
        return max(self._hbm[self._slot(tick)]
                   for tick in range(self.now, self.now + self.horizon))

    def snapshot(self, resident_pages=None, capacity_pages=None):
        windows = []
        for tick in range(self.now, self.now + self.horizon):
            slot = self._slot(tick)
            windows.append(dict(iteration=tick, bandwidth_seconds=self._bw[slot],
                                reload_pages=self._hbm[slot], planned_release_pages=self.release_pages(tick)))
        if resident_pages is not None and capacity_pages is not None:
            for window, projection in zip(windows, self.memory_profile(resident_pages, capacity_pages)):
                window.update(projection)
        return dict(iteration=self.now, horizon=self.horizon,
                    bandwidth_bytes_per_second=self.bandwidth,
                    iteration_seconds=self.iteration_seconds, windows=windows,
                    reservations=[dict(request_id=r.request_id, start=r.start,
                                       kind="repack" if r.request_id < 0 else "reload",
                                       deadline=r.deadline, transfer_bytes=r.transfer_bytes,
                                       memory_pages=r.memory_pages, materialized=r.materialized,
                                       dependencies=list(r.dependencies), release_event=r.release_event)
                                  for r in self.reservations.values()],
                    releases=[dict(event_id=e.event_id, source_key=e.source_key,
                                   available_at=e.available_at, pages=e.pages, committed=e.committed)
                              for e in self.releases.values()])
