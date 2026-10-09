"""Optional synchronous-scheduler trace; CUDA events are drained without per-step waits."""
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import time
import torch


class StepTrace:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.enabled = self.path is not None
        self.pending = deque()
        self.current = None
        self.event_pool = []
        self.steps = 0
        self.committed = 0
        self.stage_ms = {}
        self.wall_seconds = 0.
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open('w')

    def begin(self, step_id):
        self.current = {'step_id': step_id, 'gpu_ms': {}, 'cpu_ms': {},
                        'committed_by_request': {}, 'events': []}

    @contextmanager
    def stage(self, name, gpu=False):
        if not self.enabled:
            yield
            return
        start_cpu = time.perf_counter()
        if gpu:
            pair = self.event_pool.pop() if self.event_pool else (
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        try:
            yield
        finally:
            self.current['cpu_ms'][name] = 1000*(time.perf_counter()-start_cpu)
            if gpu:
                pair[1].record()
                self.current['events'].append((name, pair))

    def commit(self, request_id, count):
        if self.enabled and count:
            key = str(request_id)
            self.current['committed_by_request'][key] = self.current['committed_by_request'].get(key, 0)+count

    def finish(self, wall_seconds, **fields):
        if not self.enabled:
            return
        self.current.update(fields, step_wall_seconds=wall_seconds)
        self.current["step_end_perf_counter"] = fields["step_start_perf_counter"] + wall_seconds
        self.pending.append(self.current)
        self.current = None
        self.drain()

    def drain(self, wait=False):
        while self.pending:
            row = self.pending[0]
            events = row['events']
            if not wait and any(not pair[1].query() for _, pair in events):
                break
            if wait:
                for _, pair in events:
                    pair[1].synchronize()
            self.pending.popleft()
            for name, pair in row.pop('events'):
                row['gpu_ms'][name] = pair[0].elapsed_time(pair[1])
                self.stage_ms[name] = self.stage_ms.get(name, 0.)+row['gpu_ms'][name]
                self.event_pool.append(pair)
            row['committed_tokens'] = sum(row['committed_by_request'].values())
            self.committed += row['committed_tokens']
            self.steps += 1
            self.wall_seconds += row['step_wall_seconds']
            self.stream.write(json.dumps(row)+'\n')

    def close(self, successful):
        if not self.enabled:
            return
        try:
            self.drain(wait=True)
        finally:
            self.stream.close()
        summary = {'successful': successful, 'steps': self.steps, 'committed_tokens': self.committed,
                   'step_wall_seconds_sum': self.wall_seconds, 'gpu_stage_ms_sum': self.stage_ms,
                   'note': 'GPU stage intervals may nest; do not sum overlapping stages. CPU stage times include launch/wait costs. Final drain may synchronize.'}
        self.path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
