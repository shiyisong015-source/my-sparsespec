"""Record the actual rejection uniforms without consuming additional RNG draws."""
from collections import deque
from contextlib import contextmanager
import importlib
import json
from pathlib import Path

import numpy as np
import torch


@contextmanager
def capture_uniforms(module):
    """Intercept function results, including native in-place uniform_ fills."""
    original = module.generate_uniform_probs
    captured = []

    def generate(*args, **kwargs):
        uniform = original(*args, **kwargs)
        captured.append(uniform)
        return uniform

    module.generate_uniform_probs = generate
    try:
        yield captured
    finally:
        module.generate_uniform_probs = original


def extend(base):
    class UniformSignalObserver(base):
        def __init__(self, scheduler, output, rank):
            super().__init__(scheduler, output, rank)
            self.uniform_module = importlib.import_module(
                type(scheduler.rejection_sampler).__module__)
            self.uniform_work, self.uniform_pending = [], deque()
            self.uniform_count = self.uniform_waits = 0
            root = Path(output)
            self.uniform_meta = (root / f"uniform-signals.rank{rank}.jsonl").open("x")
            self.uniform_binary = (root / f"uniform-signals.rank{rank}.f64").open("xb")

        def sampling(self, **kwargs):
            with capture_uniforms(self.uniform_module) as captured:
                result = super().sampling(**kwargs)
            if len(captured) != 1:
                raise RuntimeError(f"Expected one native uniform generation, got {len(captured)}")
            uniform = captured[0]
            context = [dict(row) for row in self.prob_work[-1][0]]
            if uniform.dtype != torch.float64 or uniform.numel() != sum(
                    row["actual_draft_tokens"] for row in context):
                raise RuntimeError("Native uniform shape/dtype mismatch")
            self.uniform_work.append((context, uniform))
            return result

        def submit(self):
            super().submit()
            for context, uniform in self.uniform_work:
                if len(self.uniform_pending) >= 64:
                    self.uniform_waits += 1
                    self.uniform_pending[0][0].synchronize()
                    self.drain_uniforms()
                cpu = torch.empty(tuple(uniform.shape), dtype=torch.float64,
                                  device="cpu", pin_memory=True)
                cpu.copy_(uniform, non_blocking=True)
                ready = torch.cuda.current_stream().record_event()
                # Retain the device tensor until the nonblocking readback completes.
                self.uniform_pending.append((ready, context, cpu, uniform))
            self.uniform_work.clear()
            self.drain_uniforms()

        def drain_uniforms(self, force=False):
            while self.uniform_pending:
                ready, context, cpu, uniform = self.uniform_pending[0]
                if force:
                    ready.synchronize()
                elif not ready.query():
                    break
                self.uniform_pending.popleft()
                values = cpu.numpy()
                if not np.isfinite(values).all() or np.any((values < 0) | (values >= 1)):
                    raise RuntimeError("Invalid native rejection uniform")
                start = 0
                for row in context:
                    count = row["actual_draft_tokens"]
                    trace = np.full(self.k, np.nan, dtype="<f8")
                    trace[:count] = values[start:start + count]
                    row.update(binary_row=self.uniform_count,
                               available="native_uniform_generated_during_verify",
                               dtype="little-endian float64")
                    trace.tofile(self.uniform_binary)
                    self.uniform_meta.write(json.dumps(row, separators=(",", ":")) + "\n")
                    self.uniform_count += 1
                    start += count
                if start != len(values):
                    raise RuntimeError("Native uniform request offsets do not exhaust buffer")

        def flush(self):
            super().flush()
            self.drain_uniforms()
            self.uniform_meta.flush()
            self.uniform_binary.flush()

        def close(self, success):
            if self.closed:
                return
            finished = False
            try:
                super().close(success)
                self.drain_uniforms(force=True)
                if self.uniform_count != self.counts["probability"]:
                    raise RuntimeError("Probability/uniform trace counts differ")
                finished = True
            finally:
                self.uniform_meta.close()
                self.uniform_binary.close()
                (self.output / f"uniform-summary.rank{self.rank}.json").write_text(
                    json.dumps(dict(successful=bool(success and finished), rank=self.rank,
                        k=self.k, records=self.uniform_count, record_shape=[self.k],
                        dtype="little-endian float64", padding="NaN after actual_draft_tokens",
                        readback_queue_waits=self.uniform_waits,
                        source="Exact tensor returned by native generate_uniform_probs; no new RNG draws",
                        available="Collected at verify; does not implement pre-draft sampling"),
                        indent=2) + "\n")

    return UniformSignalObserver
