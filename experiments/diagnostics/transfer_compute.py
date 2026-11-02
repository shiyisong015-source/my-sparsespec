"""Observe transfer/compute overlap in the current scheduler without changing it."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(output):
    from transformers import AutoTokenizer
    source = REPO / "log/v2/samples-longbench-seed330/confirm.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    random.Random(330).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3-8B", local_files_only=True)
    requests = []
    for row in rows[:64]:
        ids = tokenizer.apply_chat_template([dict(role="user", content=row["prompt"])],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        requests.append(dict(sample_id=row["sample_id"], source=row["source"], input_ids=ids))
    reference = REPO / "log/acceptance-u-k32-dynamic-20261020/full128-mem075/r0.05/lcb128-natural/worker-payload.json"
    kwargs = json.loads(reference.read_text())["kwargs"]
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(output=str(output), kwargs=kwargs, requests=requests,
        output_caps=[2048] * len(requests), capture_seconds=12,
        input_file=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((REPO / "serve").rglob("*.py"))},
        scope="64 real LongBench prompts, seed330, first 12-second window after two restores; natural pool capacity")
    save(output / "worker-payload.json", payload)
    save(output / "inputs-summary.json", dict(requests=len(requests),
        total_input_tokens=sum(len(r["input_ids"]) for r in requests),
        min_input_tokens=min(len(r["input_ids"]) for r in requests),
        max_input_tokens=max(len(r["input_ids"]) for r in requests),
        source_counts=dict(Counter(r["source"] for r in requests)), kwargs=kwargs))
    print(output, flush=True)


def worker(path):
    print("DIAG_WORKER_IDENTITY", json.dumps(dict(uid=os.getuid(), gid=os.getgid(),
        cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"), output=str(path.parent))), flush=True)
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import serve.scheduler.dynamic_compact_scheduler as module
    from experiments.reproduction.run_compact_acceptance_scan import worker as original_worker
    payload = json.loads(path.read_text())
    for name, expected in payload["source_sha256"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Serving source changed: {name}")
    output, rank = Path(payload["output"]), int(os.environ["RANK"])

    class Observed(module.DynamicCompactSpecScheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observation = (output / f"states.rank{rank}.jsonl").open("w", buffering=1)
            self.capture_started = self.capture_finished = False
            self.pending_count = 0
            self.forward_phases = {}
            self.transfer_sequence = 0
            original_forward = self.model.forward

            def forward(*args, **kwargs):
                label = f"diag.forward step={self.step_id} phases=" + json.dumps(self.forward_phases, separators=(",", ":"))
                torch.cuda.nvtx.range_push(label)
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()
            self.model.forward = forward

        def write(self, value):
            self.observation.write(json.dumps(dict(value, rank=rank, step=self.step_id,
                wall_time_ns=time.time_ns(), perf_counter=time.perf_counter()), default=str) + "\n")

        def boundary(self):
            dist.barrier(group=self.group)
            self.compute_stream.synchronize()
            self.transfer_stream.synchronize()
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=self.group)

        def start_capture(self):
            self.boundary()
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStart()
            dist.barrier(group=self.group)
            self.capture_range = torch.cuda.nvtx.range_start("diag.capture.resource")
            self.capture_started = True
            self.capture_start_perf = time.perf_counter()
            self.write(dict(event="capture_start", metrics=dict(self.metrics)))

        def finish_capture(self):
            # Drain actual submitted work, then end the measurement range.
            self.boundary()
            torch.cuda.nvtx.range_end(self.capture_range)
            self.write(dict(event="capture_stop", metrics=dict(self.metrics)))
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
            dist.barrier(group=self.group)
            self.capture_finished = True

        def _submit_transfer(self, request, direction):
            if direction == "restore" and not self.capture_started and self.metrics["restores"] >= 2:
                self.start_capture()
            sequence = self.transfer_sequence
            self.transfer_sequence += 1
            ptr = request.kv_cache_ptr
            pages = ptr.cold_count if direction == "restore" else len(ptr.indices) - ptr.len_loading_indices
            self.write(dict(event="transfer_submit", direction=direction, sequence=sequence,
                request=request.request_id, pages=pages, bytes=pages * self.page_bytes,
                logical_tokens=ptr.len_kv_cache, free_pages=self.kv_pool.num_free_pages))
            torch.cuda.nvtx.range_push(f"diag.kv.{direction} seq={sequence} request={request.request_id} step={self.step_id}")
            try:
                return super()._submit_transfer(request, direction)
            finally:
                torch.cuda.nvtx.range_pop()

        def _poll_transfers(self):
            before = {key: job.direction for key, job in self._transfers.items()}
            result = super()._poll_transfers()
            for key in before.keys() - self._transfers.keys():
                self.write(dict(event="transfer_commit", request=key, direction=before[key]))
            return result

        def admit_requests(self, pending):
            self.pending_count = len(pending)
            if self.capture_started and not self.capture_finished and self.step_id % 10 == 0:
                command = [rank == 0 and time.perf_counter() - self.capture_start_perf >= payload["capture_seconds"]]
                dist.broadcast_object_list(command, src=0, group=self.group)
                if command[0]:
                    self.finish_capture()
            super().admit_requests(pending)

        def prepare_forward_inputs(self):
            host, tensors = super().prepare_forward_inputs()
            self.forward_phases = dict(Counter(r.status.name for r in self.active_requests if r.status.name != "STALL"))
            if self.capture_started and not self.capture_finished:
                computing = {r.request_id for r in self.active_requests if r.status.name != "STALL"}
                requests = [dict(id=r.request_id, status=r.status.name,
                    phase=getattr(r.kv_cache_ptr, "phase", "prefill"),
                    logical=r.len_kv_cache, hot=getattr(r.kv_cache_ptr, "len_loading_indices", len(r.kv_cache_ptr.indices)),
                    physical=len(r.kv_cache_ptr.indices), draft_count=r.len_draft_probs,
                    computing=r.request_id in computing, transferring=r.request_id in self._transfers)
                    for r in self._requests.values()]
                self.write(dict(event="state", pending=self.pending_count,
                    capacity=self.kv_pool.capacity, free_pages=self.kv_pool.num_free_pages,
                    queries=0 if host is None else len(host["token_ids"]),
                    phases=self.forward_phases, requests=requests,
                    metrics={k:self.metrics[k] for k in ["restores", "migrations", "recovery_memory_waits", "token_budget_waits", "metadata_waits"]}))
            return host, tensors

        def wait_for_ready_requests(self):
            started = time.perf_counter()
            torch.cuda.nvtx.range_push("diag.no_ready_wait")
            try:
                return super().wait_for_ready_requests()
            finally:
                torch.cuda.nvtx.range_pop()
                self.write(dict(event="no_ready_wait", wall_seconds=time.perf_counter() - started))

        def run(self, requests):
            try:
                result = super().run(requests)
                if self.capture_started and not self.capture_finished:
                    self.finish_capture()
                save(output / f"observer.rank{rank}.json", dict(capture_started=self.capture_started,
                    capture_finished=self.capture_finished, metrics=dict(self.metrics)))
                if not self.capture_finished:
                    raise RuntimeError("No restoration capture window; do not claim a transfer result")
                return result
            finally:
                self.observation.close()

    module.DynamicCompactSpecScheduler = Observed
    original_worker(path)


def guard(output, command):
    """Watch only our child; GPU4/5 run, GPU6/7 must remain idle."""
    spec = importlib.util.spec_from_file_location("existing_guard", REPO / "log/hardware-counter-smoke-20261019/resource_guard.py")
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    child = None
    last_owned_gpu = {}
    with output.open("w", buffering=1) as log:
        def owned(pid):
            if child is None:
                return False
            if existing.belongs_to_capture(pid, child.pid):
                return True
            # Nsight intentionally daemonizes its agent (PPid=1, new session).
            # Match both our frontend's exact session and the executable path.
            seen = set()
            while pid > 1 and pid not in seen:
                seen.add(pid)
                try:
                    tokens = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                    if "--start-agent" in tokens and "--session-name" in tokens:
                        session = tokens[tokens.index("--session-name") + 1]
                        if (session == f"profile-{child.pid}"
                                and Path(f"/proc/{pid}/exe").resolve() == Path(command[0]).resolve()):
                            return True
                    status = Path(f"/proc/{pid}/status").read_text()
                    pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
                except (OSError, ValueError, IndexError, StopIteration):
                    return False
            return False

        def check():
            state = existing.snapshot()
            problems = [f"external GPU process {p['pid']}" for p in state["processes"]
                        if not owned(p["pid"]) and Path(f"/proc/{p['pid']}").exists()]
            own_uuids = {p["uuid"] for p in state["processes"] if owned(p["pid"])}
            for uuid in own_uuids:
                last_owned_gpu[uuid] = time.monotonic()
            for gpu in state["devices"]:
                if gpu["index"] in (4, 5) and gpu["uuid"] in own_uuids:
                    continue
                # NVML can retain our allocation briefly after process exit.
                # External process checks above still reject any new task.
                if (gpu["index"] in (4, 5) and not problems
                        and time.monotonic() - last_owned_gpu.get(gpu["uuid"], -1e30) < 3):
                    continue
                if gpu["memory_mib"] > 64 or gpu["utilization"] != 0:
                    problems.append(f"GPU {gpu['index']} is not idle")
            problems += [f"external profiler {p['name']} pid={p['pid']}" for p in state["collectors"] if not owned(p["pid"])]
            log.write(json.dumps(dict(state, blocking_reasons=problems)) + "\n")
            if problems:
                raise RuntimeError("; ".join(problems))
        try:
            for _ in range(5):
                check()
                time.sleep(1)
            child = subprocess.Popen(command, start_new_session=True)
            started = time.monotonic()
            while child.poll() is None:
                time.sleep(1)
                if child.poll() is not None:
                    break
                if time.monotonic() - started > 900:
                    raise RuntimeError("Our profiling process exceeded 900 seconds")
                check()
            if child.returncode:
                raise RuntimeError(f"Profiler exited {child.returncode}")
        finally:
            if child is not None:
                existing.stop_capture(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--guard", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.worker_payload:
        worker(args.worker_payload.resolve())
    elif args.guard:
        guard(args.guard.resolve(), args.command[1:] if args.command[:1] == ["--"] else args.command)
    else:
        parser.error("Choose prepare, worker-payload or guard")
