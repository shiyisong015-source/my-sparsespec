"""AIME24 compact CPU-offload r scan, two concurrent TP2 jobs.

Reuse the audited original token inputs and their natural-EOS output caps.
Only the experiment wrapper is instrumented; TopK and inference are unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reproduction.run_resource_case import (
    REPO, completion_audit_class, read_inputs, summarize,
)


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(path):
    import torch
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import torch.distributed as dist
    import serve.run as engine
    import serve.scheduler.compact_scheduler as compact
    from serve.request.request import Request

    payload = json.loads(path.read_text())
    rank = int(os.environ["RANK"])
    output = Path(payload["output"])
    rows, caps = payload["requests"], payload["output_caps"]
    requests = [Request(row["input_ids"], cap) for row, cap in zip(rows, caps, strict=True)]
    audited = list(requests)

    class ObservedScheduler(completion_audit_class(compact.CompactSpecScheduler)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_histogram = Counter()
            self.completed_count = 0
            self.observer_start = time.perf_counter()

        def prepare_forward_inputs(self):
            self.batch_histogram[len(self.active_requests)] += 1
            self.current_phases = dict(Counter(r.status.name for r in self.active_requests))
            return super().prepare_forward_inputs()

        def update_requests(self, *args, **kwargs):
            completed = super().update_requests(*args, **kwargs)
            self.completed_count += len(completed)
            if self.step_id % 100 == 0 or self.completed_count == len(rows):
                if self.acceptance_log is not None:
                    self.acceptance_log.flush()
                progress = dict(rank=rank, step=self.step_id,
                                completed=self.completed_count, total=len(rows),
                                elapsed_seconds=time.perf_counter() - self.observer_start,
                                phases=self.current_phases,
                                batch_histogram=dict(self.batch_histogram),
                                metrics=dict(self.metrics), wall_time=time.time())
                save(output / f"progress.rank{rank}.json", progress)
                if rank == 0:
                    print("COMPACT_PROGRESS", json.dumps(progress), flush=True)
            return completed

    compact.CompactSpecScheduler = ObservedScheduler
    kwargs = dict(payload["kwargs"], result_dir=str(output / "engine"),
                  acceptance_log_path=str(output / ("rounds.jsonl" if rank == 0
                                                   else f"rounds.rank{rank}.jsonl")))
    outputs = engine.run_inference(reqs=requests, **kwargs)
    if rank == 0:
        if len(outputs) != len(rows):
            raise RuntimeError("Not every AIME input completed")
        records = []
        for i, (row, cap, generated) in enumerate(zip(rows, caps, outputs, strict=True)):
            reason = getattr(audited[i], "resource_finish_reason", "unclassified")
            if (generated["id"] != i or generated["len_input"] != len(row["input_ids"])
                    or generated["len_output"] != len(generated["output_token_ids"])
                    or reason == "unclassified"):
                raise RuntimeError(f"Inconsistent completion metadata for request {i}")
            records.append(dict(request_id=i, sample_id=row["sample_id"],
                                input_tokens=len(row["input_ids"]),
                                output_tokens=generated["len_output"], output_cap=cap,
                                finish_reason=reason,
                                tokens_beyond_output_cap=max(0, generated["len_output"] - cap)))
        save(output / "request-audit.json", records)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def gpu_inventory(pair):
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    states = {int(c[0]): c for c in
              ([s.strip() for s in line.split(",")] for line in inventory.splitlines())}
    if any(gpu not in states or int(states[gpu][3]) >= 128
           or int(states[gpu][4]) != 0 for gpu in pair):
        raise RuntimeError(f"GPU pair {pair} is not idle; no job launched on it")
    return {gpu: states[gpu] for gpu in pair}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_payload:
        worker(args.worker_payload)
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log") or output.exists():
        parser.error("Use a new output directory under SparseSpec/log")
    inputs = REPO / "log/reproduction/resource-motivation/inputs-distinct-seed330/aime24.json"
    original, rows = read_inputs(inputs, Path("/data/models/Qwen3-8B"))
    if original["dataset"] != "aime24" or len(rows) != 30:
        raise RuntimeError("Expected the full audited AIME24 set of 30 requests")
    pairs = [(4, 5), (6, 7)]
    for pair in pairs:
        gpu_inventory(pair)
    output.mkdir(parents=True)
    save(output / "inputs.json", original)
    source_files = [Path(__file__).resolve(), REPO / "serve/run.py",
                    REPO / "serve/scheduler/compact_scheduler.py",
                    REPO / "serve/request/kv_cache_ptr/compact.py",
                    REPO / "serve/attention/kv_pack.py", REPO / "serve/model/model.py"]
    save(output / "source-sha256.json", {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    caps = [32768 - len(row["input_ids"]) for row in rows]
    baseline = REPO / "log/reproduction/resource-motivation/r-scan-distinct-seed330"
    state = dict(state="running", pid=os.getpid(), started_wall_time=time.time(), cases={})
    pending, active = [.05, .10, .15], {}

    def launch(pair, ratio):
        inventory = gpu_inventory(pair)
        case = output / f"r{ratio:.2f}"
        case.mkdir()
        old = json.loads((baseline / f"scan-aime24-r{ratio:.2f}/launch.json").read_text())
        kwargs = dict(old["kwargs"], kv_cache="compact", cuda_graph=False, async_cpu=False,
                      reduce_method="customized",
                      compact_cpu_pages=(len(rows) + 1) * (32768 + 8 + 1))
        if caps != old["output_caps"] or [r["sample_id"] for r in rows] != old["sample_order"]:
            raise RuntimeError("Input order or output caps differ from the original scan")
        save(case / "worker-payload.json", dict(output=str(case), requests=rows,
                                                output_caps=caps, kwargs=kwargs))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=2", "--", str(Path(__file__).resolve()),
                   "--worker-payload", str(case / "worker-payload.json")]
        manifest = dict(state="running", command=command, kwargs=kwargs,
                        gpus=pair, gpu_inventory=inventory, old_baseline=str(baseline / f"scan-aime24-r{ratio:.2f}"),
                        input_file_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                        token_ids_sha256=original["token_ids_sha256"],
                        sample_order=old["sample_order"], repeats=1,
                        differences_from_original=dict(kv_cache="pillar -> compact",
                            cuda_graph="true -> false", async_cpu="true -> false",
                            compact_cpu_pages=kwargs["compact_cpu_pages"]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, pair)),
                   ENABLE_INTRA_NODE_COMM="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONOPTIMIZE", None)
        log = (case / "runtime.log").open("w")
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(pid=process.pid, started_wall_time=time.time())
        save(case / "launch.json", manifest)
        state["cases"][case.name] = manifest
        active[pair] = (process, log, case, manifest)
        print("LAUNCHED", case.name, "TP2 GPUs", pair, "pid", process.pid, flush=True)

    while pending or active:
        for pair in pairs:
            if pair not in active and pending:
                launch(pair, pending.pop(0))
        save(output / "controller-status.json", state)
        time.sleep(5)
        for pair, (process, log, case, manifest) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            manifest.update(exit_code=code, ended_wall_time=time.time(),
                            state="complete" if code == 0 else "failed")
            if code == 0:
                try:
                    manifest["result"] = summarize(case, len(rows))
                except Exception as exc:
                    manifest.update(state="failed", result_error=str(exc))
            save(case / "launch.json", manifest)
            print("FINISHED", case.name, manifest["state"], "exit", code, flush=True)
            del active[pair]
    state.update(state="complete" if all(c["state"] == "complete" for c in state["cases"].values())
                 else "failed", ended_wall_time=time.time())
    save(output / "controller-status.json", state)
    if state["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
