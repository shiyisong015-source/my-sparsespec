"""Run the audited AIME requests once per r with on-demand async KV.

CPU-only queue owner; launch TP2 jobs only on explicitly named idle pairs.
Use a persistent terminal session to retain the torchrun rendezvous parents.
CPU scheduling is synchronous by default; --async-cpu enables its pipeline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reproduction.run_compact_acceptance_scan import gpu_inventory, save
from experiments.reproduction.run_resource_case import REPO, summarize


def pairs_argument(value):
    try:
        pairs = [tuple(map(int, item.split(","))) for item in value.split(";")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use pairs such as 2,3;4,5;6,7") from exc
    flat = [gpu for pair in pairs for gpu in pair]
    if any(len(pair) != 2 for pair in pairs) or len(set(flat)) != len(flat) or min(flat) < 0:
        raise argparse.ArgumentTypeError("Require distinct nonnegative GPU indices in TP2 pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-pairs", type=pairs_argument, required=True)
    parser.add_argument("--async-cpu", action="store_true",
                        help="Enable one-round CPU/GPU pipelining in addition to async KV")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Capture/replay native compact model forwards; transfers stay outside graphs")
    parser.add_argument("--ratios", type=float, choices=[.05, .10, .15], nargs="+",
                        default=[.05, .10, .15], help="Run each requested ratio once")
    args = parser.parse_args()
    if len(set(args.ratios)) != len(args.ratios):
        parser.error("Each ratio may occur only once")
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log") or output.exists():
        parser.error("Use a new output directory under SparseSpec/log")
    reference_path = args.reference_payload.resolve()
    reference = json.loads(reference_path.read_text())
    rows, caps = reference["requests"], reference["output_caps"]
    kwargs = reference["kwargs"]
    if (len(rows) != 30 or len({row["sample_id"] for row in rows}) != 30
            or len(caps) != 30 or kwargs["tp_ranks"] != 2
            or kwargs["dataset_name"] != "aime24"):
        raise ValueError("Require the audited 30 distinct AIME requests and TP2 reference")
    if caps != [kwargs["clip_length"] - len(row["input_ids"]) for row in rows]:
        raise ValueError("Preserve the reference's natural-EOS full output caps")
    # Enough live cold slots for every request plus one replacement. This
    # is an admission budget, not a preallocated pinned host arena.
    cpu_pages = (len(rows) + 1) * (kwargs["clip_length"] + kwargs["spec_stride"] + 1)
    output.mkdir(parents=True)
    lock = (output / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(output / "reference.json", reference)
    state = dict(state="running", pid=os.getpid(), started_wall_time=time.time(),
                 gpu_pairs=args.gpu_pairs, pending=list(args.ratios), cases={},
                 reference_payload=str(reference_path),
                 reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                 scope=f"on-demand async KV, {'CUDA Graph' if args.cuda_graph else 'eager'}, one run per ratio",
                 async_cpu=args.async_cpu, cuda_graph=args.cuda_graph)
    active = {}

    def launch(pair, ratio, inventory):
        case = output / f"r{ratio:.2f}"
        case.mkdir()
        config = dict(kwargs, kv_cache="compact", compact_async_transfers=True,
                      compact_cpu_pages=cpu_pages, budget_ratio=ratio,
                      cuda_graph=args.cuda_graph, async_cpu=args.async_cpu)
        save(case / "worker-payload.json", dict(output=str(case), requests=rows,
                                                output_caps=caps, kwargs=config))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=2", "--",
                   str(REPO / "experiments/reproduction/run_compact_acceptance_scan.py"),
                   "--worker-payload", str(case / "worker-payload.json")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, pair)),
                   ENABLE_INTRA_NODE_COMM="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONOPTIMIZE", None)
        log = (case / "runtime.log").open("w")
        try:
            process = subprocess.Popen(command, cwd=REPO, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        except Exception:
            log.close()
            raise
        item = dict(state="running", pid=process.pid, started_wall_time=time.time(),
                    gpus=pair, gpu_inventory=inventory, command=command, kwargs=config,
                    sample_order=[row["sample_id"] for row in rows], output_caps=caps,
                    repeats=1, differences_from_reference=dict(
                        compact_async_transfers=True, compact_cpu_pages=cpu_pages,
                        budget_ratio=ratio, async_cpu=args.async_cpu,
                        cuda_graph=args.cuda_graph))
        save(case / "launch.json", item)
        state["cases"][case.name] = item
        active[pair] = (process, log, case, item)
        print("LAUNCHED", case.name, "TP2", pair, "pid", process.pid, flush=True)

    while state["pending"] or active:
        for pair, (process, log, case, item) in list(active.items()):
            code = process.poll()
            if code is None:
                for rank in range(2):
                    progress = case / f"progress.rank{rank}.json"
                    if progress.exists():
                        data = json.loads(progress.read_text())
                        item[f"progress_rank{rank}"] = {key: data[key] for key in
                            ("step", "completed", "total", "elapsed_seconds", "wall_time")}
                continue
            log.close()
            item.update(exit_code=code, ended_wall_time=time.time(),
                        state="complete" if code == 0 else "failed")
            if code == 0:
                try:
                    item["result"] = summarize(case, len(rows))
                    metrics = [json.loads((case / "engine" / f"compact-metrics.rank{rank}.json").read_text())
                               for rank in range(2)]
                    keys = ("forwards", "migrations", "restores", "request_verify_rounds")
                    if any(metrics[0][key] != metrics[1][key] for key in keys):
                        raise RuntimeError("TP counters diverged")
                    if any(m["gpu_final_free_pages"] != m["gpu_capacity_pages"] for m in metrics):
                        raise RuntimeError("GPU pool pages leaked")
                except Exception as exc:
                    item.update(state="failed", result_error=str(exc))
            save(case / "launch.json", item)
            print("FINISHED", case.name, item["state"], "exit", code, flush=True)
            del active[pair]
        busy = []
        for pair in args.gpu_pairs:
            if pair in active or not state["pending"]:
                continue
            try:
                inventory = gpu_inventory(pair)
            except RuntimeError as exc:
                busy.append(dict(gpus=pair, reason=str(exc)))
                continue
            launch(pair, state["pending"].pop(0), inventory)
        state.update(waiting_for_idle_pairs=busy, updated_wall_time=time.time())
        save(output / "controller-status.json", state)
        if state["pending"] or active:
            time.sleep(30)
    state.update(state="complete" if all(c["state"] == "complete" for c in state["cases"].values())
                 else "failed", ended_wall_time=time.time())
    save(output / "controller-status.json", state)
    lock.close()
    if state["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
