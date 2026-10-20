"""Portable fixed-k LCB128 runs with selection, p/q and actual-u diagnostics."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).resolve().parent / "inputs/lcb128-seed330.jsonl"


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(payload):
    import torch
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import serve.scheduler.async_cpu_compact_scheduler as module
    from experiments.acceptance import signal_observer
    from experiments.acceptance.uniform_observer import extend
    from experiments.reproduction.run_compact_acceptance_scan import worker as original_worker
    data = json.loads(payload.read_text())
    expected = data["source_sha256"]
    if any(digest(REPO / name) != value for name, value in expected.items()):
        raise RuntimeError("Serving sources changed after payload preparation")
    signal_observer.SignalObserver = extend(signal_observer.SignalObserver)
    signal_observer.install(module, "AsyncCPUCompactSpecScheduler", data["output"], int(os.environ["RANK"]))
    original_worker(payload)


def require_idle(gpus):
    raw = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, gpus)),
        "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    states = [[s.strip() for s in row] for row in csv.reader(raw.splitlines())]
    if {int(r[0]) for r in states} != set(gpus) or any(int(r[2]) >= 128 or int(r[3]) != 0 for r in states):
        raise RuntimeError(f"Requested GPUs {gpus} are busy; no worker launched")
    processes = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, gpus)),
        "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    if processes.strip():
        raise RuntimeError("Requested GPUs have compute processes; no worker launched")
    return states


def prepare(args):
    gpus = list(map(int, args.gpus.split(",")))
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0:
        raise ValueError("Choose exactly two distinct GPU IDs for TP2")
    if not args.ratios or len(set(args.ratios)) != len(args.ratios) or any(
            not math.isfinite(r) or not 0 < r <= .25 for r in args.ratios):
        raise ValueError("Use distinct r values in (0,.25]")
    if len({f"{r:.2f}" for r in args.ratios}) != len(args.ratios):
        raise ValueError("Ratios must have distinct two-decimal directory names")
    if not 1 <= args.requests <= 128 or not 1 <= args.batch_size <= 128 or not 1 <= args.k <= 63:
        raise ValueError("Require requests/batch in [1,128] and k in [1,63]")
    if not 0 < args.mem_fraction < 1 or (args.max_output_tokens is not None and args.max_output_tokens < 1):
        raise ValueError("Invalid memory fraction/output limit")
    model = args.model.resolve()
    identity = json.loads((REPO / "repro/model-identity.json").read_text())
    for name, value in identity["files"].items():
        if not (model / name).is_file() or digest(model / name) != value:
            raise ValueError(f"Model/tokenizer identity differs from original Qwen3-8B: {name}")
    for name, size in identity["shards"].items():
        if not (model / name).is_file() or (model / name).stat().st_size != size:
            raise ValueError(f"Model checkpoint shard missing/incomplete: {name}")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO / "log"):
        raise ValueError("Use a new directory inside this checkout's log/")
    source = {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / "serve").rglob("*.py"))}
    experiment = {str(p.relative_to(REPO)): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    baseline = json.loads((REPO / "repro/serving-source-sha256.json").read_text())
    mismatched = [name for name, value in baseline.items() if source.get(name) != value]
    if mismatched:
        raise ValueError(f"Serving differs from recorded k63 snapshot: {mismatched}")
    inputs = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()][:args.requests]
    caps = [min(row["output_cap"], args.max_output_tokens) if args.max_output_tokens is not None
            else row["output_cap"] for row in inputs]
    requests = [{k: v for k, v in row.items() if k != "output_cap"} for row in inputs]
    defaults = json.loads((REPO / "repro/serving-defaults.json").read_text())
    output.mkdir(parents=True)
    save(output / "manifest.json", dict(model_identity=identity, gpus=gpus, requests=len(inputs),
        input_sha256=digest(INPUT), sample_order=[r["sample_id"] for r in inputs],
        serving_source_sha256=source, experiment_sha256=experiment,
        command=sys.argv, diagnostic_output_cap=args.max_output_tokens,
        definition="Fixed k, cold start separate, bonus excluded; uniforms collected at verify, not predrawn"))
    for ratio in args.ratios:
        case = output / f"r{ratio:.2f}" / "lcb128-natural"
        case.mkdir(parents=True)
        kwargs = dict(defaults, model_name=str(model), spec_stride=args.k, budget_ratio=ratio,
            max_batch_size=args.batch_size, mem_fraction=args.mem_fraction,
            compact_selected_capacity=6144 if ratio <= .15 else 9216)
        if args.cpu_pages is not None:
            if args.cpu_pages <= 0:
                raise ValueError("CPU page capacity must be positive")
            kwargs["compact_cpu_pages"] = args.cpu_pages
        save(case / "worker-payload.json", dict(output=str(case), requests=requests,
            output_caps=caps, kwargs=kwargs, source_sha256=source))
    return output, gpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--ratios", nargs="+", type=float, default=[.05, .10, .15, .20, .25])
    parser.add_argument("--k", type=int, default=63)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mem-fraction", type=float, default=.8)
    parser.add_argument("--cpu-pages", type=int)
    parser.add_argument("--max-output-tokens", type=int, help="Smoke only; changes the output workload")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="CPU-only validation and payload generation")
    args = parser.parse_args()
    if args.worker_payload:
        worker(args.worker_payload.resolve())
        return
    if args.model is None or args.output is None:
        parser.error("--model and --output are required")
    output, gpus = prepare(args)
    status = dict(state="prepared", cases={}, k=args.k, gpus=gpus)
    save(output / "status.json", status)
    if args.prepare_only:
        print(f"Prepared {len(args.ratios)} cases in {output}; no GPU used")
        return
    from experiments.acceptance.verify_uniforms import validate
    for ratio in args.ratios:
        case = output / f"r{ratio:.2f}" / "lcb128-natural"
        try:
            inventory = require_idle(gpus)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                ENABLE_INTRA_NODE_COMM="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", MAX_JOBS=os.environ.get("MAX_JOBS", "2"),
                PYTHONPATH=str(REPO)+os.pathsep+os.environ.get("PYTHONPATH", ""))
            env.pop("PYTHONOPTIMIZE", None)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", "-m", "experiments.acceptance.run", "--worker-payload",
                str(case / "worker-payload.json")]
            status["state"] = "running"
            status["cases"][str(ratio)] = dict(state="running", command=command, gpu_inventory=inventory)
            save(output / "status.json", status)
            with (case / "runtime.log").open("x") as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            uniform = validate(case)
            from experiments.reproduction.run_resource_case import summarize
            summary = summarize(case, args.requests)
            if args.k == 63:
                from experiments.acceptance.analyze import validate_case
                save(case / "validation.json", validate_case(case))
            save(case / "summary.json", dict(summary, uniform_validation=uniform))
            status["cases"][str(ratio)]["state"] = "complete"
            save(output / "status.json", status)
            print(f"Completed r={ratio:.2f}: {summary}", flush=True)
        except BaseException as error:
            status["state"] = "failed"
            status["error"] = repr(error)
            save(output / "status.json", status)
            raise
    status["state"] = "complete"
    save(output / "status.json", status)
    if args.k == 63:
        from experiments.acceptance.analyze import analyze_all
        analyze_all(output)


if __name__ == "__main__":
    main()
