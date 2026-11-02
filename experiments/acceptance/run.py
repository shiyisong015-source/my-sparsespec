"""Portable LCB128 runs; probability/noise/selection diagnostics are opt-in."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

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
    data = json.loads(payload.read_text())
    if data["kwargs"].get("compact_dynamic_admission"):
        import serve.scheduler.dynamic_compact_scheduler as module
        scheduler_name = "DynamicCompactSpecScheduler"
    else:
        import serve.scheduler.async_cpu_compact_scheduler as module
        scheduler_name = "AsyncCPUCompactSpecScheduler"
    from experiments.reproduction import run_compact_acceptance_scan as original_module
    expected = data["source_sha256"]
    if any(digest(REPO / name) != value for name, value in expected.items()):
        raise RuntimeError("Serving sources changed after payload preparation")
    if any(digest(REPO / name) != value for name, value in data.get("experiment_sha256", {}).items()):
        raise RuntimeError("Experiment sources changed after payload preparation")
    record_signals = data.get("record_signals", False)
    if not record_signals and (data.get("full_vocab_trace") or data.get("trace_top_k")):
        raise ValueError("Vocabulary/noise traces require explicit record_signals=true")
    if record_signals:
        from experiments.acceptance import signal_observer
        from experiments.acceptance.uniform_observer import extend
        signal_observer.SignalObserver = extend(signal_observer.SignalObserver)
        if data.get("full_vocab_trace", False):
            from experiments.acceptance.full_trace_observer import extend as extend_full
            signal_observer.SignalObserver = extend_full(signal_observer.SignalObserver)
        if data.get("trace_top_k"):
            from experiments.acceptance.topk_trace_observer import extend as extend_top
            signal_observer.SignalObserver = extend_top(signal_observer.SignalObserver, data["trace_top_k"])
        signal_observer.install(module, scheduler_name, data["output"], int(os.environ["RANK"]))
    if data.get("max_verify_rounds"):
        from experiments.acceptance.topk_trace_observer import round_limit_class
        original_audit = original_module.completion_audit_class
        original_module.completion_audit_class = lambda base: round_limit_class(
            original_audit(base), data["max_verify_rounds"])
    original_module.worker(payload)


def require_idle(gpus):
    for attempt in range(6):
        raw = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, gpus)),
            "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        states = [[s.strip() for s in row] for row in csv.reader(raw.splitlines())]
        if {int(r[0]) for r in states} != set(gpus) or any(int(r[2]) >= 128 for r in states):
            raise RuntimeError(f"Requested GPUs {gpus} are busy; no worker launched")
        processes = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, gpus)),
            "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=15)
        if processes.strip():
            raise RuntimeError("Requested GPUs have compute processes; no worker launched")
        if all(int(r[3]) == 0 for r in states):
            return states
        # A finished worker can leave a nonzero utilization sampling window.
        # Wait only when memory and process checks already show no other job.
        if attempt < 5:
            time.sleep(1)
    raise RuntimeError(f"Requested GPUs {gpus} still report utilization; no worker launched")


def prepare(args):
    if not args.record_signals and (args.full_vocab_trace or args.trace_top_k is not None):
        raise ValueError("Add --record-signals to explicitly enable vocabulary/noise diagnostics")
    gpus = list(map(int, args.gpus.split(",")))
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0:
        raise ValueError("Choose exactly two distinct GPU IDs for TP2")
    if not args.ratios or len(set(args.ratios)) != len(args.ratios) or any(
            not math.isfinite(r) or not 0 < r <= .25 for r in args.ratios):
        raise ValueError("Use distinct r values in (0,.25]")
    if len({f"{r:.2f}" for r in args.ratios}) != len(args.ratios):
        raise ValueError("Ratios must have distinct two-decimal directory names")
    if not 1 <= args.requests <= 128 or not 1 <= args.batch_size <= 128 or not 4 <= args.k <= 63:
        raise ValueError("Require requests/batch in [1,128] and k in [4,63]")
    if args.full_vocab_trace and (args.requests > 10 or len(args.ratios) != 1):
        raise ValueError("Full vocabulary diagnostics require at most 10 requests and one r")
    if args.trace_top_k is not None:
        if args.full_vocab_trace or not 1 <= args.trace_top_k <= 100 or args.requests > 10 or len(args.ratios) != 1:
            raise ValueError("Top-K diagnostics require K<=100, <=10 requests, one r, and no full trace")
    if args.max_verify_rounds is not None and args.max_verify_rounds < 1:
        raise ValueError("Per-request verify round limit must be positive")
    if not 0 < args.mem_fraction < 1 or (args.max_output_tokens is not None and args.max_output_tokens < 1):
        raise ValueError("Invalid memory fraction/output limit")
    if args.dynamic_admission:
        budget = 512 if args.max_num_batched_tokens is None else args.max_num_batched_tokens
        if not args.k + 1 <= budget < 32768:
            raise ValueError("Require k+1 <= per-forward token budget < 32768")
    elif args.max_num_batched_tokens is not None:
        raise ValueError("--max-num-batched-tokens requires --dynamic-admission")
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
        record_signals=args.record_signals, full_vocab_trace=args.full_vocab_trace,
        trace_top_k=args.trace_top_k, max_verify_rounds=args.max_verify_rounds,
        admission_mode="dynamic_current_stage" if args.dynamic_admission else "legacy_peak_reservation",
        max_num_batched_tokens=budget if args.dynamic_admission else None,
        definition=("k upper bound with pressure-triggered shorter windows" if args.dynamic_admission else "Fixed k")
            + "; cold start separate, bonus excluded; signal diagnostics "
            + ("enabled, uniforms collected at verify, not predrawn" if args.record_signals else "disabled")))
    for ratio in args.ratios:
        case = output / f"r{ratio:.2f}" / "lcb128-natural"
        case.mkdir(parents=True)
        kwargs = dict(defaults, model_name=str(model), spec_stride=args.k, budget_ratio=ratio,
            max_batch_size=args.batch_size, mem_fraction=args.mem_fraction,
            compact_selected_capacity=6144 if ratio <= .15 else 9216)
        if args.dynamic_admission:
            kwargs.update(compact_dynamic_admission=True, max_num_batched_tokens=budget)
        if args.cpu_pages is not None:
            if args.cpu_pages <= 0:
                raise ValueError("CPU page capacity must be positive")
            kwargs["compact_cpu_pages"] = args.cpu_pages
        save(case / "worker-payload.json", dict(output=str(case), requests=requests,
            output_caps=caps, kwargs=kwargs, source_sha256=source,
            record_signals=args.record_signals,
            full_vocab_trace=args.full_vocab_trace, trace_top_k=args.trace_top_k,
            max_verify_rounds=args.max_verify_rounds, experiment_sha256=experiment))
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
    parser.add_argument("--dynamic-admission", action="store_true",
                        help="Use current-stage KV admission, multi-prefill and pressure-triggered verify")
    parser.add_argument("--max-num-batched-tokens", type=int,
                        help="Per-forward compute token budget; default 512 with dynamic admission")
    parser.add_argument("--cpu-pages", type=int)
    parser.add_argument("--max-output-tokens", type=int, help="Smoke only; changes the output workload")
    parser.add_argument("--record-signals", action="store_true",
                        help="Opt in to extra probability/U/selection diagnostics; disabled by default")
    parser.add_argument("--full-vocab-trace", action="store_true",
                        help="Record actual E and full FP32 Q/P on rank 0; at most 10 requests, one r")
    parser.add_argument("--trace-top-k", type=int,
                        help="Record union(Q top-K, P top-K, sampled token) and actual E/U; K<=100")
    parser.add_argument("--max-verify-rounds", type=int,
                        help="Retire each request after this many completed verify rounds; natural EOS can occur earlier")
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
            uniform = dict(enabled=False, reason="Extra signal diagnostics disabled")
            if args.record_signals:
                from experiments.acceptance.verify_uniforms import validate
                uniform = validate(case)
            if args.full_vocab_trace:
                from experiments.acceptance.full_trace_observer import validate as validate_full
                full_trace = validate_full(case)
                if not full_trace["successful"]:
                    raise RuntimeError("Full vocabulary diagnostic failed validation")
            if args.trace_top_k is not None:
                from experiments.acceptance.topk_trace_observer import validate as validate_top
                validate_top(case)
            from experiments.reproduction.run_resource_case import summarize
            summary = summarize(case, args.requests)
            if args.max_verify_rounds is not None:
                audit = json.loads((case / "request-audit.json").read_text())
                summary["verify_round_limited_requests"] = sum(
                    r["finish_reason"] == "verify_round_limit" for r in audit)
            if args.record_signals and args.k == 63 and not args.dynamic_admission:
                from experiments.acceptance.analyze import validate_case
                save(case / "validation.json", validate_case(case))
            save(case / "summary.json", dict(summary, record_signals=args.record_signals,
                                              uniform_validation=uniform))
            status["cases"][str(ratio)]["state"] = "complete"
            save(output / "status.json", status)
            print(f"Completed r={ratio:.2f}: {summary}", flush=True)
        except BaseException as error:
            status["state"] = "failed"
            status["error"] = repr(error)
            if str(ratio) in status["cases"]:
                status["cases"][str(ratio)].update(state="failed", error=repr(error))
            save(output / "status.json", status)
            raise
    status["state"] = "complete"
    save(output / "status.json", status)
    if args.record_signals and args.k == 63 and not args.dynamic_admission:
        from experiments.acceptance.analyze import analyze_all
        analyze_all(output)


if __name__ == "__main__":
    main()
