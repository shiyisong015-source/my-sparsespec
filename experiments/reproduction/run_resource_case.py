"""Run one generation from audited token inputs in fresh TP2 workers.

Input JSON contains dataset, model_path, enable_thinking, and requests.
Each request contains a unique sample_id and input_ids. GPU imports stay
inside the worker, so the launcher never initializes a CUDA context.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_inputs(path, model, limit=None):
    data = json.loads(path.read_text())
    if data.get("enable_thinking") is not True:
        raise ValueError("This reproduction scan requires thinking-enabled inputs")
    if Path(data["model_path"]).resolve() != model:
        raise ValueError("Token inputs belong to a different model/tokenizer")
    if data["dataset"] not in {"aime24", "olympiadbench", "livecodebench"}:
        raise ValueError("Unexpected dataset")
    rows = data["requests"]
    if not rows or (limit is not None and len(rows) < limit):
        raise ValueError("Not enough input requests")
    if limit is not None:
        rows = rows[:limit]
    config = json.loads((model / "config.json").read_text())
    identities = set()
    for row in rows:
        identity = str(row["sample_id"])
        if identity in identities:
            raise ValueError("Each original sample_id may appear only once per configuration")
        identities.add(identity)
        ids = row["input_ids"]
        if not isinstance(ids, list) or not ids or any(
            type(token) is not int or not 0 <= token < config["vocab_size"] for token in ids
        ):
            raise ValueError("Invalid token IDs")
        if len(ids) >= 32768:
            raise ValueError("Input leaves no output room within the 32768-token context")
    if config["max_position_embeddings"] < 32768:
        raise ValueError("Model context limit is below the approved scan limit")
    return data, rows


def completion_audit_class(base_scheduler):
    """Preserve the scheduler's actual completion cause before EOS is lost."""
    class CompletionAuditedScheduler(base_scheduler):
        def is_request_complete(self, request, check_idx=-1):
            complete = super().is_request_complete(request, check_idx)
            if complete:
                if request.status.name == "NORMAL":
                    at_limit = (request.len_oracle_ctx is not None
                                and request.len_kv_cache + check_idx >= request.len_oracle_ctx - 1)
                else:
                    at_limit = (request.len_oracle_ctx is not None
                                and request.len_kv_cache >= request.len_oracle_ctx)
                request.resource_finish_reason = "length_limit" if at_limit else "eos"
            return complete
    return CompletionAuditedScheduler


def worker(payload_path):
    # Import only after torchrun sets rank and CUDA_VISIBLE_DEVICES.
    import serve.run as engine
    from serve.request.request import Request
    import torch.distributed as dist

    data = json.loads(payload_path.read_text())
    rank = int(os.environ["RANK"])
    output = Path(data["output"])
    kwargs = dict(data["kwargs"])
    kwargs["result_dir"] = str(output / "engine") if rank == 0 else None
    kwargs["acceptance_log_path"] = str(output / "rounds.jsonl") if rank == 0 else None
    rows = data["requests"]
    requests = [Request(row["input_ids"], cap)
                for row, cap in zip(rows, data["output_caps"], strict=True)]
    # run() consumes the pending list; retain references for completion auditing.
    audited_requests = list(requests)
    #completion_audit_class返回一个specScheduler子类增加对于请求完成情况的统计
    engine.SpecScheduler = completion_audit_class(engine.SpecScheduler)
    if data["observer"] is not None:
        #增加请求状态kv占用，卸载/恢复事件，以及nsys采集窗口
        from experiments.reproduction.resource_observer import scheduler_class
        engine.SpecScheduler = scheduler_class(engine.SpecScheduler, output, rank, data["observer"])
    if data["selection"] is not None:
        #读取实际参与draft的选择索引保存快照，统计不同层/head的选择病机
        from experiments.reproduction.selection_probe import scheduler_class
        engine.SpecScheduler = scheduler_class(engine.SpecScheduler, output, rank, data["selection"])
    #实际跑模型运行代码
    outputs = engine.run_inference(reqs=requests, **kwargs)
    if rank == 0:
        if len(outputs) != len(rows):
            raise ValueError("Generation did not complete every input request")
        records = []
        for request_id, (row, cap, generated) in enumerate(
            zip(rows, data["output_caps"], outputs, strict=True)
        ):
            tokens = generated["output_token_ids"]
            if (generated["id"] != request_id or generated["len_input"] != len(row["input_ids"])
                    or generated["len_output"] != len(tokens)):
                raise ValueError(f"Inconsistent output for request {request_id}")
            reason = getattr(audited_requests[request_id], "resource_finish_reason", "unclassified")
            if reason == "unclassified":
                raise ValueError(f"Unexpected early stop for request {request_id}")
            records.append(dict(request_id=request_id, sample_id=row["sample_id"],
                                input_tokens=len(row["input_ids"]),
                                output_tokens=len(tokens), output_cap=cap, finish_reason=reason,
                                tokens_beyond_output_cap=max(0, len(tokens) - cap)))
        write_json(output / "request-audit.json", records)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def summarize(output, expected_requests):
    metrics = {}
    for line in (output / "engine/summary.log").read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 4:
            try:
                metrics[cells[1]] = float(cells[2])
            except ValueError:
                pass
    records = json.loads((output / "request-audit.json").read_text())
    seconds = metrics["time (s)"]
    output_tokens = sum(row["output_tokens"] for row in records)
    # The display table rounds large token counts; aggregate exact counts from records.
    if (len(records) != expected_requests or metrics["Finished Requests"] != expected_requests
            or seconds <= 0):
        raise ValueError("Incomplete request results or invalid engine duration")
    if not (output / "rounds.jsonl").stat().st_size:
        raise ValueError("No verify rounds were recorded")
    return dict(requests=len(records), engine_seconds=seconds, output_tokens=output_tokens,
                output_tps=output_tokens / seconds,
                input_tokens=sum(row["input_tokens"] for row in records),
                eos_requests=sum(row["finish_reason"] == "eos" for row in records),
                length_limited_requests=sum(row["finish_reason"] == "length_limit" for row in records),
                tokens_beyond_output_caps=sum(row["tokens_beyond_output_cap"] for row in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", help="Two idle physical GPU indices, e.g. 0,1")
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3-8B"))
    parser.add_argument("--ratio", type=float, choices=[.05, .10, .15], default=.05)
    parser.add_argument("--phase", choices=["preflight", "scan"], default="scan")
    parser.add_argument("--limit", type=int, help="Explicit preflight or selection diagnostic request count")
    parser.add_argument("--max-output-tokens", type=int, help="Explicit preflight output cap")
    parser.add_argument("--disable-resource-observer", action="store_true", help="Instrumentation overhead control")
    parser.add_argument("--resource-interval", type=int, default=100)
    parser.add_argument("--capture-kind", choices=["none", "low", "pressure", "both"], default="none")
    parser.add_argument("--window-seconds", type=float, default=20)
    parser.add_argument("--nsys", type=Path, help="Nsight Systems executable; required for capture")
    parser.add_argument("--selection-probe", action="store_true", help="Separate footprint diagnostic")
    parser.add_argument("--selection-interval", type=int, default=500)
    parser.add_argument("--selection-max-samples", type=int, default=64)
    parser.add_argument("--selection-min-context", type=int, default=2048)
    args = parser.parse_args()
    # worker-payload配置文件
    if args.worker_payload:
        worker(args.worker_payload)
        return
    if args.inputs is None or args.output is None or args.gpus is None:
        parser.error("--inputs, --output and --gpus are required")
    if args.phase != "preflight":
        if args.max_output_tokens is not None:
            parser.error("Only explicit preflight runs may reduce output length")
        if args.limit is not None and not args.selection_probe:
            parser.error("Only preflight or selection diagnostics may reduce request count")
    if any(value is not None and value <= 0 for value in [args.limit, args.max_output_tokens]):
        parser.error("Request/output limits must be positive")
    if args.resource_interval < 1 or args.window_seconds <= 0:
        parser.error("Resource interval/window must be positive")
    if args.capture_kind != "none" and (args.nsys is None or args.disable_resource_observer):
        parser.error("Capture requires --nsys and the resource observer")
    if args.nsys is not None and (not args.nsys.is_file() or args.capture_kind == "none"):
        parser.error("Provide an existing nsys executable and a capture kind")
    if args.selection_probe and args.nsys is not None:
        parser.error("Run selection snapshots separately from nsys tracing")
    if min(args.selection_interval, args.selection_max_samples) < 1 or args.selection_min_context < 0:
        parser.error("Invalid selection snapshot limits")
    model = args.model.resolve()
    inputs = args.inputs.resolve()
    original, rows = read_inputs(inputs, model, args.limit)
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log") or output.exists():
        parser.error("Use a new directory under SparseSpec/log; existing results are preserved")
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("GPU indices must be integers")
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0:
        parser.error("TP2 requires two distinct GPU indices")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    states = {int(cells[0]): cells for cells in
              ([cell.strip() for cell in line.split(",")] for line in inventory.splitlines())}
    if any(gpu not in states or int(states[gpu][4]) >= 128 or int(states[gpu][5]) != 0 for gpu in gpus):
        parser.error("Selected GPUs are not idle")
    # Strict single-request selection bound; aggregate buffers are checked in preflight.
    if max(192, math.ceil(32768 * args.ratio)) >= 6144:
        parser.error("Selection buffer bound exceeded")
    caps = [min(args.max_output_tokens or 32768, 32768 - len(row["input_ids"])) for row in rows]
    kwargs = dict(model_name=str(model), max_batch_size=128, tp_ranks=2,
                  max_mem_per_rank=None, mem_fraction=.8, cuda_graph=True, async_cpu=True,
                  dtype="float16", kv_cache="pillar", admit_policy="offloading",
                  batch_policy="greedy", chunk_size=256, clip_length=32768,
                  spec_stride=8, budget_ratio=args.ratio, num_min_budget=192,
                  temperature=.6, seed=330, enable_torch_profiler=False,
                  step_log_path=None, dataset_name=original["dataset"])
    output.mkdir(parents=True)
    write_json(output / "inputs.json", original)
    observer = None if args.disable_resource_observer else dict(
        interval=args.resource_interval, capture_kind=args.capture_kind, window_seconds=args.window_seconds,
        low_min_step=0 if args.phase == "preflight" else 100,
        low_min_kv_tokens=0 if args.phase == "preflight" else 4096)
    selection = dict(interval=args.selection_interval, max_samples=args.selection_max_samples,
                     min_context_tokens=args.selection_min_context) if args.selection_probe else None
    write_json(output / "worker-payload.json", dict(
        requests=rows, output_caps=caps, kwargs=kwargs, output=str(output), observer=observer, selection=selection))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", "--", str(Path(__file__).resolve()),
               "--worker-payload", str(output / "worker-payload.json")]
    if args.nsys is not None:
        command = [str(args.nsys.resolve()), "profile", "--trace=cuda,nvtx", "--sample=none",
                   "--cpuctxsw=none", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                   "--capture-range-end=repeat:2" if args.capture_kind == "both" else "--capture-range-end=stop",
                   "--output=" + str(output / "nsys"), "--force-overwrite=false"] + command
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               ENABLE_INTRA_NODE_COMM="1", TOKENIZERS_PARALLELISM="false")
    env.pop("PYTHONOPTIMIZE", None)
    report = dict(state="running", phase=args.phase, pid=os.getpid(), command=command,
                  inputs=str(inputs), input_file_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                  token_ids_sha256=hashlib.sha256(json.dumps([row["input_ids"] for row in rows]).encode()).hexdigest(),
                  sample_order=[row["sample_id"] for row in rows],
                  kwargs=kwargs, output_caps=caps, gpus=gpus, gpu_inventory=inventory,
                  enable_thinking=True, single_generation=True, python_assertions_enabled=True,
                  observer=observer, selection_probe=selection, nsys_enabled=args.nsys is not None,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  git_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)).hexdigest(),
                  source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for tree in ["serve", "experiments/reproduction"]
                                 for path in sorted((REPO / tree).rglob("*.py"))},
                  packages={name: importlib.metadata.version(name) for name in ["torch", "transformers"]})
    write_json(output / "launch.json", report)
    started = time.perf_counter()
    try:
        with (output / "runtime.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["worker_launcher_pid"] = process.pid
            write_json(output / "launch.json", report)
            report["exit_code"] = process.wait()
        if report["exit_code"]:
            raise RuntimeError(f"Generation failed: exit {report['exit_code']}; see runtime.log")
        report["metrics"] = summarize(output, len(rows))
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        report["launcher_elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "launch.json", report)
    print(json.dumps(dict(state=report["state"], output=str(output), metrics=report["metrics"])))


if __name__ == "__main__":
    main()
