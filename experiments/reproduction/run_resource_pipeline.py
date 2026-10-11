"""Gate the ratio sweep on one full case; preserve failed cases without fallback."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_resource_case.py")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def idle_pairs():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    states = {int(cells[0]): (int(cells[1]), int(cells[2])) for cells in
              ([cell.strip() for cell in line.split(",")] for line in raw.splitlines())}
    return [pair for pair in PAIRS if all(gpu in states and states[gpu][0] < 128
                                        and states[gpu][1] == 0 for gpu in pair)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nsys", required=True, type=Path)
    parser.add_argument("--idle-grace-seconds", type=float, default=60)
    parser.add_argument("--preflights", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log") or output.exists():
        parser.error("Use a new pipeline directory under SparseSpec/log")
    if not args.nsys.is_file() or args.idle_grace_seconds < 0:
        parser.error("Invalid nsys path/idle grace")
    for preflight in args.preflights:
        report = json.loads((preflight / "launch.json").read_text())
        if report["state"] != "complete":
            parser.error(f"Preflight has not passed: {preflight}")
    output.mkdir(parents=True)
    ratios = [.05, .10, .15]
    first = dict(dataset="livecodebench", ratio=.05, kind="scan")
    scans = [dict(dataset=dataset, ratio=ratio, kind="scan")
             for dataset in ["aime24", "olympiadbench", "livecodebench"] for ratio in ratios
             if (dataset, ratio) != (first["dataset"], first["ratio"])]
    random.Random(330).shuffle(scans)
    diagnostics = [dict(dataset="livecodebench", ratio=ratio, kind=kind)
                   for kind in ["nsys", "selection"] for ratio in ratios]
    pending, active, completed, failed = [first] + scans + diagnostics, {}, [], []
    free_since = {}
    pair_cursor = 2
    analysis_written = False
    status_path = output / "pipeline-status.json"

    def key(job):
        return f"{job['kind']}-{job['dataset']}-r{job['ratio']:.2f}"

    def save(state, error=None):
        report = dict(state=state, pid=os.getpid(), completed=completed, failed=failed,
                      pending=[key(job) for job in pending],
                      active={name: {k: str(v) if isinstance(v, Path) else v for k, v in item.items()
                                     if k not in {"process", "log"}} for name, item in active.items()},
                      idle_grace_seconds=args.idle_grace_seconds,
                      initial_full_case=key(first), preflights=[str(path.resolve()) for path in args.preflights])
        if error:
            report["error"] = str(error)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(status_path)

    save("running")
    try:
        while pending or active:
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                report = json.loads((item["output"] / "launch.json").read_text())
                if code != 0 or report["state"] != "complete":
                    failed.append(dict(case=name, exit_code=code, output=str(item["output"])))
                    if name == key(first):
                        raise RuntimeError("First full case failed; sweep remains gated")
                else:
                    if item["kind"] == "nsys":
                        subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_nsys_resources.py")),
                                        "--case", str(item["output"]), "--nsys", str(args.nsys.resolve())],
                                       check=True, cwd=REPO)
                    completed.append(name)
                del active[name]
            if failed:
                raise RuntimeError("A case failed; remaining launches stopped without changing its parameters")
            scan_paths = [output / name for name in completed if name.startswith("scan-")]
            if len(scan_paths) == 9 and not analysis_written:
                subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_resource_cases.py")),
                                "--cases", *map(str, scan_paths), "--output", str(output / "analysis")],
                               check=True, cwd=REPO)
                analysis_written = True
            free = idle_pairs()
            now = time.monotonic()
            for pair in list(free_since):
                if pair not in free:
                    del free_since[pair]
            for pair in free:
                free_since.setdefault(pair, now)
            eligible = [pair for pair in free if now - free_since[pair] >= args.idle_grace_seconds
                        and not any(tuple(item["gpus"]) == pair for item in active.values())]
            eligible.sort(key=lambda pair: (PAIRS.index(pair) - pair_cursor) % len(PAIRS))
            for pair in eligible:
                if not pending or (key(first) not in completed and active):
                    break
                # Profile a single GPU pair at a time to isolate the diagnostic timeline.
                if any(item["kind"] == "nsys" for item in active.values()):
                    break
                if pending[0]["kind"] == "nsys" and active:
                    break
                # Complete unprofiled scans before running the diagnostic workload again.
                if pending[0]["kind"] != "scan" and any(item["kind"] == "scan" for item in active.values()):
                    break
                job = pending.pop(0)
                name = key(job)
                directory = output / name
                command = [sys.executable, str(RUNNER), "--inputs", str(args.inputs.resolve() / f"{job['dataset']}.json"),
                           "--output", str(directory), "--gpus", ",".join(map(str, pair)), "--ratio", str(job["ratio"])]
                if job["kind"] == "nsys":
                    command += ["--nsys", str(args.nsys.resolve()), "--capture-kind", "both"]
                elif job["kind"] == "selection":
                    command += ["--selection-probe"]
                log = (output / f"{name}-controller.log").open("w")
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active[name] = dict(process=process, log=log, pid=process.pid, output=directory,
                                    kind=job["kind"], gpus=pair, command=command)
                pair_cursor = (PAIRS.index(pair) + 1) % len(PAIRS)
                free_since.pop(pair, None)
            save("running" if active else "waiting_idle_gpu_pairs")
            time.sleep(10)
        scan_paths = [output / name for name in completed if name.startswith("scan-")]
        subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_resource_cases.py")),
                        "--cases", *map(str, scan_paths), "--output", str(output / "analysis")], check=True, cwd=REPO)
        selection_paths = [output / name for name in completed if name.startswith("selection-")]
        subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_footprints.py")),
                        "--cases", *map(str, selection_paths), "--output", str(output / "footprint-analysis")],
                       check=True, cwd=REPO)
        save("complete")
    except Exception as error:
        save("failed", str(error))
        # Existing jobs keep their artifacts and may finish normally; no external processes are touched.
        raise


if __name__ == "__main__":
    main()
