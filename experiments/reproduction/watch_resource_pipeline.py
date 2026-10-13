"""Periodically observe a resource experiment; resume only a healthy queue owner.

This CPU-only monitor preserves experiment parameters and failed case artifacts.
GPU/code failures are recorded for diagnosis rather than retried blindly.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from run_resource_pipeline import AttachedCase, REPO, process_running


def tail(path, limit=131072):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors="replace")


def latest_resource(path):
    for line in reversed(tail(path).splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def inspect(pipeline, stale_seconds):
    status = json.loads((pipeline / "pipeline-status.json").read_text())
    alerts, cases = [], {}
    now = time.time()
    for name, item in status["active"].items():
        directory = Path(item["output"])
        running = process_running(item["pid"])
        resource = latest_resource(directory / "resources.rank0.jsonl")
        files = [directory / filename for filename in
                 ["runtime.log", "resources.rank0.jsonl", "transfers.rank0.jsonl"]]
        mtimes = [path.stat().st_mtime for path in files if path.exists()]
        age = now - max(mtimes) if mtimes else None
        error_lines = [line[-1200:] for line in tail(directory / "runtime.log").splitlines()
                       if any(marker in line.lower() for marker in
                              ["traceback (most recent call last)", "illegal memory access",
                               "cuda out of memory", "runtimeerror:", "[error]"])]
        cases[name] = dict(pid=item["pid"], running=running, gpus=item["gpus"],
                           seconds_since_log_update=age, latest_resource=resource,
                           error_lines=error_lines[-5:])
        launch_path = directory / "launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        if not running and launch.get("state") != "complete":
            alerts.append(dict(kind="case_process_exited", case=name,
                               launch_state=launch.get("state"), exit_code=launch.get("exit_code")))
        if age is not None and age > stale_seconds and running:
            alerts.append(dict(kind="logs_stale", case=name, seconds=round(age),
                               note="No automatic termination; long initialization or waits need diagnosis."))
        if error_lines:
            alerts.append(dict(kind="runtime_error_detected", case=name, lines=error_lines[-5:]))
    if status.get("failed") or status["state"] == "failed":
        alerts.append(dict(kind="pipeline_failed", cases=status.get("failed"), error=status.get("error")))
    return status, dict(wall_time=now, timestamp=datetime.now().astimezone().isoformat(),
                        pipeline_state=status["state"], controller_pid=status["pid"],
                        controller_running=process_running(status["pid"]),
                        completed=status["completed"], pending=status["pending"],
                        cases=cases, alerts=alerts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--controller-launcher", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--stale-seconds", type=float, default=1800)
    parser.add_argument("--once", action="store_true", help="Read-only check; do not resume a controller")
    args = parser.parse_args()
    pipeline = args.pipeline.resolve()
    if not pipeline.is_relative_to(REPO / "log") or args.interval < 30 or args.stale_seconds < 60:
        parser.error("Use SparseSpec/log and positive monitoring intervals")
    config = json.loads(args.controller_launcher.read_text())
    command = config["command"]
    if Path(command[1]).resolve() != Path(__file__).with_name("run_resource_pipeline.py"):
        parser.error("Launcher does not belong to the resource pipeline")
    if Path(command[command.index("--output") + 1]).resolve() != pipeline:
        parser.error("Launcher output differs from the monitored pipeline")
    lock = (pipeline / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seen = set()
    restart_attempts = 0
    child = None
    with (pipeline / "watch-checks.jsonl").open("a", buffering=1) as checks, \
            (pipeline / "watch-alerts.jsonl").open("a", buffering=1) as alert_log:
        while True:
            if child is not None:
                child.poll()  # Reap a restarted controller once it exits.
            try:
                status, report = inspect(pipeline, args.stale_seconds)
                can_resume = (not report["controller_running"]
                              and status["state"] in {"running", "waiting_idle_gpu_pairs"}
                              and not report["alerts"] and restart_attempts < 2)
                if can_resume and not args.once:
                    # Verify existing case identities before adopting their CUDA workers.
                    for item in status["active"].values():
                        AttachedCase(item["pid"], item["output"], item["command"])
                    restart_attempts += 1
                    log = pipeline / f"watch-controller-restart-{restart_attempts}.log"
                    with log.open("a") as stream:
                        child = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                    report["controller_restart"] = dict(pid=child.pid, command=command, log=str(log))
                elif not report["controller_running"] and status["state"] != "complete":
                    report["alerts"].append(dict(kind="controller_not_running", state=status["state"]))
                report["watch_pid"] = os.getpid()
                write_json(pipeline / "watch-status.json", report)
                checks.write(json.dumps(report, ensure_ascii=False) + "\n")
                for alert in report["alerts"]:
                    identity = json.dumps({k: v for k, v in alert.items() if k != "seconds"}, sort_keys=True)
                    if identity not in seen:
                        alert_log.write(json.dumps(dict(timestamp=report["timestamp"], **alert),
                                                   ensure_ascii=False) + "\n")
                        seen.add(identity)
                if args.once or status["state"] == "complete":
                    break
            except Exception as error:
                alert_log.write(json.dumps(dict(timestamp=datetime.now().astimezone().isoformat(),
                                               kind="monitor_error", error=repr(error))) + "\n")
                if args.once:
                    raise
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
