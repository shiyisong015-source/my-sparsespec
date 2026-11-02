"""Multi-GPU TP1 batch sweep, one job per GPU CPU-affinity/PCIe locality group.

The root controller only launches user workers via the existing shell launcher.
Each case retains a separate Nsight session, preflight evidence and raw files.
"""
import argparse
import csv
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import pwd
import queue
import signal
import subprocess
import threading
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "log/component-profile-verify-waits-tp1-k8-20261102"


def is_idle(row):
    return not row["pids"] and row["used_bytes"] <= 128 * 1024**2


def select_idle_gpu(devices, gpus, rotation):
    by_gpu = {row["gpu"]: row for row in devices}
    for offset in range(len(gpus)):
        index = (rotation + offset) % len(gpus)
        gpu = gpus[index]
        if is_idle(by_gpu[gpu]):
            return gpu, index + 1
    return None, rotation


def resume_batches(jobs, batches):
    missing = [batch for batch in batches if str(batch) not in jobs]
    if missing:
        raise ValueError(f"Batches absent from original sweep: {missing}")
    return [batch for batch in batches if jobs[str(batch)]["status"] != "complete"]


def placement(gpus):
    import pynvml as nvml
    nvml.nvmlInit()
    rows = []
    try:
        for gpu in gpus:
            handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle) + nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            masks = nvml.nvmlDeviceGetCpuAffinity(handle, (os.cpu_count() + 63) // 64)
            cpus = [i for i in os.sched_getaffinity(0) if masks[i // 64] & (1 << (i % 64))]
            if not cpus:
                raise RuntimeError(f"GPU{gpu} has no allowed local CPUs")
            rows.append(dict(gpu=gpu, cpu_affinity=sorted(cpus), used_bytes=memory.used,
                             pids=sorted({p.pid for p in processes})))
    finally:
        nvml.nvmlShutdown()
    # Group overlapping locality masks conservatively. On this host 2/3 and
    # 6/7 respectively share locality and PCIe switches. Do not claim isolated
    # bandwidth if two jobs intentionally share that domain.
    groups = []
    for row in rows:
        matches = [g for g in groups if set(g["cpus"]) & set(row["cpu_affinity"])]
        merged = dict(gpus=[row["gpu"]], cpus=row["cpu_affinity"])
        for group in matches:
            groups.remove(group)
            merged["gpus"] += group["gpus"]
            merged["cpus"] = sorted(set(merged["cpus"]) | set(group["cpus"]))
        groups.append(merged)
    for group in groups:
        group["gpus"].sort(key=gpus.index)
    groups.sort(key=lambda group:min(gpus.index(g) for g in group["gpus"]))
    return rows, groups


def combine_tables(root, jobs, owner):
    rows = []
    for batch, job in sorted(jobs.items(), key=lambda item:int(item[0])):
        if job["status"] not in ("complete","failed"):
            continue
        path = Path(job.get("output", "")) / "component-latencies.csv"
        if not path.is_file():
            continue
        with path.open() as source:
            rows += [dict(physical_gpu=job["gpu"], job_status=job["status"], **r)
                     for r in csv.DictReader(source)]
    if not rows:
        return
    path = root / "component-latencies.csv"
    with path.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.chown(path, *owner)
    lines = ["# Batch扫描：组件延迟", "", "各档使用同一真实请求集合。每个任务TP1，同一CPU/PCIe局部性组内不并行。", "",
             "各时段独立列出；同一任务的窗口不是独立重复实验，不将间隔期计入采样时间。",
             "细kernel仅代表抽样forward的实测工作时间；父区间、子项、异步传输及CPU等待不可重复相加。",
             "缺失细kernel表示未采集。原始数据、D/V/P分组及完整采集状态见jobs/bN目录。", "",
             "| Batch/窗口 | GPU | 支线 | 类型 | 组件 | 次数 | 均值ms | P50ms | P90ms | 实测累计ms |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["cohort"] not in ("all_window", "sampled_forwards"):
            continue
        lines.append(f"| {row['case']} | {row['physical_gpu']} | {row['lane']} | {row['domain']} | {row['label']} | "
                     f"{row['samples']} | {float(row['mean_ms']):.3f} | {float(row['p50_ms']):.3f} | "
                     f"{float(row['p90_ms']):.3f} | {float(row['measured_sum_ms']):.3f} |")
    path = root / "component-latencies_zh.md"
    path.write_text("\n".join(lines) + "\n");os.chown(path, *owner)


def main(args):
    devices, groups = placement(args.gpus)
    plan = dict(gpus=devices, locality_groups=groups, batches=args.batches, tp=1,
                max_simultaneous_gpu_jobs=len(groups), buffer_batch_capacity=128,
                record_windows=args.windows,record_seconds=args.record_seconds,window_gap_seconds=args.window_gap_seconds,
                policy="One active job per overlapping GPU CPU-affinity group; rotate GPUs within each group. No cross-job TP.")
    if args.dry_run:
        print(json.dumps(plan, indent=2));return
    if os.geteuid() != 0 or not os.environ.get("SUDO_USER"):
        raise SystemExit("Run via sudo bash run_component_profile.sh --sweep; workers remain your normal user")
    user = os.environ["SUDO_USER"]
    identity = pwd.getpwnam(user)
    owner = (identity.pw_uid, identity.pw_gid)
    subprocess.run(["runuser","-u",user,"--","mkdir","-p",str(BASE)],check=True)
    sweep_lock = (BASE/"parallel-sweep.lock").open("a")
    try:
        if args.wait_for_controller:
            print("Waiting for the active sweep controller to finish; existing workers continue.",flush=True)
        fcntl.flock(sweep_lock,fcntl.LOCK_EX | (0 if args.wait_for_controller else fcntl.LOCK_NB))
    except BlockingIOError:
        raise SystemExit("Another batch sweep controller is active")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    if args.resume:
        root = args.resume.resolve()
        previous = json.loads((root/"sweep-status.json").read_text())
        original_plan = previous["plan"]
        for key in ("record_windows","record_seconds","window_gap_seconds","buffer_batch_capacity","tp"):
            if plan[key] != original_plan[key]:
                raise SystemExit(f"Resume must preserve {key}={original_plan[key]}")
        jobs = previous["jobs"]
        batches = resume_batches(jobs,args.batches)
        plan = dict(original_plan,resumptions=original_plan.get("resumptions",[]) + [dict(
            at=stamp,gpus=devices,locality_groups=groups,batches=batches)])
    else:
        root = BASE / f"hardware-sweep-{stamp}"
        if root.exists():
            raise SystemExit("Output directory already exists")
        subprocess.run(["runuser", "-u", user, "--", "mkdir", "-p", str(root)], check=True)
        jobs = {str(batch):dict(status="pending") for batch in args.batches}
        batches = args.batches
    mutex = threading.Lock()
    stopped = threading.Event()
    children = {}
    work = queue.Queue()
    for batch in batches:work.put(batch)
    def save_status():
        path = root / "sweep-status.json"
        temporary = root / "sweep-status.tmp"
        temporary.write_text(json.dumps(dict(plan=plan,jobs=jobs),indent=2)+"\n")
        os.chown(temporary,*owner);temporary.replace(path)
    save_status()
    latest = BASE / "latest-sweep-path.txt"
    latest.write_text(str(root)+"\n");os.chown(latest,*owner)
    topology_path = root / (f"topology-resume-{stamp}.txt" if args.resume else "topology.txt")
    with topology_path.open("w") as out:
        subprocess.run(["nvidia-smi","topo","-m"],stdout=out,check=True)
    os.chown(topology_path,*owner)
    print(f"Sweep output: {root}",flush=True)
    print(f"GPUs {args.gpus}; {len(groups)} concurrent locality-isolated TP1 jobs; batches {batches}",flush=True)
    print(f"Progress: cat {root}/sweep-status.json",flush=True)
    def lane(group):
        rotation = 0
        while not stopped.is_set():
            try:batch = work.get_nowait()
            except queue.Empty:return
            while not stopped.is_set():
                try:
                    current_devices,_ = placement(group["gpus"])
                    gpu,next_rotation = select_idle_gpu(current_devices,group["gpus"],rotation)
                except Exception as error:
                    with mutex:
                        jobs[str(batch)].update(status="waiting_for_gpu",gpu_check_error=str(error))
                        save_status()
                    stopped.wait(30)
                    continue
                if gpu is not None:
                    rotation = next_rotation
                    break
                with mutex:
                    jobs[str(batch)].update(status="waiting_for_gpu",last_gpu_check=current_devices)
                    save_status()
                print(f"WAIT batch={batch}; GPUs {group['gpus']} occupied; checking again in 30s",flush=True)
                stopped.wait(30)
            if stopped.is_set():
                work.task_done()
                return
            output = root / "jobs" / f"b{batch}"
            if output.exists():
                output = output / f"attempt-{stamp}-{datetime.now().strftime('%H%M%S%f')}"
            subprocess.run(["runuser","-u",user,"--","mkdir","-p",str(output)],check=True)
            environment = dict(os.environ,COMPONENT_PROFILE_GPU=str(gpu),
                COMPONENT_PROFILE_BATCHES=str(batch), COMPONENT_PROFILE_OUTPUT=str(output),
                COMPONENT_PROFILE_ROOT=str(root/"placements"/f"gpu{gpu}"),
                COMPONENT_PROFILE_PREFLIGHT=str(root/"preflights"/f"gpu{gpu}"),
                COMPONENT_PROFILE_WINDOWS=str(args.windows),COMPONENT_PROFILE_RECORD_SECONDS=str(args.record_seconds),
                COMPONENT_PROFILE_WINDOW_GAP_SECONDS=str(args.window_gap_seconds))
            with mutex:
                previous_job = jobs[str(batch)]
                attempts = previous_job.get("attempts",[])
                if "output" in previous_job:
                    attempts = attempts + [{k:v for k,v in previous_job.items() if k!="attempts"}]
                jobs[str(batch)] = dict(status="running",gpu=gpu,output=str(output),
                    attempts=attempts,started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
                save_status()
            print(f"START batch={batch} GPU={gpu}; log={output}/launcher.log",flush=True)
            try:
                with (output/"launcher.log").open("w") as log:
                    os.chown(output/"launcher.log",*owner)
                    process = subprocess.Popen(["bash",str(REPO/"experiments/diagnostics/run_component_profile.sh")],
                        env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    with mutex:children[batch] = process
                    code = process.wait()
            except Exception as error:
                code = -1
                with mutex:jobs[str(batch)]["controller_error"] = str(error)
            with mutex:
                children.pop(batch,None)
                jobs[str(batch)].update(status=("complete" if code==0 else "waiting_for_gpu" if code==75 else "failed"),returncode=code,
                    finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
                if code==75:
                    # Guard race: another workload can start after our idle check.
                    # Retain the attempt, return this batch to the queue, and retry.
                    work.put(batch)
                if code==0:
                    try:
                        coverage=json.loads((output/'component-summary.json').read_text()).get('sampling_coverage',{})
                        jobs[str(batch)].update(sampling_coverage=coverage,sampling_complete=bool(coverage) and all(
                            c['status']=='complete' for c in coverage.values()))
                    except (OSError,ValueError,KeyError) as error:
                        jobs[str(batch)].update(status='failed',summary_error=str(error))
                try:
                    combine_tables(root,jobs,owner)
                except Exception as error:
                    jobs[str(batch)].update(status="failed",merge_error=str(error))
                save_status()
            print(f"DONE batch={batch} GPU={gpu} status={jobs[str(batch)]['status']}",flush=True)
            work.task_done()
            if code==75:
                stopped.wait(30)
    threads = [threading.Thread(target=lane,args=(group,)) for group in groups]
    for thread in threads:thread.start()
    try:
        for thread in threads:thread.join()
    except KeyboardInterrupt:
        stopped.set()
        with mutex:running = list(children.values())
        for process in running:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for thread in threads:thread.join()
        for job in jobs.values():
            if job["status"] in ("pending","waiting_for_gpu"):job["status"]="cancelled"
        save_status()
        sweep_lock.close()
        raise SystemExit(f"Sweep interrupted; partial results retained in {root}")
    sweep_lock.close()
    if any(job["status"]!="complete" for job in jobs.values()):
        raise SystemExit(f"Some cases failed or were not started; inspect {root}/sweep-status.json")
    if any(not job.get('sampling_complete',False) for job in jobs.values()):
        print('Some natural-EOS runs ended before all requested windows; inspect sampling_coverage in sweep-status.json.',flush=True)
    print(f"Sweep complete: {root}/component-latencies_zh.md",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",type=int,nargs="+",default=[2,3,6,7])
    parser.add_argument("--batches",type=int,nargs="+",default=[16,32,48,64,72,128])
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--resume",type=Path,help="Retry unfinished batches, retaining all previous attempt logs")
    parser.add_argument("--wait-for-controller",action="store_true",help="Wait for the existing sweep before resuming")
    parser.add_argument("--windows",type=int,default=3)
    parser.add_argument("--record-seconds",type=float,default=180.)
    parser.add_argument("--window-gap-seconds",type=float,default=180.)
    args = parser.parse_args()
    if not 1<=len(args.gpus)<=4 or len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0:
        parser.error("Select one to four distinct nonnegative GPU indices")
    if not args.batches or len(set(args.batches))!=len(args.batches) or not all(1<=b<=128 for b in args.batches):
        parser.error("Select distinct batch limits between 1 and 128")
    if args.windows<1 or not 0<args.record_seconds<float('inf') or not 0<=args.window_gap_seconds<float('inf'):
        parser.error("Require positive windows/duration and finite nonnegative gap")
    main(args)
