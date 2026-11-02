"""Interval-union overlap and hardware-counter summaries, per TP GPU."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sqlite3

import numpy as np


def process_id(global_id):
    # Nsight: hardware:8, VM:8, process:24, thread/undefined:24.
    return (global_id >> 24) & 0xFFFFFF


def union(values, lower, upper):
    result = []
    for a, b in sorted((max(a, lower), min(b, upper)) for a, b in values):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(values):
    return sum(b - a for a, b in values)


def overlap(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def contains(times, intervals):
    if not intervals:
        return np.zeros(len(times), dtype=bool)
    a = np.array(intervals)
    idx = np.searchsorted(a[:, 0], times, side="right") - 1
    return (idx >= 0) & (times < a[np.maximum(idx, 0), 1])


def analyze(case):
    db = sqlite3.connect(f"file:{case / 'nsys.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    windows, transfer_ranges, forward_ranges = {}, defaultdict(list), defaultdict(list)
    for row in db.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        text = row["text"] or names.get(row["textId"], "")
        if text == "diag.capture.resource":
            windows[process_id(row["globalTid"])] = [row["start"], row["end"]]
        elif text.startswith("diag.kv."):
            transfer_ranges[row["globalTid"]].append((row["start"], row["end"], text))
        elif text.startswith("diag.forward "):
            forward_ranges[row["globalTid"]].append((row["start"], row["end"], text))
    if len(windows) != 2:
        raise RuntimeError(f"Need complete windows from two TP ranks, got {windows}")
    lower, upper = max(v[0] for v in windows.values()), min(v[1] for v in windows.values())
    if upper <= lower:
        raise RuntimeError("No common TP window")
    api_transfer, api_forward = {}, set()
    for ranges in [transfer_ranges, forward_ranges]:
        for rows in ranges.values():
            rows.sort()
    starts_t = {tid:[r[0] for r in rows] for tid, rows in transfer_ranges.items()}
    starts_f = {tid:[r[0] for r in rows] for tid, rows in forward_ranges.items()}
    for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
        if table not in tables:
            continue
        for row in db.execute(f"SELECT * FROM {table}"):
            tid, pid = row["globalTid"], process_id(row["globalTid"])
            key = (pid, row["correlationId"])
            for ranges, starts, is_transfer in [(transfer_ranges, starts_t, True), (forward_ranges, starts_f, False)]:
                if tid not in ranges:
                    continue
                idx = bisect_right(starts[tid], row["start"]) - 1
                if idx >= 0 and row["end"] <= ranges[tid][idx][1]:
                    if is_transfer:
                        api_transfer[key] = ranges[tid][idx][2]
                    else:
                        api_forward.add(key)
    kernels = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?", (lower, upper))
               if process_id(r["globalPid"]) in windows]
    forward_streams = defaultdict(set)
    for row in kernels:
        pid = process_id(row["globalPid"])
        name = names.get(row["demangledName"], "")
        if ((pid, row["correlationId"]) in api_forward or
            re.search(r"attention|gemm|xmma", name, re.I)) and not any(n in name for n in ("_scatter_kv_pages", "_restore_kv_pages_bounded")):
            forward_streams[pid].add(row["streamId"])
    categories = defaultdict(lambda: defaultdict(list))
    counts, top = CounterLike(), defaultdict(lambda: [0, 0])
    devices = {}
    for row in kernels:
        pid = process_id(row["globalPid"])
        key = (pid, row["correlationId"])
        devices[pid] = row["deviceId"]
        name = names.get(row["demangledName"], names.get(row["shortName"], "unknown"))
        label = api_transfer.get(key, "")
        if "_pack_selected_kv" in name:
            category = "D2H"
        elif any(n in name for n in ("_scatter_kv_pages", "_restore_kv_pages_bounded")):
            category = "H2D" if "diag.kv.restore " in label else "hot_gather" if "diag.kv.repack " in label else "unknown_scatter"
        elif label:
            category = "transfer_setup"
        else:
            category = "compute"
        interval = (row["start"], row["end"])
        categories[pid][category].append(interval)
        categories[pid]["any_kernel"].append(interval)
        if category == "compute":
            if row["streamId"] in forward_streams[pid]:
                categories[pid]["forward_stream"].append(interval)
            if re.search(r"attention", name, re.I):
                categories[pid]["attention"].append(interval)
        counts[(pid, category)] += 1
        top[(pid, category, name)][0] += 1
        top[(pid, category, name)][1] += min(row["end"], upper) - max(row["start"], lower)
    groups = {pid:{k:union(v, lower, upper) for k, v in c.items()} for pid, c in categories.items()}
    physical_devices = {}
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        mapping = {(r["pid"], r["cudaId"]):r["gpuId"] for r in db.execute("SELECT * FROM TARGET_INFO_CUDA_DEVICE")}
        physical_devices = {pid:mapping.get((pid,dev)) for pid,dev in devices.items()}
        if any(dev is None for dev in physical_devices.values()):
            raise RuntimeError("Could not map process/CUDA ordinal to physical GPU")
    results = []
    for pid in sorted(windows):
        g = groups[pid]
        restore, compute, forward = g.get("H2D", []), g.get("compute", []), g.get("forward_stream", [])
        transfer_ns = duration(restore)
        if not transfer_ns or counts[(pid, "unknown_scatter")]:
            raise RuntimeError("Missing or ambiguous restore-kernel attribution")
        item = dict(pid=pid, trace_device=devices[pid], physical_gpu=physical_devices.get(pid), window_seconds=(upper-lower)/1e9,
            H2D_kernel_count=counts[(pid, "H2D")], H2D_seconds=transfer_ns/1e9,
            H2D_without_any_compute_seconds=(transfer_ns-overlap(restore,compute))/1e9,
            H2D_without_any_compute_fraction=1-overlap(restore,compute)/transfer_ns,
            H2D_without_forward_stream_seconds=(transfer_ns-overlap(restore,forward))/1e9,
            H2D_without_forward_stream_fraction=1-overlap(restore,forward)/transfer_ns,
            compute_window_fraction=duration(compute)/(upper-lower),
            no_kernel_window_fraction=1-duration(g.get("any_kernel",[]))/(upper-lower),
            category_kernel_counts={k:counts[(pid,k)] for k in g if k not in ("any_kernel", "forward_stream", "attention")})
        results.append(item)
    metrics, metric_series = [], {}
    if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        catalog = [dict(r) for r in db.execute("SELECT * FROM TARGET_INFO_GPU_METRICS")]
        (case / "metrics-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        for entry in catalog:
            metric_name = entry["metricName"]
            if not re.search(r"SM.*Active|SM.*Issue|Tensor.*Active|DRAM.*|PCI.*|Unallocated.*Warp|Warp.*Occup|Occupancy", metric_name, re.I):
                continue
            samples = list(db.execute("SELECT timestamp,value FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
                (entry["typeId"], entry["metricId"], lower, upper)))
            if not samples:
                continue
            # Nsight's own GPU metric recipe maps typeId & 0xFF to GPU id.
            device = entry["typeId"] & 0xFF
            matching = [pid for pid, dev in physical_devices.items() if dev == device]
            if len(matching) != 1:
                metrics.append(dict(name=metric_name, device=device, status="unmatched_device"))
                continue
            pid = matching[0]
            times = np.array([r[0] for r in samples], dtype=np.int64)
            values = np.array([r[1] for r in samples], dtype=float)
            if re.search(r"SM.*Active|SM.*Issue|Tensor.*Active", metric_name, re.I):
                metric_series[(pid, metric_name)] = (times, values)
            restores = contains(times, groups[pid].get("H2D", []))
            forwards = contains(times, groups[pid].get("forward_stream", []))
            for label, mask in [("window", np.ones(len(times),bool)), ("restore",restores),
                                ("restore_without_forward", restores & ~forwards),
                                ("restore_with_forward", restores & forwards)]:
                a = values[mask & np.isfinite(values)]
                if len(a):
                    metrics.append(dict(name=metric_name, device=device, condition=label,
                        samples=len(a), mean=float(np.mean(a)), p25=float(np.percentile(a,25)),
                        median=float(np.median(a)), p90=float(np.percentile(a,90))))
    report = dict(case=str(case), window_ns=[lower,upper], devices=results, hardware_metrics=metrics,
        source_unchanged=verify_sources(case), notes=[
            "H2D means mapped-host restoration kernel execution, not the entire CPU submit-to-commit interval.",
            "Compute includes non-transfer sampling/control/communication kernels; forward_stream additionally isolates streams with model launches or attention/GEMM.",
            "A running kernel does not measure SM utilization. Only hardware_metrics contains sampled hardware counters.",
            "SM Active includes the SMs used by mapped-host transfer itself; low Tensor Active alone does not prove spare schedulable resources.",
            "Hardware summaries use uniformly sampled points in each interval, not additive kernel durations.",
            "This is a profiled mechanism diagnostic, not an end-to-end throughput comparison."])
    (case / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (case / "top-kernels.json").write_text(json.dumps([dict(pid=k[0],category=k[1],name=k[2],calls=v[0],gpu_seconds=v[1]/1e9)
        for k,v in sorted(top.items(),key=lambda x:x[1][1],reverse=True)[:100]],indent=2)+"\n")
    plot(case, groups, lower, upper)
    if metric_series:
        plot_hardware(case, groups, metric_series, lower, upper)
    audit_submission(case, db, kernels, transfer_ranges, forward_ranges, names)
    print(json.dumps(dict(case=str(case), devices=results,
        hardware_metric_summaries=len(metrics), source_unchanged=report["source_unchanged"]), ensure_ascii=False, indent=2))


class CounterLike(defaultdict):
    def __init__(self):
        super().__init__(int)


def verify_sources(case):
    import hashlib
    payload = json.loads((case / "worker-payload.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    changed = [name for name,h in payload["source_sha256"].items() if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=h]
    if changed:
        raise RuntimeError(f"Serving sources changed: {changed}")
    return True


def audit_submission(case, db, kernels, transfer_ranges, forward_ranges, names):
    ranges = [(process_id(tid),*r) for tid,rows in transfer_ranges.items() for r in rows]
    runtime = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
    by_correlation = defaultdict(list)
    for row in runtime:
        by_correlation[(process_id(row["globalTid"]),row["correlationId"])].append(row)
    records = []
    for kernel in kernels:
        if not any(n in names.get(kernel["shortName"], "") for n in ("_scatter_kv_pages", "_restore_kv_pages_bounded")):
            continue
        pid = process_id(kernel["globalPid"])
        calls = by_correlation[(pid,kernel["correlationId"])]
        labels = [label for pr,a,b,label in ranges if pr==pid and label.startswith("diag.kv.restore ")
                  for api in calls if a <= api["start"] and api["end"] <= b]
        if not labels:
            continue
        label = labels[-1]
        start, end = kernel["start"], kernel["end"]
        forward_calls = [r for tid,rows in forward_ranges.items() for r in rows
                         if process_id(tid)==pid and r[0] < end and r[1] > start]
        synchronizes = [dict(name=names.get(r["nameId"],""),
            start_from_restore_ms=(r["start"]-start)/1e6,
            wall_ms=(r["end"]-r["start"])/1e6)
            for r in runtime if process_id(r["globalTid"])==pid
            and r["start"]<end and r["end"]>start
            and "Synchronize" in names.get(r["nameId"],"") and r["end"]-r["start"]>1_000_000]
        queued_metadata = []
        for k in kernels:
            if process_id(k["globalPid"])!=pid or "concat_kv_indices_contiguous" not in names.get(k["shortName"],""):
                continue
            apis = by_correlation[(pid,k["correlationId"])]
            for api in apis:
                if start <= api["start"] < end:
                    queued_metadata.append(dict(name=names[k["shortName"]],
                        launch_from_restore_ms=(api["start"]-start)/1e6,
                        execution_from_restore_ms=(k["start"]-start)/1e6,
                        launch_to_execution_ms=(k["start"]-api["end"])/1e6,
                        execution_ms=(k["end"]-k["start"])/1e6))
        records.append(dict(pid=pid,label=label,restore_ms=(end-start)/1e6,
            forward_CPU_ranges_during_restore=len(forward_calls),
            synchronize_calls=synchronizes,queued_metadata=queued_metadata))
    (case / "restore-submission-audit.json").write_text(json.dumps(records,indent=2)+"\n")


def plot(case, groups, lower, upper):
    os.environ.setdefault("MPLCONFIGDIR", str(case / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for ax, pid in zip(axes, sorted(groups)):
        g = groups[pid]
        for y,(label,category,color) in enumerate([
            ("Model forward stream", "forward_stream", "#2878b5"),
            ("All non-transfer compute", "compute", "#4eaa6b"),
            ("Cold KV restore", "H2D", "#e07b39"),
            ("Cold KV offload", "D2H", "#9e77b5")]):
            intervals = [((a-lower)/1e9,(b-a)/1e9) for a,b in g.get(category,[])]
            ax.broken_barh(intervals, (y-.35,.7), facecolors=color, rasterized=True)
        ax.set_yticks(range(4), ["Forward", "Compute", "Restore", "Offload"])
        ax.set_title(f"TP process {pid}", loc="left")
        ax.set_xlim(0,(upper-lower)/1e9)
        ax.grid(axis="x",alpha=.2)
    axes[-1].set_xlabel("Time within common TP capture window (seconds)")
    fig.tight_layout()
    fig.savefig(case / "transfer-compute-timeline.png", dpi=180)
    fig.savefig(case / "transfer-compute-timeline.pdf")
    plt.close(fig)


def plot_hardware(case, groups, series, lower, upper):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    edges = np.arange(lower, upper + 5_000_000, 5_000_000, dtype=np.int64)
    for ax, pid in zip(axes, sorted(groups)):
        restores = [((a-lower)/1e9,(b-a)/1e9) for a,b in groups[pid].get("H2D",[])]
        ax.broken_barh(restores,(0,100),facecolors="#e07b39",alpha=.15,rasterized=True)
        for (process, name), (times, values) in series.items():
            if process != pid:
                continue
            valid = np.isfinite(values)
            totals, _ = np.histogram(times[valid],bins=edges,weights=values[valid])
            counts, _ = np.histogram(times[valid],bins=edges)
            means = np.divide(totals,counts,out=np.full(len(counts),np.nan),where=counts>0)
            ax.plot(((edges[:-1]-lower)+(edges[1:]-lower))/2e9,means,label=name,linewidth=.8)
        ax.set_ylim(0,105)
        ax.set_ylabel("Counter value (%)")
        ax.set_title(f"TP process {pid}; 5 ms bins; orange = cold restore",loc="left")
        ax.legend(fontsize=7,loc="upper right")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Time within common TP capture window (seconds)")
    fig.tight_layout()
    fig.savefig(case / "restore-sm-counters.png",dpi=180)
    fig.savefig(case / "restore-sm-counters.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.case.resolve())
