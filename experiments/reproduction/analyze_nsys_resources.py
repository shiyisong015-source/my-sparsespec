"""Measure interval unions from Nsight SQLite within aligned NVTX windows."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess

REPO = Path(__file__).resolve().parents[2]


def merge(intervals, lower, upper):
    result = []
    for start, end in sorted((max(start, lower), min(end, upper)) for start, end in intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def overlap(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(database):
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"} <= tables:
        raise ValueError("Missing CUDA/NVTX trace tables")
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges, transfers = defaultdict(list), defaultdict(list)
    for row in db.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        text = row["text"] or names.get(row["textId"], "")
        if text.startswith("resource.capture."):
            ranges[text].append((row["start"], row["end"], row["globalTid"] >> 24))
        elif text.startswith("resource.kv_"):
            transfers[row["globalTid"]].append((row["start"], row["end"], text))
    for rows in transfers.values():
        rows.sort()
    transfer_starts = {tid: [row[0] for row in rows] for tid, rows in transfers.items()}
    api_transfer = {}
    blocking = []
    for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        name = names.get(row["nameId"], "")
        if "Synchronize" in name:
            blocking.append((row["globalTid"], row["start"], row["end"], name))
        tid = row["globalTid"]
        if tid not in transfers:
            continue
        idx = bisect_right(transfer_starts[tid], row["start"]) - 1
        if idx >= 0 and transfers[tid][idx][1] >= row["start"]:
            api_transfer[(tid >> 24, row["correlationId"])] = transfers[tid][idx][2]
    kernels = list(db.execute("SELECT start,end,deviceId,globalPid,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    copies = list(db.execute("SELECT start,end,deviceId,globalPid,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    process_ids = {row["globalPid"] >> 24 for row in kernels}
    if len(process_ids) != 2:
        raise ValueError(f"Expected both TP2 workers, observed {len(process_ids)} CUDA processes")
    windows = []
    for name, rows in ranges.items():
        pids = {row[2] for row in rows}
        if pids != process_ids:
            raise ValueError(f"Capture {name} does not contain aligned ranges from both TP ranks")
        lower, upper = max(row[0] for row in rows), min(row[1] for row in rows)
        if upper <= lower:
            raise ValueError("No common TP capture interval")
        seconds = (upper - lower) / 1e9
        devices = []
        for device in sorted({row["deviceId"] for row in kernels}):
            rows_gpu = [row for row in kernels if row["deviceId"] == device]
            kernel_intervals = merge([(row["start"], row["end"]) for row in rows_gpu], lower, upper)
            kv_prepare = merge([(row["start"], row["end"]) for row in rows_gpu
                                if (row["globalPid"] >> 24, row["correlationId"]) in api_transfer], lower, upper)
            kv_copies = [row for row in copies if row["deviceId"] == device and row["copyKind"] in {1, 2}
                         and (row["globalPid"] >> 24, row["correlationId"]) in api_transfer
                         and row["end"] > lower and row["start"] < upper]
            dma_intervals = merge([(row["start"], row["end"]) for row in kv_copies], lower, upper)
            shared = overlap(kernel_intervals, dma_intervals)
            devices.append(dict(
                trace_device_id=device, window_seconds=seconds,
                kernel_busy_seconds=duration(kernel_intervals) / 1e9,
                kernel_busy_fraction=duration(kernel_intervals) / (upper - lower),
                kv_prepare_kernel_seconds=duration(kv_prepare) / 1e9,
                kv_dma_seconds=duration(dma_intervals) / 1e9,
                kv_dma_fraction=duration(dma_intervals) / (upper - lower),
                kernel_kv_dma_overlap_seconds=shared / 1e9,
                kv_dma_without_kernels_seconds=(duration(dma_intervals) - shared) / 1e9,
                kv_copy_events=len(kv_copies),
                kv_copy_bytes_of_overlapping_events=sum(row["bytes"] for row in kv_copies)))
        waits = defaultdict(list)
        for tid, start, end, label in blocking:
            waits[tid].append((start, end))
        windows.append(dict(name=name, start_ns=lower, end_ns=upper, devices=devices,
                            cpu_cuda_synchronize_seconds_by_thread={str(tid): duration(merge(values, lower, upper)) / 1e9
                                                                   for tid, values in waits.items()}))
    if not windows:
        raise ValueError("No complete capture window")
    db.close()
    return dict(database=str(database), windows=windows, cuda_processes=sorted(process_ids), notes=[
        "Kernel busy time includes inference, sampling, communication and KV preparation; it is not SM utilization.",
        "KV DMA attribution requires a CUDA API launch inside the resource.kv_* NVTX range with matching process/correlation ID.",
        "Other host/device metadata and peer transfers are not counted as KV DMA.",
        "Reported intervals are clipped to the common TP NVTX window, excluding profiler start/stop overhead.",
        "DMA bytes count whole overlapping copy events; boundary-crossing copies are not prorated.",
        "DMA without kernels is not automatically a critical-path delay.",
        "SM/Tensor/HBM hardware counters are unavailable due to current driver permissions."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--nsys", required=True, type=Path)
    args = parser.parse_args()
    case = args.case.resolve()
    if not case.is_relative_to(REPO / "log"):
        parser.error("Use a case under SparseSpec/log")
    reports = sorted(case.glob("*.nsys-rep"))
    if not reports:
        raise ValueError("No Nsight reports; check missing capture windows")
    results = []
    for report in reports:
        database = report.with_suffix(".sqlite")
        if not database.exists():
            with report.with_suffix(".export.log").open("w") as log:
                subprocess.run([str(args.nsys.resolve()), "export", "--type=sqlite", "--output=" + str(database),
                                str(report)], check=True, stdout=log, stderr=subprocess.STDOUT)
        results.append(analyze(database))
    observer = json.loads((case / "observer.rank0.json").read_text())
    result = dict(reports=results, observer=observer, profiler_throughput_is_diagnostic_only=True)
    (case / "nsys-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(case=str(case), reports=len(results))))


if __name__ == "__main__":
    main()
