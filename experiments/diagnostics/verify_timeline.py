"""Windowed host-side Verify lifecycle; no scheduler decisions or CUDA waits.

Host observation times and GPU event times are kept in separate clock domains.
Request waiting intervals may overlap each other and useful GPU computation.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time


class VerifyTimeline:
    def __init__(self, path, rank):
        self.file = Path(path).open("w", buffering=1024 * 1024)
        self.rank = rank
        self.active = False
        self.cycles = Counter()
        self.current = {}

    def emit(self, event, request_id=None, *, cycle=None, cpu_ns=None, **fields):
        row = dict(event=event, rank=self.rank,
                   cpu_ns=time.perf_counter_ns() if cpu_ns is None else cpu_ns, **fields)
        if request_id is not None:
            row.update(request_id=request_id, cycle=self.cycles[request_id] if cycle is None else cycle)
        self.file.write(json.dumps(row) + "\n")

    def start(self, cpu_ns, **fields):
        self.active = True
        self.emit("window_start", cpu_ns=cpu_ns, **fields)

    def observe(self, request_id, phase, *, carry_in=False, **fields):
        if not self.active or phase not in ("DRAFT", "VERIFY", "STALL"):
            return
        if request_id not in self.current:
            self.current[request_id] = dict(need=False, submitted=False, wait=None)
            self.emit("cycle_observed", request_id, phase=phase, **fields)
            # A Verify/STALL already present at the window boundary has an
            # unknown need time. Do not substitute the boundary as a true need.
            if phase == "STALL":
                self.current[request_id]["submitted"] = True
        state = self.current[request_id]
        if phase == "VERIFY" and not state["need"]:
            state["need"] = True
            self.emit("verify_need", request_id, need_time_known=not carry_in, **fields)
            self.wait(request_id, "await_admission", **fields)

    def wait(self, request_id, reason, **fields):
        state = self.current.get(request_id)
        if not self.active or state is None or not state["need"] or state["submitted"]:
            return
        # Save transition snapshots, rather than a row per request per poll.
        if state["wait"] != reason:
            state["wait"] = reason
            self.emit("wait_state", request_id, reason=reason, **fields)

    def note(self, event, request_id, **fields):
        if self.active:
            self.emit(event, request_id, **fields)

    def submit_verify(self, request_id, step):
        if not self.active:
            return
        state = self.current[request_id]
        if not state["need"] or state["submitted"]:
            raise RuntimeError("Verify submit without a unique observed need")
        state["submitted"] = True
        self.emit("verify_forward_submit", request_id, step=step)

    def result(self, request_id):
        if self.active:
            self.emit("verify_result_consumed", request_id)
            self.current.pop(request_id, None)
            self.cycles[request_id] += 1

    def stop(self, cpu_ns):
        self.emit("window_end", cpu_ns=cpu_ns)
        self.active = False
        self.file.flush()

    def close(self):
        self.file.close()


def _distribution(values):
    if not values:
        return dict(count=0)
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered) - 1) * p
        low = int(index)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=percentile(.9), maximum=max(values))


def analyze_verify_timeline(case, rank=0, *, plots=True):
    """Can run after window end while natural-EOS generation is still running."""
    case = Path(case)
    path = case / f"verify-timeline.rank{rank}.jsonl"
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [e for e in events if e["event"] == "window_start"]
    ends = [e for e in events if e["event"] == "window_end"]
    if not starts or not ends:
        return dict(status="window_not_finished", path=str(path))
    begin, end = starts[0]["cpu_ns"], ends[-1]["cpu_ns"]
    forwards = {}
    component_path = case / f"components.rank{rank}.jsonl"
    if component_path.exists():
        for line in component_path.read_text().splitlines():
            event = json.loads(line)
            if event["component"] == "forward" and "gpu_start_ms" in event:
                forwards[event["producer_step"]] = event
    grouped = defaultdict(list)
    for event in events:
        if "request_id" in event:
            grouped[(event["request_id"], event["cycle"])].append(event)
    rows, intervals = [], []
    totals = Counter()
    for (request_id, cycle), group in sorted(grouped.items()):
        need = next((e for e in group if e["event"] == "verify_need"), None)
        if need is None:
            continue  # Prefetch-only observation is not Verify waiting.
        submit = next((e for e in group if e["event"] == "verify_forward_submit"), None)
        finish = min(submit["cpu_ns"] if submit else end, end)
        onset = max(begin, need["cpu_ns"])
        changes = sorted([e for e in group if e["event"] == "wait_state"
                          and onset <= e["cpu_ns"] <= finish], key=lambda e:e["cpu_ns"])
        spans, cursor, reason = Counter(), onset, "await_admission"
        for event in changes + [dict(cpu_ns=finish, reason="end")]:
            duration = max(0, event["cpu_ns"] - cursor) / 1e6
            if duration:
                spans[reason] += duration
                intervals.append(dict(request_id=request_id, cycle=cycle, reason=reason,
                                      start_ns=cursor, end_ns=event["cpu_ns"]))
            cursor, reason = event["cpu_ns"], event["reason"]
        elapsed = max(0, finish - onset) / 1e6
        if abs(sum(spans.values()) - elapsed) > 1e-5:
            raise RuntimeError("Verify waiting states must partition the observed interval")
        totals.update(spans)
        def first(name):
            return next((e for e in group if e["event"] == name), None)
        milestones = {name:first(name) for name in (
            "restore_submit_begin", "restore_pages_allocated", "restore_submitted",
            "restore_ready_observed", "restore_page_table_committed", "restore_device_ready")}
        commit = milestones["restore_page_table_committed"]
        allocation = milestones["restore_pages_allocated"]
        forward = forwards.get(submit["step"]) if submit else None
        device_ready = milestones["restore_device_ready"]
        row = dict(request_id=request_id, cycle=cycle, need_cpu_ns=need["cpu_ns"],
                   need_time_known=need["need_time_known"],
                   right_censored=submit is None, observed_wait_ms=elapsed,
                   wait_ms=dict(spans), milestones=milestones,
                   verify_submit=submit,
                   verify_forward_gpu_event=forward,
                   device_ready_to_forward_event_ms=(max(0., forward["gpu_start_ms"] - device_ready["gpu_ready_ms"])
                        if forward and device_ready else None),
                   prefetch_committed_before_need=bool(commit and commit["cpu_ns"] < need["cpu_ns"]),
                   destination_allocated_before_need=bool(allocation and allocation["cpu_ns"] < need["cpu_ns"]))
        rows.append(row)
    complete = [r for r in rows if r["need_time_known"] and not r["right_censored"]]
    summary = dict(status="complete", rank=rank, window_seconds=(end - begin) / 1e9,
        observed_verify_cycles=len(rows), complete_uncensored_cycles=len(complete),
        left_censored_cycles=sum(not r["need_time_known"] for r in rows),
        right_censored_cycles=sum(r["right_censored"] for r in rows),
        host_need_to_submit_ms=_distribution([r["observed_wait_ms"] for r in complete]),
        window_request_wait_ms_by_state=dict(totals),
        complete_cycle_wait_ms_by_state={name:_distribution([r["wait_ms"].get(name, 0.) for r in complete])
                                         for name in sorted(totals)},
        prefetch_committed_before_need=sum(r["prefetch_committed_before_need"] for r in complete),
        complete_cycles_with_restore_commit_observed=sum(
            r["milestones"]["restore_page_table_committed"] is not None for r in complete),
        device_ready_to_forward_event_ms=_distribution([r["device_ready_to_forward_event_ms"]
            for r in complete if r["device_ready_to_forward_event_ms"] is not None]),
        notes=["Host milestones are observations at scheduler checkpoints, not exact GPU execution timestamps.",
               "State totals are request-milliseconds; overlapping request waits are not system elapsed time or GPU idle time.",
               "Wait reasons follow the current policy's gate order. Multiple constraints can coexist; snapshots retain flags, not causal proof.",
               "Destination allocation is observed at the actual allocator return, not inferred from nominal free capacity.",
               "Device-ready GPU event timestamps are separate from host completion observation and page-table commit.",
               "Carry-in need times and unsubmitted carry-out cycles are censored; full-wait quantiles exclude them.",
               "Compare CUDA kernel/hardware timelines to establish GPU idle gaps; forward event envelopes alone cannot measure SM utilization."])
    (case / f"verify-wait-cycles.rank{rank}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (case / f"verify-wait-summary.rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    if plots and intervals:
        _plot(case, rank, begin, end, totals, intervals, forwards)
    return summary


def _plot(case, rank, begin, end, totals, intervals, forwards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(totals, key=lambda k: -totals[k])
    colors = {name:plt.get_cmap("tab20")(i % 20) for i, name in enumerate(names)}
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw=dict(height_ratios=[1, 2, 1]))
    axes[0].barh(names[::-1], [totals[k] for k in names[::-1]], color=[colors[k] for k in names[::-1]])
    axes[0].set_xlabel("Summed request waiting (ms); NOT GPU idle time")
    # Show the most delayed requests; selection is explicitly stated.
    weights = Counter()
    for row in intervals:
        weights[row["request_id"]] += row["end_ns"] - row["start_ns"]
    selected = [key for key, _ in weights.most_common(24)]
    positions = {key:i for i, key in enumerate(selected)}
    for row in intervals:
        if row["request_id"] not in positions:
            continue
        axes[1].broken_barh([((row["start_ns"] - begin) / 1e9,
                              (row["end_ns"] - row["start_ns"]) / 1e9)],
                            (positions[row["request_id"]] - .35, .7), facecolors=colors[row["reason"]])
    axes[1].set_yticks(range(len(selected)), [str(key) for key in selected])
    axes[1].set_ylabel("Request ID (top 24 by observed wait)")
    axes[1].set_xlabel("Host seconds from window start")
    axes[1].set_xlim(0, (end - begin) / 1e9)
    transfer_path = case / f"transfers.rank{rank}.jsonl"
    transfers = [json.loads(line) for line in transfer_path.read_text().splitlines()
                 if json.loads(line)["event"] == "commit"] if transfer_path.exists() else []
    if forwards:
        origin = min(row["gpu_start_ms"] for row in forwards.values())
        tracks = [("Forward envelope", list(forwards.values()), "tab:green")]
        tracks += [(direction, [r for r in transfers if r["direction"] == direction], color)
                   for direction, color in (("restore", "tab:blue"), ("repack", "tab:orange"))]
        for y, (label, group, color) in enumerate(tracks):
            axes[2].broken_barh([((row["gpu_start_ms"] - origin) / 1000,
                                 (row["gpu_end_ms"] - row["gpu_start_ms"]) / 1000) for row in group],
                               (y - .3, .6), facecolors=color)
        axes[2].set_yticks(range(3), [t[0] for t in tracks])
        axes[2].set_xlabel("GPU event seconds from first sampled forward (separate clock origin)")
    else:
        axes[2].text(.05, .5, "No flushed GPU event intervals available", transform=axes[2].transAxes)
    axes[2].set_title("GPU stream envelopes: overlap is NOT proof of simultaneous useful SM work", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"{case.name}, rank {rank}: Verify wait states; host checkpoint observations")
    fig.tight_layout()
    fig.savefig(case / f"verify-waits.rank{rank}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for path in sorted(args.root.glob("b*/verify-timeline.rank*.jsonl")):
        rank = int(path.stem.split("rank")[-1])
        results[f"{path.parent.name}/rank{rank}"] = analyze_verify_timeline(path.parent, rank)
    print(json.dumps(results, indent=2))
