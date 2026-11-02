"""Audit the no-offload batch sweep and export latency/throughput figures."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def union_duration(intervals):
    if not intervals:
        return 0.
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.
    for first, last in intervals[1:]:
        if first > end:
            total += end - start
            start, end = first, last
        else:
            end = max(end, last)
    return total + end - start


def trace_summary(path):
    trace = json.loads(path.read_text())
    intervals, kernel_totals, copies, category_intervals = [], {}, {}, {}
    forward = [event for event in trace["traceEvents"]
               if event.get("name") == "batch_knee.full_forward" and event.get("ph") == "X"]
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category == "kernel":
            span = (event["ts"], event["ts"] + event["dur"])
            intervals.append(span)
            kernel_totals[event["name"]] = kernel_totals.get(event["name"], 0.) + event["dur"]
            name = event["name"].lower()
            if "nccl" in name or "cross_device_reduce" in name or "custom_all_reduce" in name:
                kind = "communication_and_possible_peer_wait"
            elif "persistentattentionscore" in name:
                kind = "selection_scores"
            elif "persistentkerneltemplate" in name:
                kind = "attention_core"
            elif "gemm" in name or "cutlass::kernel" in name:
                kind = "gemm"
            else:
                kind = "other_compute"
            category_intervals.setdefault(kind, []).append(span)
        elif "memcpy" in category:
            copies[event["name"]] = copies.get(event["name"], 0.) + event["dur"]
    elapsed = max((end for start, end in intervals), default=0.) - min(
        (start for start, end in intervals), default=0.)
    busy = union_duration(intervals)
    return dict(file=str(path), kernel_union_ms=busy / 1000,
                first_to_last_kernel_ms=elapsed / 1000,
                kernel_timeline_coverage=busy / elapsed if elapsed else None,
                annotated_forward_cpu_ms=forward[0]["dur"] / 1000 if forward else None,
                top_kernels_us=sorted(kernel_totals.items(), key=lambda item: -item[1])[:12],
                category_union_ms={key: union_duration(spans) / 1000
                                   for key, spans in category_intervals.items()},
                compute_kernel_union_ms_excluding_comm=union_duration([
                    span for key, spans in category_intervals.items()
                    if key != "communication_and_possible_peer_wait" for span in spans]) / 1000,
                cuda_memcpy_events_us=copies,
                interpretation="Instrumented timeline coverage only. Communication kernels may spin waiting for peers. Neither kernel presence nor duration measures SM compute utilization; category unions can overlap.")


def main(root):
    manifests = [json.loads(path.read_text()) for path in sorted(root.glob("manifest.rank*.json"))]
    if not manifests:
        raise RuntimeError("No manifest")
    world = manifests[0]["tp"]
    ranks = [json.loads((root / f"results.rank{rank}.json").read_text()) for rank in range(world)]
    for rank in range(world):
        assert (root / f"complete.rank{rank}.json").exists(), "Sweep is incomplete"
        assert manifests[rank]["native_max_query_rows"] == 9
        assert manifests[rank]["probability_rows"] == 10
        assert manifests[rank]["packed_verify_query_rows"] == 36
        assert manifests[rank]["graph"] is False and manifests[rank]["offload"] is False
        assert manifests[rank]["source_sha256"] == manifests[0]["source_sha256"]
    rows = ranks[0]
    for other in ranks[1:]:
        assert [(r["phase"], r["context_tokens_before_forward"], r["batch"], r["status"], r["reason"])
                for r in rows] == [
               (r["phase"], r["context_tokens_before_forward"], r["batch"], r["status"], r["reason"])
                for r in other]
    complete = [row for row in rows if row["status"] == "complete"]
    for row in complete:
        assert row["kernel_buffers"]["max_spec_stride"] == 9
        assert row["kernel_buffers"]["score_tile_q"] == 64
        assert row["kernel_buffers"]["uses_paged_scores"] is False
        assert row["kernel_buffers"]["draft_probability_shape"] == [row["batch"], 10, 151936]
        assert not any(row[key] for key in ["kv_d2h_bytes", "kv_h2d_bytes", "repack_calls", "restore_calls"])
    contexts = sorted({row["context_tokens_before_forward"] for row in rows})
    phases = [phase for phase in ["draft", "verify"] if any(r["phase"] == phase for r in rows)]
    groups = []
    for phase in phases:
        for context in contexts:
            points = sorted((r for r in complete if r["phase"] == phase
                             and r["context_tokens_before_forward"] == context), key=lambda r: r["batch"])
            if not points:
                continue
            best = max(r["query_tokens_per_second"] for r in points)
            near = {str(threshold): next(r["batch"] for r in points
                                        if r["query_tokens_per_second"] >= threshold * best)
                    for threshold in [.9, .95]}
            marginal = []
            for previous, current in zip(points, points[1:]):
                ratio = current["batch"] / previous["batch"]
                gain = current["query_tokens_per_second"] / previous["query_tokens_per_second"]
                marginal.append(dict(batch_from=previous["batch"], batch_to=current["batch"],
                                     throughput_gain=gain - 1,
                                     throughput_log_elasticity=math.log(gain) / math.log(ratio)))
            groups.append(dict(phase=phase, context=context, measured_batches=[r["batch"] for r in points],
                best_measured_query_tokens_per_second=best,
                near_best_observed_batch=near, marginal=marginal,
                skips=[dict(batch=r["batch"], reason=r["reason"]) for r in rows
                       if r["phase"] == phase and r["context_tokens_before_forward"] == context
                       and r["status"] == "skipped"],
                scope="Near-best among measured points, not an identified physical saturation point or global optimum."))
    traces = []
    for row in complete:
        profile = row.get("profile")
        if profile:
            result = trace_summary(Path(profile["trace"]))
            result.update(phase=row["phase"], context=row["context_tokens_before_forward"], batch=row["batch"],
                          cpu_annotation_ranges=[r for r in profile["ranges"] if r["cpu_total_us"] > 0])
            traces.append(result)
    result = dict(groups=groups, traces_rank0=traces, measured_points=len(complete),
                  skipped_points=len(rows) - len(complete), checks_passed=True,
                  source_sha256=manifests[0]["source_sha256"])
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["phase", "context_tokens_before_forward", "batch", "status", "reason",
               "total_query_tokens", "visible_kv_tokens_per_request", "median_forward_ms", "p90_forward_ms",
               "query_tokens_per_second", "peak_allocated_bytes"]
    with (root / "results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    for quantity, label, filename in [
            ("median_forward_ms", "Full forward latency (ms)", "forward-latency"),
            ("query_tokens_per_second", "Processed query tokens / second", "query-throughput")]:
        fig, axes = plt.subplots(len(phases), len(contexts), figsize=(15, 4.0 * len(phases)), squeeze=False)
        for i, phase in enumerate(phases):
            for j, context in enumerate(contexts):
                ax = axes[i, j]
                points = sorted((r for r in complete if r["phase"] == phase
                                 and r["context_tokens_before_forward"] == context), key=lambda r: r["batch"])
                if not points:
                    continue
                x, y = [r["batch"] for r in points], [r[quantity] for r in points]
                if quantity == "median_forward_ms":
                    ax.errorbar(x, y, yerr=[[r["median_forward_ms"]-r["p25_forward_ms"] for r in points],
                                           [r["p75_forward_ms"]-r["median_forward_ms"] for r in points]],
                                marker="o", capsize=3, color="#ba4a39")
                else:
                    ax.plot(x, y, "o-", color="#246ba5")
                    group = next(g for g in groups if g["phase"] == phase and g["context"] == context)
                    ax.axhline(.9 * group["best_measured_query_tokens_per_second"], ls=":", color="gray",
                               label="90% of observed maximum")
                    ax.legend(fontsize=7, loc="lower right")
                ticks = sorted({1, *[round(max(x) * i / 4) for i in range(1, 5)]})
                ax.set_xticks(ticks)
                ax.set_xlim(0, max(x) * 1.04)
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Actual requests in forward")
                ax.set_ylabel(label)
                ax.set_title(f"{phase.upper()} | prior context {context:,} | q/request={1 if phase=='draft' else 9}")
                ax.grid(alpha=.25)
        fig.suptitle("Qwen3-8B | TP2 | k=8 | r=0.05 | eager | all full KV on GPU\n"
                     "Fixed synthetic shapes; throughput counts processed queries, not accepted/generated tokens", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .92])
        fig.savefig(root / (filename + ".png"), dpi=180)
        fig.savefig(root / (filename + ".pdf"))
        plt.close(fig)
    print(json.dumps([dict(phase=g["phase"], context=g["context"], batches=g["measured_batches"],
                           near_best=g["near_best_observed_batch"],
                           best_query_tokens_per_second=g["best_measured_query_tokens_per_second"])
                      for g in groups], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
