"""Plot recorded verify lengths; compare only identical unprofiled workloads."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def read_case(path):
    launch = json.loads((path / "launch.json").read_text())
    if launch["state"] != "complete":
        raise ValueError(f"Incomplete case: {path}")
    if launch["phase"] != "scan" or launch["nsys_enabled"] or launch["selection_probe"] is not None:
        raise ValueError("Main comparison accepts only unprofiled scans without selection snapshots")
    k = launch["kwargs"]["spec_stride"]
    all_hist, hot_hist, cold_hist = [np.zeros(k + 1, dtype=np.int64) for _ in range(3)]
    requests = defaultdict(lambda: dict(rounds=0, accepted=0, drafted=0))
    context_hist = {edge: np.zeros(k + 1, dtype=np.int64) for edge in [2048, 4096, 8192, 16384, 32768]}
    with (path / "rounds.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            accepted, drafted = row["accepted_draft_tokens"], row["actual_draft_tokens"]
            if not 0 <= accepted <= drafted <= k:
                raise ValueError("Invalid verify counts")
            all_hist[accepted] += 1
            if row["cold_start"]:
                cold_hist[accepted] += 1
                continue
            hot_hist[accepted] += 1
            request = requests[row["request_id"]]
            request["rounds"] += 1
            request["accepted"] += accepted
            request["drafted"] += drafted
            edge = next((edge for edge in context_hist if row["kv_len"] <= edge), 32768)
            context_hist[edge][accepted] += 1
    means = [row["accepted"] / row["rounds"] for row in requests.values()]
    states = [json.loads(line) for line in (path / "resources.rank0.jsonl").read_text().splitlines()]
    weighted = defaultdict(float)
    seconds = 0.0
    for left, right in zip(states, states[1:]):
        duration = right["perf_counter"] - left["perf_counter"]
        if duration < 0:
            raise ValueError("Non-monotonic resource timestamps")
        seconds += duration
        for key in ["active_requests", "pending_requests", "offloaded_requests", "active_kv_ratio", "allocated_kv_ratio"]:
            weighted[key] += left[key] * duration
    observer = json.loads((path / "observer.rank0.json").read_text())
    return dict(dataset=launch["kwargs"]["dataset_name"], ratio=launch["kwargs"]["budget_ratio"],
                k=k, case=str(path), input_sha256=launch["token_ids_sha256"], metrics=launch["metrics"],
                histogram_all=all_hist.tolist(), histogram_cold=cold_hist.tolist(), histogram_hot=hot_hist.tolist(),
                histogram_hot_by_context_upper_bound={edge: counts.tolist() for edge, counts in context_hist.items()},
                hot_mean_accepted=float(np.dot(np.arange(k + 1), hot_hist) / hot_hist.sum()) if hot_hist.sum() else None,
                hot_acceptance_rate=sum(row["accepted"] for row in requests.values()) /
                                    sum(row["drafted"] for row in requests.values()) if requests else None,
                request_mean_accepted_percentiles=np.percentile(means, [10, 50, 90]).tolist() if means else None,
                resource_sample_coverage_seconds=seconds,
                resource_time_weighted_mean={key: value / seconds for key, value in weighted.items()} if seconds else {},
                kv_transfer_calls_rank0=observer["transfer_calls"], kv_transfer_bytes_rank0=observer["transfer_bytes"],
                requests_with_hot_rounds=len(requests), source_commit=launch["git_commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log"):
        parser.error("Keep results under SparseSpec/log")
    results = [read_case(path.resolve()) for path in args.cases]
    groups = defaultdict(list)
    for result in results:
        groups[result["dataset"]].append(result)
    for cases in groups.values():
        if len({case["input_sha256"] for case in cases}) != 1:
            raise ValueError("Unequal tokenized workloads; cannot compare ratios")
        if len({case["ratio"] for case in cases}) != len(cases):
            raise ValueError("Repeated ratios need an explicit repeat analysis")
    output.mkdir(parents=True, exist_ok=True)
    histogram_rows = []
    for dataset, cases in groups.items():
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for case in sorted(cases, key=lambda case: case["ratio"]):
            counts = np.asarray(case["histogram_hot"])
            lengths = np.arange(len(counts))
            label = f"r={case['ratio']:.2f} (n={counts.sum():,})"
            axes[0].plot(lengths, counts, marker="o", label=label)
            axes[1].plot(lengths, 100 * counts / max(1, counts.sum()), marker="o", label=label)
            for length, count in enumerate(counts):
                histogram_rows.append(dict(dataset=dataset, ratio=case["ratio"], accepted_length=length,
                                           hot_rounds=int(count), all_rounds=case["histogram_all"][length],
                                           cold_rounds=case["histogram_cold"][length]))
        for axis, ylabel in zip(axes, ["Verify rounds", "Share of verify rounds (%)"]):
            axis.set(xlabel="Accepted draft tokens (bonus excluded)", ylabel=ylabel,
                     xticks=np.arange(cases[0]["k"] + 1), title=dataset)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        figure.savefig(output / f"{dataset}-acceptance.png", dpi=180)
        figure.savefig(output / f"{dataset}-acceptance.pdf")
        plt.close(figure)
    with (output / "histograms.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histogram_rows[0]))
        writer.writeheader()
        writer.writerows(histogram_rows)
    report = dict(cases=results, notes=[
        "Primary histograms exclude cold-start rounds; all/cold counts are retained in CSV.",
        "Accepted counts exclude bonus and precede final EOS trimming.",
        "Verify rounds within a request are correlated; no independent-round confidence claim.",
        "Throughput uses actual saved output token counts and the engine generation window.",
        "Activity includes STALL; active KV excludes GPU pages retained by offloaded requests.",
        "Transfer bytes are KV payload per rank, not measured PCIe utilization or DMA latency.",
        "Sparse resource time averages use left-held samples and exclude the unsampled final interval."])
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), complete_cases=len(results))))


if __name__ == "__main__":
    main()
