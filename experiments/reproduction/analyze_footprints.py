"""Combine simultaneous TP snapshots; report per-request KV capacity bounds."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reproduction.footprint_stats import summarize_selection

REPO = Path(__file__).resolve().parents[2]


def read_case(case):
    launch = json.loads((case / "launch.json").read_text())
    if launch["state"] != "complete" or launch["selection_probe"] is None or launch["nsys_enabled"]:
        raise ValueError("Use a completed standalone selection diagnostic")
    ranks = []
    for rank in [0, 1]:
        directory = case / f"selection.rank{rank}"
        report = json.loads((directory / "summary.json").read_text())
        if not report["successful"]:
            raise ValueError("Selection probe failed")
        rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        ranks.append({(row["request_id"], row["step_id"]): row for row in rows})
    common = ranks[0].keys() & ranks[1].keys()
    records = []
    for key in sorted(common):
        rows = [rank[key] for rank in ranks]
        if any(rows[0][field] != rows[1][field] for field in
               ["context_tokens", "head_dim", "element_bytes", "budget_ratio", "draft_kv_tokens"]):
            raise ValueError("TP snapshot metadata disagree")
        selections = []
        for rank, row in enumerate(rows):
            with np.load(case / f"selection.rank{rank}" / row["indices_file"]) as data:
                logical = data["logical_indices"]
                if not np.array_equal(data["physical_to_logical_page_order"][logical], data["physical_indices"]):
                    raise ValueError("Physical/logical mapping mismatch")
                selections.append(logical.copy())
        indices = np.concatenate(selections, axis=1)
        stats = summarize_selection(indices, rows[0]["context_tokens"], rows[0]["head_dim"], rows[0]["element_bytes"])
        stats.update(case=str(case), request_id=key[0], step=key[1], ratio=rows[0]["budget_ratio"],
                     actual_draft_kv_tokens=rows[0]["draft_kv_tokens"], tp_ranks=2)
        records.append(stats)
    if not records:
        raise ValueError("No matching instantaneous TP snapshots")
    return records, dict(case=str(case), paired_samples=len(common),
                         unpaired_rank0=len(ranks[0]) - len(common), unpaired_rank1=len(ranks[1]) - len(common))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "log"):
        parser.error("Keep figures under SparseSpec/log")
    records, pairing = [], []
    for case in args.cases:
        rows, report = read_case(case.resolve())
        records.extend(rows)
        pairing.append(report)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, granularity in zip(axes, ["token", "layer", "head"]):
        for ratio in sorted({row["ratio"] for row in records}):
            rows = [row for row in records if row["ratio"] == ratio]
            axis.scatter([row["context_tokens"] for row in rows],
                         [100 * row["releasable_fraction"][granularity] for row in rows],
                         s=18, alpha=.65, label=f"r={ratio:.2f}")
        axis.set(xlabel="Context tokens", ylabel="Ideal releasable KV (%)", title=granularity, ylim=(0, 100))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.savefig(output / "tp2-footprint.png", dpi=180)
    figure.savefig(output / "tp2-footprint.pdf")
    plt.close(figure)
    flat = [dict(case=row["case"], request_id=row["request_id"], step=row["step"], ratio=row["ratio"],
                 context_tokens=row["context_tokens"], full_kv_bytes=row["full_kv_bytes"],
                 **{name + "_releasable_fraction": value for name, value in row["releasable_fraction"].items()})
            for row in records]
    with (output / "footprints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = dict(samples=records, pairing=pairing, notes=[
        "Each point combines the same request and step across both TP ranks.",
        "Snapshots across different steps/requests are not added into a batch release estimate.",
        "These are ideal KV-data bounds, excluding packing, temporary buffers and recovery costs.",
        "Throughput of the snapshot diagnostic is not used for the main ratio comparison."])
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), paired_samples=len(records))))


if __name__ == "__main__":
    main()
