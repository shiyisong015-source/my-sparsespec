"""Run the agreed grid on validated fixed requests, with per-case resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330"
REQUEST_FILE = OUTPUT_ROOT / "requests.jsonl"
OUT = OUTPUT_ROOT / "Qwen3-8B"
KS = (4, 8, 10, 12, 14)
RATIOS = (0.05, 0.10, 0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batch-size", type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.max_batch_size <= 100
    output_root = OUTPUT_ROOT if args.max_batch_size == 1 else OUTPUT_ROOT / f"batch{args.max_batch_size}"
    OUT = output_root / "Qwen3-8B"
    OUT.mkdir(parents=True, exist_ok=True)
    sample_sha = hashlib.sha256(REQUEST_FILE.read_bytes()).hexdigest()
    manifest = json.loads((OUTPUT_ROOT / "sample_manifest.json").read_text())
    assert sample_sha == manifest["sha256"]
    assert manifest["sample_size"] == 100
    reports = []
    for ratio in RATIOS:
        for k in KS:
            directory = OUT / f"k{k}_r{ratio:g}"
            report_path = directory / "case.json"
            if not report_path.exists():
                print(f"Running k={k}, r={ratio}", flush=True)
                for attempt in range(2):
                  with (OUT / f"k{k}_r{ratio:g}_attempt{attempt+1}.log").open("w") as log:
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/run_acceptance_case.py"),
                         "--spec-stride", str(k), "--budget-ratio", str(ratio),
                         "--request-file", str(REQUEST_FILE), "--output-root", str(output_root),
                         "--max-batch-size", str(args.max_batch_size)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    )
                  if result.returncode == 0:
                    break
                  print(f"Failed k={k}, r={ratio}, attempt={attempt+1}; logs preserved", flush=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Case k={k}, r={ratio} failed twice; rerun to resume completed cases")
            report = json.loads(report_path.read_text())
            assert report.get("max_batch_size", 1) == args.max_batch_size
            assert len(report["outputs"]) == 100
            assert report["model"] == "/data/models/Qwen3-8B"
            assert report["spec_stride"] == k and report["budget_ratio"] == ratio
            assert report["desired_output_tokens"] == 1024
            assert report["enable_thinking"] is False
            assert Path(report["dataset"]).resolve() == REQUEST_FILE.resolve()
            assert report["sample_sha256"] == sample_sha
            assert len(report["inputs"]) == 100
            counts = np.zeros((100, 2), dtype=np.int64)
            with (directory / "rounds.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row["cold_start"]:
                        counts[row["request_id"]] += (row["accepted_draft_tokens"], row["actual_draft_tokens"])
            assert counts[:, 1].sum() > 0
            rng = np.random.default_rng(330)
            sums = counts[rng.integers(0, 100, size=(2000, 100))].sum(axis=1)
            rates = sums[sums[:, 1] > 0, 0] / sums[sums[:, 1] > 0, 1]
            report["acceptance_ci95"] = np.quantile(rates, [0.025, 0.975]).tolist()
            report["requests_with_post_cold_rounds"] = int((counts[:, 1] > 0).sum())
            (directory / "per_request_counts.json").write_text(json.dumps(counts.tolist()))
            reports.append(report)
            temporary = OUT / "grid.json.tmp"
            temporary.write_text(json.dumps(reports, indent=2))
            temporary.replace(OUT / "grid.json")
            print(f"Done k={k}, r={ratio}: {report['post_cold_acceptance_rate']:.4f}", flush=True)
    (OUT / "grid.json").write_text(json.dumps(reports, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ratio in RATIOS:
        rows = sorted((x for x in reports if x["budget_ratio"] == ratio), key=lambda x: x["spec_stride"])
        xs = [x["spec_stride"] for x in rows]
        ax.plot(xs, [100*x["post_cold_acceptance_rate"] for x in rows],
                marker="o", label=f"KV retention ratio={ratio:g}")
        ax.fill_between(xs, [100*x["acceptance_ci95"][0] for x in rows],
                        [100*x["acceptance_ci95"][1] for x in rows], alpha=0.15)
    ax.set(xlabel="Draft length", ylabel="Accepted draft tokens (%)",
           title=f"Qwen3-8B: 100 LiveCodeBench requests, max batch={args.max_batch_size}", xticks=KS, ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "acceptance_pilot.png", dpi=200)
    fig.savefig(OUT / "acceptance_pilot.pdf")


if __name__ == "__main__":
    main()
