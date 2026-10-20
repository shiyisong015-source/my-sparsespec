"""Join real u, p/q and observed first rejection without inventing new draws."""
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(output):
    output = Path(output)
    payload = json.loads((output / "worker-payload.json").read_text())
    k = payload["kwargs"]["spec_stride"]
    results = []
    for rank in range(payload["kwargs"]["tp_ranks"]):
        suffix = "" if rank == 0 else f".rank{rank}"
        observed = rows(output / f"rounds{suffix}.jsonl")
        key = lambda r: (r["request_id"], r["round"])
        target = {key(r): r for r in observed}
        umeta = rows(output / f"uniform-signals.rank{rank}.jsonl")
        pmeta = rows(output / f"probability-signals.rank{rank}.jsonl")
        for name, meta in [("uniform", umeta), ("probability", pmeta)]:
            if (len(meta) != len(target) or {key(r) for r in meta} != set(target)
                    or [r["binary_row"] for r in meta] != list(range(len(meta)))):
                raise RuntimeError(f"Rank {rank}: {name}/acceptance record join failed")
        if (output / f"uniform-signals.rank{rank}.f64").stat().st_size != len(umeta)*k*8:
            raise RuntimeError("Uniform binary has incorrect size")
        u = np.memmap(output / f"uniform-signals.rank{rank}.f64", dtype="<f8", mode="r",
                      shape=(len(umeta), k))
        p = np.memmap(output / f"probability-signals.rank{rank}.f32", dtype="<f4", mode="r",
                      shape=(len(pmeta), 3*k+2))
        index = {key(r): r["binary_row"] for r in pmeta}
        boundary, mismatches = [], []
        for row in umeta:
            identity = key(row)
            expected = target[identity]
            n = row["actual_draft_tokens"]
            if n != expected["actual_draft_tokens"]:
                raise RuntimeError("Uniform trace has incorrect draft count")
            uniform = u[row["binary_row"], :n]
            probability = p[index[identity]]
            if not np.isfinite(uniform).all() or np.any((uniform < 0) | (uniform >= 1)):
                raise RuntimeError("Uniform outside [0,1)")
            if not np.isnan(u[row["binary_row"], n:]).all():
                raise RuntimeError("Uniform padding must be NaN")
            qtoken, ptoken = probability[k:k+n], probability[2*k:2*k+n]
            ratio = np.divide(ptoken, qtoken, out=np.zeros_like(ptoken), where=qtoken > 0)
            failed = np.flatnonzero((qtoken <= 0) | (uniform > ratio.astype(np.float64)))
            accepted = int(failed[0]) if len(failed) else n
            if accepted != expected["accepted_draft_tokens"]:
                pos = min(accepted, expected["accepted_draft_tokens"])
                entry = dict(request_id=identity[0], round=identity[1], predicted=accepted,
                             observed=expected["accepted_draft_tokens"], position=pos,
                             margin=float(ratio[pos]-uniform[pos]))
                # CPU and Triton FP32 division may differ at a comparison boundary.
                (boundary if abs(entry["margin"]) <= 1e-6 else mismatches).append(entry)
        summary = json.loads((output / f"uniform-summary.rank{rank}.json").read_text())
        if not summary["successful"] or summary["records"] != len(umeta):
            raise RuntimeError("Uniform observer did not close successfully")
        results.append(dict(rank=rank, rounds=len(umeta), unexplained_mismatches=mismatches,
                            near_comparison_boundary=boundary))
    rank_acceptance_identical = (output / "rounds.jsonl").read_bytes() == (
        output / "rounds.rank1.jsonl").read_bytes()
    uniforms_identical = (output / "uniform-signals.rank0.f64").read_bytes() == (
        output / "uniform-signals.rank1.f64").read_bytes()
    result = dict(successful=rank_acceptance_identical and not any(
        r["unexplained_mismatches"] for r in results), ranks=results,
        rank_acceptance_identical=rank_acceptance_identical,
        rank_uniforms_identical=uniforms_identical, k=k,
        notes="Actual native u, collected during verify. Predraw scheduling is not implemented.")
    (output / "uniform-validation.json").write_text(json.dumps(result, indent=2)+"\n")
    if not result["successful"]:
        raise RuntimeError("Actual u/p/q do not explain observed rejection; see uniform-validation.json")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(validate(parser.parse_args().output), indent=2))
