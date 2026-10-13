"""CPU exploration of acceptance distributions using past-only runtime features.

No policy switching or throughput gains are inferred from fixed-policy logs.
Optional second stage waits for complete index diagnostics and joins each draft
snapshot to its subsequent verification before request-held-out evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from footprint_stats import summarize_selection

RATIOS = ["0.05", "0.10", "0.15"]
STATE = ["log_context", "effective_retention", "log_input", "generated_fraction",
         "log_round", "context_2_4K", "context_4_8K", "context_8_16K", "context_16K_plus"]
HISTORY = ["last_accept", "mean8", "mean32", "std32", "full_fraction8",
           "full_fraction32", "short_fraction8", "short_fraction32", "mean8_change"]
INDEX = ["token_union_fraction", "layer_union_fraction", "head_retention_fraction",
         "old_token_mean_position", "old_token_position_std", "old_token_first_quarter_fraction",
         "has_previous_snapshot", "old_union_jaccard_with_previous"]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_case(path):
    launch = json.loads((path / "launch.json").read_text())
    if launch["state"] != "complete" or launch["kwargs"]["spec_stride"] != 8:
        raise ValueError(f"Expected completed k=8 case: {path}")
    groups = defaultdict(list)
    digest = hashlib.sha256()
    with (path / "rounds.jsonl").open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row["cold_start"]:
                continue
            if row["actual_draft_tokens"] != 8:
                raise ValueError("This exploration requires a constant observed draft length")
            groups[row["request_id"]].append([row[k] for k in
                ["round", "step_id", "kv_len", "draft_kv_len", "accepted_draft_tokens"]])
    audit = json.loads((path / "request-audit.json").read_text())
    inputs = {r["request_id"]: r["input_tokens"] for r in audit}
    samples = {r["request_id"]: r["sample_id"] for r in audit}
    arrays = {rid: np.asarray(rows, dtype=np.int64) for rid, rows in groups.items()}
    for rows in arrays.values():
        if np.any(np.diff(rows[:, 0]) != 1) or np.any(np.diff(rows[:, 1]) <= 0):
            raise ValueError("Rounds must be contiguous and verification steps increasing")
    return arrays, inputs, samples, dict(path=str(path), rounds_sha256=digest.hexdigest(),
                                        token_ids_sha256=launch["token_ids_sha256"])


def feature(rows, index, input_len, context_override=None, retention_override=None):
    row = rows[index]
    length = row[2] if context_override is None else context_override
    retention = row[3] / row[2] if retention_override is None else retention_override
    state = [np.log1p(length), retention, np.log1p(input_len),
             (length - input_len) / 32768, np.log1p(row[0]),
             int(2048 <= length < 4096), int(4096 <= length < 8192),
             int(8192 <= length < 16384), int(length >= 16384)]
    past = rows[index - 32:index, 4] / 8
    history = [past[-1], past[-8:].mean(), past.mean(), past.std(),
               (past[-8:] == 1).mean(), (past == 1).mean(),
               (past[-8:] <= .25).mean(), (past <= .25).mean(),
               past[-8:].mean() - past[-16:-8].mean()]
    return state + history


def classes(values):
    return np.where(values <= 2, 0, np.where(values == 8, 2, 1))


def pair_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    return weights / weights.sum()


def fit_predict(x, target, groups, train, test, columns, regularization):
    weights = pair_weights(groups[train])
    selected = x[:, columns]
    mean = np.average(selected[train], axis=0, weights=weights)
    scale = np.sqrt(np.average((selected[train] - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-9] = 1
    design = np.column_stack([np.ones(len(x)), (selected - mean) / scale])
    a, y = design[train], target[train]
    prior = np.clip(np.average(y, axis=0, weights=weights), 1e-6, 1)
    initial = np.zeros((design.shape[1], 3))
    initial[0] = np.log(prior)

    def loss(flat):
        coef = flat.reshape(initial.shape)
        logits = a @ coef
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        value = -(weights[:, None] * y * logp).sum() + regularization * (coef[1:] ** 2).sum() / 2
        gradient = a.T @ (weights[:, None] * (np.exp(logp) - y))
        gradient[1:] += regularization * coef[1:]
        return value, gradient.ravel()

    fit = minimize(loss, initial.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=300, ftol=1e-10))
    if not fit.success:
        raise RuntimeError(f"Probability model did not converge: {fit.message}")
    logits = design[test] @ fit.x.reshape(initial.shape)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def evaluate(x, y, request_ids, groups, models, regularization):
    ids = np.unique(request_ids)
    if len(ids) < 15:
        return dict(insufficient_requests=len(ids), note="Descriptive results only; no reliable five-fold estimate")
    shuffled = np.random.default_rng(330).permutation(ids)
    fold_map = {int(rid): fold for fold, values in enumerate(np.array_split(shuffled, 5)) for rid in values}
    folds = np.asarray([fold_map[int(rid)] for rid in request_ids])
    predictions = {name: np.zeros_like(y) for name in models}
    for fold in range(5):
        test = folds == fold
        for name, columns in models.items():
            predictions[name][test] = fit_predict(x, y, groups, ~test, test, columns, regularization)
    metrics, per_request = {}, {}
    for name, p in predictions.items():
        losses = np.column_stack([-(y * np.log(np.clip(p, 1e-12, 1))).sum(axis=1),
                                  (p[:, 0] - y[:, 0]) ** 2, (p[:, 2] - y[:, 2]) ** 2])
        per_group = {g: losses[groups == g].mean(axis=0) for g in np.unique(groups)}
        # Equal request weight, and equal ratio weight within each request.
        per_request[name] = np.asarray([
            np.mean([v for g, v in per_group.items() if int(g.split(":")[-1]) == rid], axis=0)
            for rid in ids])
        metrics[name] = dict(zip(["log_loss", "short_brier", "full_brier"],
                                per_request[name].mean(axis=0).tolist()))
        weight = pair_weights(groups)
        order = np.argsort(-p[:, 0])
        chosen = order[np.cumsum(weight[order]) <= .20]
        metrics[name]["actual_short_fraction_in_top20pct_predicted_risk"] = float(
            np.average(y[chosen, 0], weights=weight[chosen]))
    reference, candidate = list(models)[-2:]
    delta = per_request[reference] - per_request[candidate]
    rng = np.random.default_rng(331)
    bootstrap = delta[rng.integers(0, len(ids), size=(2000, len(ids)))].mean(axis=1)
    return dict(anchors=len(x), requests=len(ids), folds=fold_map, metrics=metrics,
                comparison=dict(reference=reference, candidate=candidate,
                                reduction=dict(zip(["log_loss", "short_brier", "full_brier"], delta.mean(axis=0).tolist())),
                                ci95=dict(zip(["log_loss", "short_brier", "full_brier"],
                                              np.quantile(bootstrap, [.025, .975], axis=0).T.tolist()))))


def phase1(root, output):
    result = dict(protocol=dict(k=8, warmup_rounds=32, anchor_stride=8, folds=5,
                               features=STATE + HISTORY, regularization=.001,
                               weighting="equal requests", bootstrap_unit="request",
                               targets=["accept_0_2", "accept_3_7", "accept_8"],
                               limitations=["Observed fixed-r policies only; no r-switch or throughput counterfactual.",
                                            "Request-held-out histories use only already verified rounds.",
                                            "Acceptance at 8 is censored; larger-k acceptance is unknown."]), cases={})
    for ratio in RATIOS:
        arrays, inputs, _, source = load_case(root / f"scan-livecodebench-r{ratio}")
        x, targets, ids = [], [], []
        distributions = defaultdict(lambda: np.zeros(9, dtype=np.int64))
        for rid, rows in arrays.items():
            for t in range(32, len(rows), 8):
                x.append(feature(rows, t, inputs[rid])); targets.append(classes(rows[t, 4])); ids.append(rid)
                if 4096 <= rows[t, 2] < 8192:
                    short_count = np.sum(rows[t - 8:t, 4] <= 2)
                    regime = "no_short_in_past8" if short_count == 0 else "at_least2_short_in_past8" if short_count >= 2 else "one_short_in_past8"
                    distributions[regime][rows[t, 4]] += 1
        x, ids, target = np.asarray(x), np.asarray(ids), np.eye(3)[targets]
        groups = np.asarray([f"{ratio}:{rid}" for rid in ids])
        models = dict(prior=[], state=list(range(len(STATE))),
                      state_history=list(range(len(STATE) + len(HISTORY))))
        scores = evaluate(x, target, ids, groups, models, .001)
        scores["source"] = source
        scores["descriptive_4_8K_history_conditioned_histograms"] = {
            k: dict(anchors=int(v.sum()), counts=v.tolist(),
                    next_short_fraction=float(v[:3].sum() / v.sum()),
                    next_full_fraction=float(v[8] / v.sum())) for k, v in distributions.items()}
        result["cases"][ratio] = scores
        save(output / "history-results.json", result)
        print(json.dumps(dict(ratio=ratio, **scores), ensure_ascii=False), flush=True)
    plot_history(result, output)
    return result


def plot_history(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, ratio in zip(axes, RATIOS):
        distributions = result["cases"][ratio]["descriptive_4_8K_history_conditioned_histograms"]
        for offset, key, label in [(-.2, "no_short_in_past8", "Past 8: no acceptance <=2"),
                                    (.2, "at_least2_short_in_past8", "Past 8: at least two <=2")]:
            values = np.asarray(distributions[key]["counts"])
            ax.bar(np.arange(9) + offset, values / values.sum(), width=.4, label=label)
        ax.set(xlabel="Next-round accepted draft tokens", title=f"r={ratio}", xticks=np.arange(9))
    axes[0].set_ylabel("Fraction of sampled rounds")
    axes[0].legend(fontsize=8)
    fig.suptitle("LCB, k=8, context 4-8K; descriptive distributions conditioned on past history")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"history-conditioned-distributions.{suffix}", dpi=180)
    plt.close(fig)


def selection_ready(root):
    for ratio in RATIOS:
        path = root / f"selection-livecodebench-r{ratio}" / "launch.json"
        if not path.exists() or json.loads(path.read_text())["state"] != "complete":
            return False
    return True


def phase2(root, output):
    x, targets, ids, groups, details = [], [], [], [], []
    unmatched, invalid, duplicates, sources = 0, 0, 0, []
    seen_targets = set()
    for ratio in RATIOS:
        path = root / f"selection-livecodebench-r{ratio}"
        arrays, inputs, samples, source = load_case(path); sources.append(source)
        by_rank = []
        for rank in [0, 1]:
            records = [json.loads(line) for line in (path / f"selection.rank{rank}" / "samples.jsonl").open()]
            by_rank.append({(v["request_id"], v["step_id"]): v for v in records})
        common = sorted(set(by_rank[0]) & set(by_rank[1]), key=lambda pair: pair[1])
        unmatched += len(set(by_rank[0]) ^ set(by_rank[1]))
        previous = {}
        for rid, step in common:
            records = [rank[(rid, step)] for rank in by_rank]
            a, b = records
            if (a["context_tokens"], a["draft_kv_tokens"], a["selected_prefix_tokens"]) != (
                    b["context_tokens"], b["draft_kv_tokens"], b["selected_prefix_tokens"]):
                raise ValueError("TP snapshot metadata differs")
            rows = arrays.get(rid)
            t = int(np.searchsorted(rows[:, 1], step)) if rows is not None else 0
            if rows is None or t < 32 or t >= len(rows) or not 0 <= a["context_tokens"] - rows[t, 2] <= 9:
                invalid += 1; continue
            target_key = (ratio, samples[rid], int(rows[t, 0]))
            if target_key in seen_targets:
                duplicates += 1; continue
            seen_targets.add(target_key)
            planes = []
            for rank, record in enumerate(records):
                with np.load(path / f"selection.rank{rank}" / record["indices_file"]) as data:
                    planes.append(data["logical_indices"])
            logical = np.concatenate(planes, axis=1)
            summary = summarize_selection(logical, a["context_tokens"], a["head_dim"], a["element_bytes"])
            old = logical[..., :a["selected_prefix_tokens"]]
            union = np.unique(old); prev = previous.get(rid)
            shared = len(np.intersect1d(prev, union)) if prev is not None else 0
            jaccard = shared / len(np.union1d(prev, union)) if prev is not None else 0
            previous[rid] = union
            positions = old / a["context_tokens"]
            extra = [1 - summary["releasable_fraction"][key] for key in ["token", "layer", "head"]]
            extra += [float(positions.mean()), float(positions.std()), float((positions < .25).mean()),
                      int(prev is not None), jaccard]
            features = feature(rows, t, inputs[rid], a["context_tokens"],
                               a["draft_kv_tokens"] / a["context_tokens"])
            x.append(features + [float(ratio)] + extra)
            targets.append(classes(rows[t, 4])); ids.append(samples[rid]); groups.append(f"{ratio}:{samples[rid]}")
            details.append(dict(ratio=ratio, sample_id=samples[rid], request_id=rid, snapshot_step=step,
                                verify_step=int(rows[t, 1]), accepted=int(rows[t, 4]), index_features=extra,
                                context=int(a["context_tokens"]), target_round=int(rows[t, 0])))
    names = STATE + HISTORY + ["ratio"] + INDEX
    base = len(STATE) + len(HISTORY) + 1
    scores = evaluate(np.asarray(x), np.eye(3)[targets], np.asarray(ids), np.asarray(groups),
                      dict(state_history=list(range(base)),
                           state_history_indices=list(range(len(names)))), .01) if x else dict(insufficient_anchors=0)
    scores.update(features=names, regularization=.01, sources=sources, records=details,
                  unmatched_tp_snapshots=unmatched, rejected_alignment_or_warmup=invalid,
                  duplicate_verify_targets_skipped=duplicates,
                  limitation="Small diagnostic sample, not a policy-switching or throughput experiment.")
    save(output / "selection-results.json", scores)
    print(json.dumps({k: v for k, v in scores.items() if k != "records"}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--wait-for-selection", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.phase2_only:
        phase1(args.root, args.output)
    while args.wait_for_selection and not selection_ready(args.root):
        status = json.loads((args.root / "pipeline-status.json").read_text())
        save(args.output / "selection-wait.json", dict(pid=os.getpid(), wall_time=time.time(),
                                                       state="waiting", pipeline_state=status["state"],
                                                       active=list(status["active"]), failed=status["failed"]))
        if status["state"] == "failed":
            raise RuntimeError("Experiment queue failed; inspect it before continuing selection analysis")
        time.sleep(300)
    if selection_ready(args.root):
        try:
            phase2(args.root, args.output)
        except Exception as error:
            save(args.output / "selection-wait.json", dict(pid=os.getpid(), wall_time=time.time(),
                                                           state="failed", error=repr(error)))
            raise
        save(args.output / "selection-wait.json", dict(pid=os.getpid(), wall_time=time.time(), state="complete"))


if __name__ == "__main__":
    main()
