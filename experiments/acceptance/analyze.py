"""k63 acceptance distributions and request-held-out upper-length signals."""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".repro/matplotlib"))
import numpy as np

K = 63
RATIOS = [.05, .10, .15, .20, .25]
CONTEXT_LABELS = ["<2K", "2-4K", "4-8K", "8-16K", "16-32K"]
PHASE_START = np.array([0, 1, 4, 8, 16, 32, 48])
PHASE_END = np.array([1, 4, 8, 16, 32, 48, 63])
PHASE = np.repeat(np.arange(7), PHASE_END - PHASE_START)


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_jsonl(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def key(row):
    return row["request_id"], row["round"]


def validate_case(output, expected_k=K):
    payload = json.loads((output / "worker-payload.json").read_text())
    rows = read_jsonl(output / "rounds.jsonl")
    with (output / "rounds.jsonl").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != hashlib.sha256((output / "rounds.rank1.jsonl").read_bytes()).hexdigest():
        raise RuntimeError("Rank acceptance traces differ")
    audit = json.loads((output / "request-audit.json").read_text())
    n = len(payload["requests"])
    if len(audit) != n or {r["request_id"] for r in rows} != set(range(n)):
        raise RuntimeError("Incomplete request generation")
    normal = [r for r in rows if not r["cold_start"]]
    if not normal or any(r["actual_draft_tokens"] != expected_k or
                         not 0 <= r["accepted_draft_tokens"] <= expected_k for r in normal):
        raise RuntimeError("Fixed-k trace contains an unexpected draft length")
    probability_keys, selection_keys = [], []
    for rank in [0, 1]:
        summary = json.loads((output / f"signal-summary.rank{rank}.json").read_text())
        if not summary["successful"] or summary["k"] != K:
            raise RuntimeError("Signal observer did not complete")
        for kind in ["probability", "selection"]:
            meta = read_jsonl(output / f"{kind}-signals.rank{rank}.jsonl")
            if len(meta) != summary["counts"][kind]:
                raise RuntimeError("Signal record count differs from completion summary")
            shape = summary[f"{kind}_record_shape"]
            size = (output / f"{kind}-signals.rank{rank}.f32").stat().st_size
            if size != len(meta) * int(np.prod(shape)) * 4:
                raise RuntimeError("Signal binary record has incomplete bytes")
            if [r["binary_row"] for r in meta] != list(range(len(meta))):
                raise RuntimeError("Signal binary row alignment is not contiguous")
            keys = {key(r) for r in meta}
            expected = {key(r) for r in rows} if kind == "probability" else {key(r) for r in normal}
            if keys != expected or len(keys) != len(meta):
                raise RuntimeError(f"{kind} signal/verify join failed on rank {rank}")
            (probability_keys if kind == "probability" else selection_keys).append(keys)
        selections = {key(r): r for r in read_jsonl(output / f"selection-signals.rank{rank}.jsonl")}
        for row in normal:
            feature = selections[key(row)]
            if (feature["selection_step"] >= row["step_id"]
                    or feature["context_tokens"] != row["kv_len"]
                    or feature["selected_kv_tokens"] != row["draft_kv_len"]):
                raise RuntimeError("Pre-draft feature temporal alignment failed")
    histogram = np.bincount([r["accepted_draft_tokens"] for r in normal], minlength=K+1)
    result = dict(successful=True, k=expected_k, requests=n, normal_rounds=len(normal),
        cold_rounds=sum(r["cold_start"] for r in rows),
        accepted_histogram=histogram.tolist(), mean_accepted=float(np.arange(K+1) @ histogram / histogram.sum()),
        rank_acceptance_identical=True, predraft_signal_alignment=True,
        probability_records=len(rows), selection_records=len(normal), source_sha256=digest)
    save(output / "acceptance-summary.json", result)
    return result


def describe(hist):
    values = np.repeat(np.arange(len(hist)), hist)
    quantiles = np.quantile(values, [.1, .25, .5, .75, .9, .95])
    return dict(rounds=int(hist.sum()), mean=float(values.mean()), std=float(values.std()),
        variance=float(values.var()), **dict(zip(["p10", "p25", "p50", "p75", "p90", "p95"], map(float, quantiles))),
        probability_gt32=float(hist[33:].sum() / hist.sum()),
        probability_ge48=float(hist[48:].sum() / hist.sum()),
        probability_full63=float(hist[K] / hist.sum()), probability_le2=float(hist[:3].sum() / hist.sum()))


def distributions(case):
    output = case / "lcb128-natural"
    rows = read_jsonl(output / "rounds.jsonl")
    histogram = np.zeros(K+1, dtype=np.int64)
    binned = np.zeros((5, K+1), dtype=np.int64)
    requests = defaultdict(lambda: np.zeros(K+1, dtype=np.int64))
    for row in rows:
        if row["cold_start"]:
            continue
        value = row["accepted_draft_tokens"]
        index = int(np.searchsorted([2048, 4096, 8192, 16384], row["kv_len"], side="right"))
        histogram[value] += 1
        binned[index, value] += 1
        requests[row["request_id"]][value] += 1
    description = describe(histogram)
    description["request_equal_mean"] = float(np.mean([np.arange(K+1) @ h / h.sum() for h in requests.values()]))
    result = dict(ratio=float(case.name[1:]), k=K, histogram=histogram.tolist(), **description,
        context_bins=[dict(context=label, histogram=h.tolist(), **describe(h))
                      for label, h in zip(CONTEXT_LABELS, binned) if h.sum()])
    with (case / "per-request-acceptance.csv").open("w", newline="") as file:
        records = [dict(request_id=i, **describe(hist)) for i, hist in sorted(requests.items())]
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save(case / "distribution-summary.json", result)
    return result


def load_signals(case):
    output = case / "lcb128-natural"
    selection_meta = [read_jsonl(output / f"selection-signals.rank{r}.jsonl") for r in [0, 1]]
    if [key(r) for r in selection_meta[0]] != [key(r) for r in selection_meta[1]]:
        raise RuntimeError("TP selection record order differs")
    selection_arrays = []
    for rank in [0, 1]:
        summary = json.loads((output / f"signal-summary.rank{rank}.json").read_text())
        shape = summary["selection_record_shape"]
        selection_arrays.append(np.memmap(output / f"selection-signals.rank{rank}.f32",
            dtype="<f4", mode="r", shape=(len(selection_meta[rank]), *shape)))
    # Keep layers aligned; merge TP-sharded KV heads into global heads.
    masses = np.concatenate(selection_arrays, axis=3)
    probability_meta = read_jsonl(output / "probability-signals.rank0.jsonl")
    paths = np.memmap(output / "probability-signals.rank0.f32", dtype="<f4", mode="r",
                      shape=(len(probability_meta), 3*K + 2))
    probability_index = {key(row): i for i, row in enumerate(probability_meta)}
    selection_index = {key(row): i for i, row in enumerate(selection_meta[0])}
    rows = [r for r in read_jsonl(output / "rounds.jsonl") if not r["cold_start"]]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["request_id"]].append(row)
    x, y, ids, t, histories = [], [], [], [], []
    names = ["log_context", "effective_retention", "log_generated", "previous_accepted",
             "mean8_accepted", "std8_accepted", "mean32_accepted", "past4_conditional_mean",
             "past4_survival16", "past4_survival32", "past4_survival48"]
    mass_names = ["dropped_mass_mean", "dropped_mass_p90", "dropped_mass_max", "dropped_mass_sd",
                  "retention_fraction_mean", "retention_fraction_p10", "retention_fraction_sd",
                  "old_mass_mean", "old_concentration", "dropped_mass_change_mean",
                  "dropped_mass_change_p90", "worst_layer_mean_dropped", "source_accept_fraction"]
    names += mass_names + ["first_q_entropy", "first_q_max", "first_q_sample_logprob"]
    mass_correlations = []
    for rid, requests in grouped.items():
        requests.sort(key=lambda r: r["round"])
        accepted = np.array([r["accepted_draft_tokens"] for r in requests], dtype=float)
        curves = []
        dropped_previous = None
        for i, row in enumerate(requests):
            si, pi = selection_index[key(row)], probability_index[key(row)]
            planes = masses[si]
            retained, total, squared, _ = planes
            dropped = np.maximum(total - retained, 0)
            fraction = retained / np.maximum(total, 1e-12)
            alpha = paths[pi, :K].astype(float)
            current_curve = np.cumprod(alpha)
            feature = selection_meta[0][si]
            delta = np.zeros_like(dropped) if dropped_previous is None else dropped - dropped_previous
            mass_values = [dropped.mean(), np.quantile(dropped, .9), dropped.max(), dropped.std(),
                           fraction.mean(), np.quantile(fraction, .1), fraction.std(), total.mean(),
                           np.mean(squared / np.maximum(total**2, 1e-12)), delta.mean(),
                           np.quantile(np.abs(delta), .9), dropped.mean(axis=1).max(),
                           feature["source_accepted"] / feature["source_draft_tokens"]]
            if i >= 8:
                previous_curves = np.array(curves[-4:]).mean(axis=0)
                state = [np.log1p(row["kv_len"]), row["draft_kv_len"] / row["kv_len"],
                    np.log1p(row["kv_len"] - requests[0]["kv_len"]), accepted[i-1]/K,
                    accepted[i-8:i].mean()/K, accepted[i-8:i].std()/K,
                    accepted[max(0, i-32):i].mean()/K, previous_curves.mean(),
                    *previous_curves[np.array([16, 32, 48])-1]]
                first = [float(paths[pi, -2]), float(paths[pi, -1]),
                         float(np.log(max(float(paths[pi, K]), 1e-30)))]
                x.append(state + mass_values + first)
                y.append(int(accepted[i])); ids.append(rid); t.append(i)
                histories.append(accepted[max(0, i-64):i].astype(int))
            curves.append(current_curve)
            dropped_previous = dropped.copy()
    result = dict(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=int),
                  rid=np.asarray(ids), time=np.asarray(t), histories=histories,
                  feature_names=names, n_base=11, n_mass=13)
    if not np.isfinite(result["x"]).all():
        raise RuntimeError("Signal feature matrix contains nonfinite values")
    return result


def fit_hazard(x, y, ids):
    from scipy.optimize import minimize
    from scipy.special import expit
    count = np.bincount(ids, minlength=128)
    weights = 1 / np.maximum(count[ids], 1)
    weights /= weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    std[std < 1e-8] = 1
    design = np.column_stack([np.ones(len(x)), (x-mean)/std])
    exposure = np.clip(np.minimum(y+1, K)[:, None] - PHASE_START, 0, PHASE_END-PHASE_START)
    failure = ((y[:, None] >= PHASE_START) & (y[:, None] < PHASE_END) & (y[:, None] < K)).astype(float)
    coefficient = np.zeros((design.shape[1], len(PHASE_START)))
    f = np.average(failure, axis=0, weights=weights)
    success = np.average(exposure-failure, axis=0, weights=weights)
    coefficient[0] = np.log((f+1e-6)/(success+1e-6))
    penalty = .01

    def objective(flat):
        coef = flat.reshape(coefficient.shape)
        z = design @ coef
        loss = float(np.sum(weights[:, None] * (exposure*np.logaddexp(0, z)-failure*z))
                     + penalty*np.square(coef[1:]).sum()/2)
        gradient = design.T @ (weights[:, None]*(exposure*expit(z)-failure))
        gradient[1:] += penalty*coef[1:]
        return loss, gradient.ravel()

    result = minimize(objective, coefficient.ravel(), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=200, ftol=1e-9))
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-3:
        raise RuntimeError(f"Hazard fit failed: {result.message}")
    return mean, std, result.x.reshape(coefficient.shape)


def predict_survival(x, fitted):
    from scipy.special import expit
    mean, std, coefficient = fitted
    design = np.column_stack([np.ones(len(x)), (x-mean)/std])
    hazard = expit(design @ coefficient)
    return np.exp(np.cumsum(np.log1p(-np.minimum(hazard[:, PHASE], 1-1e-12)), axis=1))


def upper_metrics(y, upper, q, ids):
    residual = y-upper
    pinball = np.maximum(q*residual, (q-1)*residual)
    per = []
    for rid in np.unique(ids):
        mask = ids == rid
        per.append(dict(request_id=int(rid), rounds=int(mask.sum()),
            coverage=float(np.mean(residual[mask] <= 0)), mean_upper=float(upper[mask].mean()),
            shortfall=float(np.maximum(residual[mask], 0).mean()),
            excess=float(np.maximum(-residual[mask], 0).mean()), pinball=float(pinball[mask].mean()),
            at_cap=float(np.mean(upper[mask] == K))))
    means = {name: float(np.mean([r[name] for r in per])) for name in
             ["coverage", "mean_upper", "shortfall", "excess", "pinball", "at_cap"]}
    return dict(request_equal=means, per_request=per)


def signal_models(case):
    data = load_signals(case)
    x, y, ids = data["x"], data["y"], data["rid"]
    # Fixed folds by original request ID. A held-out request contributes no rows
    # to global fits; its already-observed earlier rounds remain usable features.
    models = ["constant", "history64", "state_history", "state_history_selection",
              "state_history_selection_firstdraft"]
    levels = [.8, .9, .95]
    predictions = {name: {q: np.empty(len(y), dtype=int) for q in levels} for name in models}
    for q in levels:
        predictions["history64"][q] = np.array([
            np.quantile(h, q, method="inverted_cdf") for h in data["histories"]], dtype=int)
    for fold in range(5):
        train = ids % 5 != fold
        test = ~train
        anchors = np.flatnonzero(train)
        rng = np.random.default_rng(20261020 + fold)
        # At most 128 training rows per other request, sampled once per fold.
        anchors = np.concatenate([rng.choice(np.flatnonzero(train & (ids == rid)),
                     min(128, int(np.sum(train & (ids == rid)))), replace=False)
                     for rid in np.unique(ids[train])])
        for q in levels:
            predictions["constant"][q][test] = int(np.quantile(y[anchors], q, method="inverted_cdf"))
        for name, width in [("state_history", data["n_base"]),
                            ("state_history_selection", data["n_base"] + data["n_mass"]),
                            ("state_history_selection_firstdraft", x.shape[1])]:
            fitted = fit_hazard(x[anchors, :width], y[anchors], ids[anchors])
            survival = predict_survival(x[test, :width], fitted)
            for q in levels:
                # F(a)=1-S(a+1), with a=63 right-censored at the observed cap.
                cdf = np.column_stack([1-survival, np.ones(len(survival))])
                predictions[name][q][test] = np.clip((cdf >= q).argmax(axis=1), 4, K)
    results = {name: {str(q): upper_metrics(y, pred, q, ids) for q, pred in byq.items()}
               for name, byq in predictions.items()}
    comparisons = {}
    rng = np.random.default_rng(330)
    for q in levels:
        left = results["state_history"][str(q)]["per_request"]
        right = results["state_history_selection"][str(q)]["per_request"]
        delta = np.array([a["pinball"]-b["pinball"] for a, b in zip(left, right, strict=True)])
        boots = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
        comparisons[str(q)] = dict(selection_pinball_improvement=float(delta.mean()),
            request_bootstrap95=np.quantile(boots, [.025, .975]).tolist())
    result = dict(protocol=dict(k=K, request_folds=5, folds="original request_id %5",
        minimum_previous_normal_rounds=8, quantiles=levels, fixed_L2=.01,
        training_sample_cap_per_request=128, feature_names=data["feature_names"],
        prediction_times="state_history_selection before draft; firstdraft model after exactly one draft token",
        target="Censored actual accepted draft tokens A63; current p/alpha and future q are excluded from current features",
        primary="Coverage, upper length, shortfall, excess and pinball together; cap=63 remains censored; offline results do not prove adaptive-k throughput"),
        scored_rounds=len(y), requests=int(len(np.unique(ids))), results=results, comparisons=comparisons)
    save(case / "signal-models.json", result)
    with (case / "signal-upper-predictions.csv").open("w", newline="") as file:
        names = ["request_id", "round_index", "accepted"] + [f"{m}_q{q}" for m in models for q in levels]
        writer = csv.writer(file); writer.writerow(names)
        for i in range(len(y)):
            writer.writerow([int(ids[i]), int(data["time"][i]), int(y[i])] +
                            [int(predictions[m][q][i]) for m in models for q in levels])
    return result


def analyze_all(root, fit_signals=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = []
    for ratio in RATIOS:
        case = root / f"r{ratio:.2f}"
        if (case / "lcb128-natural/validation.json").exists():
            results.append(distributions(case))
            if fit_signals and not (case / "signal-models.json").exists():
                signal_models(case)
    if not results:
        return
    save(root / "distribution-summary.json", dict(k=K, completed=results,
        pending=[r for r in RATIOS if r not in [v["ratio"] for v in results]],
        definition="Actual normal k63 verify rounds, bonus excluded; full63 is right-censored"))
    x = np.arange(K+1)
    fig, axes = plt.subplots(1, len(results), figsize=(max(6, 4*len(results)), 4.7),
                             squeeze=False, layout="constrained")
    for ax, row in zip(axes[0], results):
        hist = np.asarray(row["histogram"])
        colors = ["#2878a8"]*K + ["#c65a46"]
        ax.bar(x, hist, color=colors, width=.85)
        ax.axvline(32.5, color="#c47f35", linestyle="--", linewidth=1)
        ax.set(xlabel="Accepted draft tokens", ylabel="Verify rounds",
               title=f"r={row['ratio']:.2f}; mean={row['mean']:.2f}\nA>32: {row['probability_gt32']:.1%}; A=63: {row['probability_full63']:.1%}",
               xlim=(-1, 64))
        ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63]); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Qwen3-8B / LCB128 / k=63 / TP2 / thinking on / temperature .6\nCold rounds excluded; red bar is the observation cap")
    for ext in ["png", "pdf"]: fig.savefig(root / f"accepted-length-counts.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for row in results:
        hist = np.asarray(row["histogram"], dtype=float)
        axes[0].plot(x, hist/hist.sum(), label=f"r={row['ratio']:.2f}")
        axes[1].plot(x, np.cumsum(hist[::-1])[::-1]/hist.sum(), label=f"r={row['ratio']:.2f}")
    for ax in axes:
        ax.set_xlabel("Accepted draft tokens"); ax.legend(); ax.grid(alpha=.2)
    axes[0].set(ylabel="Fraction of verify rounds", title="Accepted-length distribution")
    axes[1].set(ylabel="P(A >= j)", title="Acceptance survival", ylim=(0, 1.02))
    for ext in ["png", "pdf"]: fig.savefig(root / f"acceptance-proportions-survival.{ext}", dpi=180)
    plt.close(fig)
    with (root / "accepted-length-distributions.csv").open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["ratio", "accepted_tokens", "rounds", "fraction"])
        for row in results:
            for value, count in enumerate(row["histogram"]):
                writer.writerow([row["ratio"], value, count, count/row["rounds"]])
    lines = ["# k=63 接受长度分布及上限信号诊断", "",
        "同一128条LCB请求，TP2、thinking开启、temperature=.6、seed330、自然EOS上限。排除cold-start全KV轮，接受数不含bonus。所有原始逐轮数据保留。", "",
        "|r|正常轮数|均值|方差|P25/P50/P90/P95|接受>32|接受≥48|全接受63|",
        "|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"|{row['ratio']:.2f}|{row['rounds']:,}|{row['mean']:.2f}|{row['variance']:.2f}|"
            f"{row['p25']:g}/{row['p50']:g}/{row['p90']:g}/{row['p95']:g}|"
            f"{row['probability_gt32']:.2%}|{row['probability_ge48']:.2%}|{row['probability_full63']:.2%}|")
    lines += ["", "![各r接受轮数](accepted-length-counts.png)", "",
              "![分布与生存曲线](acceptance-proportions-survival.png)", "",
              "63依然是右删失：满接受63不代表真实上限恰好63。k变化会改变选集刷新及生成轨迹，不能把k63轮次和既有k32轮次逐个配对。诊断归约和读回会增加开销，耗时不能当作无观察开销的吞吐基准。", "",
              "信号按可用时刻区分：选集质量在对应draft开始前可用；过去q/p条件生存曲线在上轮verify后可用；q_first熵/置信度须先执行一个draft token。当前p和整条当前alpha只用于事后分析，禁止作为本轮先验输入。"]
    for row in results:
        path = root / f"r{row['ratio']:.2f}/signal-models.json"
        if not path.exists():
            continue
        model = json.loads(path.read_text())
        lines += ["", f"## r={row['ratio']:.2f}：留出请求的上限预测", "",
                  "|模型(q=.9)|覆盖率|平均上限|少覆盖token|多余token|Pinball|上限=63比例|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, levels in model["results"].items():
            m = levels["0.9"]["request_equal"]
            lines.append(f"|{name}|{m['coverage']:.2%}|{m['mean_upper']:.2f}|{m['shortfall']:.2f}|"
                         f"{m['excess']:.2f}|{m['pinball']:.3f}|{m['at_cap']:.2%}|")
    (root / "report_zh.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Output root from experiments.acceptance.run (k63)")
    parser.add_argument("--fit-signals", action="store_true")
    args = parser.parse_args()
    analyze_all(args.root.resolve(), fit_signals=args.fit_signals)
