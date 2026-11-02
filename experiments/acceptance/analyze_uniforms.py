"""Request-held-out diagnostics for actual rejection uniforms and draft limits.

Current p/q is used only to validate rejection and, after that round, as history.
The u-informed predictions are an offline *known-u* experiment: recording native
verify-time uniforms does not implement safe pre-draft sampling or stopping.
"""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sparsespec-u-analysis-matplotlib")
import numpy as np


def read_rows(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def key(row):
    return row["request_id"], row["round"]


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def first_high(u, threshold):
    """Include the trigger position, numbered from 1; no trigger returns k."""
    high = u > threshold
    return np.where(high.any(axis=1), high.argmax(axis=1) + 1, u.shape[1])


def metrics(y, pred):
    under, over = np.maximum(y - pred, 0), np.maximum(pred - y, 0)
    return dict(rounds=len(y), mean_k=float(np.mean(pred)), coverage=float(np.mean(y <= pred)),
                mean_under=float(under.mean()), mean_over=float(over.mean()),
                proxy_cost_2under_1over=float((2 * under + over).mean()),
                mean_absolute_error=float(np.abs(y - pred).mean()))


def load(case):
    payload = json.loads((case / "worker-payload.json").read_text())
    k = payload["kwargs"]["spec_stride"]
    validation = json.loads((case / "uniform-validation.json").read_text())
    if not validation["successful"] or validation["k"] != k:
        raise ValueError("Complete a successful actual-u validation before analyzing")
    observed = read_rows(case / "rounds.jsonl")
    umeta = read_rows(case / "uniform-signals.rank0.jsonl")
    pmeta = read_rows(case / "probability-signals.rank0.jsonl")
    for kind, meta in [("uniform", umeta), ("probability", pmeta)]:
        if len({key(r) for r in meta}) != len(meta) or {key(r) for r in meta} != {key(r) for r in observed}:
            raise ValueError(f"Duplicate or missing {kind} keys")
    uniform = np.memmap(case / "uniform-signals.rank0.f64", mode="r", dtype="<f8", shape=(len(umeta), k))
    probability = np.memmap(case / "probability-signals.rank0.f32", mode="r", dtype="<f4", shape=(len(pmeta), 3*k + 2))
    uindex = {key(row): row["binary_row"] for row in umeta}
    pindex = {key(row): row["binary_row"] for row in pmeta}
    rows = [row for row in observed if not row["cold_start"] and row["actual_draft_tokens"] == k]
    if not rows:
        raise ValueError("No complete normal draft window; shorter rounds are censored and cannot label A_k")
    rows.sort(key=key)
    u = np.asarray(uniform[[uindex[key(row)] for row in rows]], dtype=np.float64)
    alpha = np.asarray(probability[[pindex[key(row)] for row in rows], :k], dtype=np.float64)
    if not np.isfinite(u).all() or not np.isfinite(alpha).all():
        raise ValueError("A full normal window has missing u or alpha")
    y = np.asarray([row["accepted_draft_tokens"] for row in rows])
    groups = np.asarray([row["request_id"] for row in rows])
    lengths = np.asarray([row["actual_draft_tokens"] for row in observed if not row["cold_start"]])
    counts = dict(total_verify_rounds=len(observed), cold_rounds=sum(row["cold_start"] for row in observed),
                  normal_rounds=len(lengths), full_window_rounds=len(rows),
                  shorter_normal_rounds=int(np.sum(lengths < k)),
                  normal_draft_lengths={str(n): int(np.sum(lengths == n)) for n in np.unique(lengths)},
                  requests=len(payload["requests"]), requests_with_full_window=len(np.unique(groups)))
    return k, payload, rows, u, alpha, y, groups, counts


def features(rows, u, alpha, y, k):
    """Every alpha/acceptance feature uses STRICTLY earlier full-window rounds."""
    history = defaultdict(lambda: deque(maxlen=64))
    baseline, extra = [], []
    checkpoints = sorted(set(min(k, n) for n in [1, 2, 4, 8, 16, 24, k]))
    thresholds = [.5, .7, .8, .9, .95, .975, .99]
    for i, row in enumerate(rows):
        old = list(history[row["request_id"]])
        a = np.array([item[0] for item in old], dtype=float)
        state = [np.log1p(row["kv_len"]), np.log1p(row["draft_kv_len"]),
                 row["draft_kv_len"] / max(1, row["kv_len"]), np.log1p(row["round"]), len(old)]
        for window in [1, 4, 16, 64]:
            values = a[-window:]
            state.extend([float(values.mean()), float(values.std()), float(np.mean(values == k))]
                         if len(values) else [np.nan]*3)
        if old:
            old_alpha = np.stack([item[1] for item in old])
            marginal_survival = np.cumprod(old_alpha, axis=1).mean(axis=0)
            # Laplace smoothing toward one accepted observation. Dependence
            # across positions is NOT assumed exact; this is a model feature.
            pass_probability = (np.sum(old_alpha >= u[i], axis=0) + 1) / (len(old) + 1)
            conditional_survival = np.cumprod(pass_probability)
        else:
            marginal_survival = pass_probability = conditional_survival = np.full(k, np.nan)
        baseline.append(state + marginal_survival.tolist())
        additional = u[i].tolist()
        additional += [float(u[i, :n].max()) for n in checkpoints]
        additional += [float(first_high(u[i:i+1], threshold)[0]) for threshold in thresholds]
        additional += pass_probability.tolist() + conditional_survival.tolist()
        extra.append(additional)
        history[row["request_id"]].append((y[i], alpha[i]))
    names = ["log_context", "log_selected_kv", "selected_fraction", "log_round", "history_count"]
    names += [f"history{w}_{s}" for w in [1, 4, 16, 64] for s in ["mean", "std", "full_fraction"]]
    names += [f"past_alpha_survival_{j+1}" for j in range(k)]
    unames = [f"current_u_{j+1}" for j in range(k)]
    unames += [f"prefix_max_u_{j}" for j in checkpoints]
    unames += [f"first_u_above_{t}" for t in thresholds]
    unames += [f"history_pass_probability_at_u_{j+1}" for j in range(k)]
    unames += [f"history_u_survival_{j+1}" for j in range(k)]
    x = np.asarray(baseline, dtype=np.float32)
    added = np.asarray(extra, dtype=np.float32)
    assert x.shape[1] == len(names) and added.shape[1] == len(unames)
    return x, np.concatenate([x, added], axis=1), names, unames


def bootstrap_difference(y, base, informed, groups, seed=330):
    difference = 2*np.maximum(y-base, 0) + np.maximum(base-y, 0)
    difference -= 2*np.maximum(y-informed, 0) + np.maximum(informed-y, 0)
    ids, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(2000, len(ids)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(cost_reduction=float(difference.mean()),
                request_bootstrap_95ci=[float(v) for v in np.quantile(values, [.025, .975])],
                requests=len(ids), draws=2000)


def fit_models(rows, u, alpha, y, groups, k):
    local_dependencies = Path(__file__).resolve().parents[2] / ".repro/analysis-packages"
    if local_dependencies.is_dir():
        sys.path.insert(0, str(local_dependencies))
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least five independent requests; use --descriptive-only for smoke")
    x, xu, names, unames = features(rows, u, alpha, y, k)
    predictions, results = {}, {}
    splits = list(GroupKFold(n_splits=5).split(x, y, groups))
    for quantile in [2/3, .9]:
        tag = f"q{quantile:.3f}"
        estimates = {name: np.empty(len(y)) for name in ["constant", "state_history", "state_history_u"]}
        for fold, (train, test) in enumerate(splits):
            estimates["constant"][test] = np.clip(np.ceil(np.quantile(y[train], quantile)), 4, k)
            for name, values in [("state_history", x), ("state_history_u", xu)]:
                model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                    learning_rate=.08, max_iter=100, max_leaf_nodes=15, min_samples_leaf=100,
                    l2_regularization=1., max_bins=64, early_stopping=False, random_state=330)
                with threadpool_limits(limits=2):
                    model.fit(values[train], y[train])
                    estimates[name][test] = np.clip(np.ceil(model.predict(values[test])), 4, k)
            print(f"Fitted {tag}, request-held-out fold {fold+1}/5", flush=True)
        predictions[tag] = estimates
        results[tag] = {name: metrics(y, pred) for name, pred in estimates.items()}
        results[tag]["paired_u_gain"] = bootstrap_difference(
            y, estimates["state_history"], estimates["state_history_u"], groups)
    return results, predictions, dict(state_history=names, added_u=unames)


def analyze(case, output, descriptive_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    output.mkdir(parents=True, exist_ok=True)
    k, payload, rows, u, alpha, y, groups, counts = load(case)
    summary = dict(case=str(case), k=k, ratio=payload["kwargs"]["budget_ratio"], counts=counts,
        dynamic_admission=payload["kwargs"].get("compact_dynamic_admission", False),
        max_num_batched_tokens=payload["kwargs"].get("max_num_batched_tokens"),
        data_scope="Full normal windows only; cold and shorter pressure-triggered windows reported separately",
        u_availability="Offline known-u test using actual native verify-time draws, not a deployed predraw policy",
        mean_accepted=float(y.mean()), acceptance_histogram=np.bincount(y, minlength=k+1).tolist(),
        uniform_min=float(u.min()), uniform_max=float(u.max()), uniform_mean=float(u.mean()),
        uniform_equal_one=int(np.sum(u == 1)), analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Include only positions actually reached by rejection. Later positions
    # are not independent observed rejects and must not be labeled as such.
    at_risk = y[:, None] >= np.arange(k)
    rejected = y[:, None] == np.arange(k)
    bins = np.minimum((u*10).astype(int), 9)
    deciles = []
    for decile in range(10):
        selected = (bins == decile) & at_risk
        n = int(selected.sum())
        failures = int(np.sum(rejected & selected))
        deciles.append(dict(u_lower=decile/10, u_upper=(decile+1)/10,
                            reached_positions=n, rejects=failures,
                            rejection_rate=failures/n if n else None))
    summary["u_deciles_at_reached_positions"] = deciles
    thresholds = [.5, .7, .8, .9, .95, .975, .99]
    summary["first_high_u_heuristics"] = {
        str(t): metrics(y, np.clip(first_high(u, t), 4, k)) for t in thresholds}
    summary["heuristic_definition"] = "k=max(4, first 1-based index with u>threshold); no hit gives k_max; trigger position included"
    if not descriptive_only:
        fitted, predictions, feature_names = fit_models(rows, u, alpha, y, groups, k)
        summary["request_held_out_models"] = fitted
        save(output / "feature-names.json", feature_names)
        with (output / "held-out-predictions.csv").open("w") as file:
            writer = csv.writer(file)
            columns = [(tag, name) for tag, values in predictions.items() for name in values]
            writer.writerow(["request_id", "round", "context", "accepted"] + [f"{t}_{n}" for t,n in columns])
            for i, row in enumerate(rows):
                writer.writerow([row["request_id"], row["round"], row["kv_len"], int(y[i])] +
                                [int(predictions[t][n][i]) for t,n in columns])
    save(output / "summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].bar(np.arange(k+1), np.bincount(y, minlength=k+1), color="#287FA6")
    axes[0].set(xlabel="Accepted draft tokens (bonus excluded)", ylabel="Verify rounds",
                title=f"r={summary['ratio']:.2f}, k={k}: full normal windows")
    rates = [v["rejection_rate"] for v in deciles]
    axes[1].bar(np.arange(10)+.5, rates, width=.85, color="#C64B3D")
    axes[1].set(xticks=np.arange(11), xticklabels=[f"{i/10:.1f}" for i in range(11)],
                xlabel="Actual native uniform u", ylabel="Rejection rate at reached positions",
                title="Does a larger u indicate more rejection?")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"u-and-acceptance.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# 本轮实际 u 对 draft 上限是否有指导意义", "",
        f"配置：r={summary['ratio']:.2f}，k 上限={k}，TP2，128 集合中的 {counts['requests']} 个请求；"
        f"动态准入={summary['dynamic_admission']}，每轮 token 预算={summary['max_num_batched_tokens']}。", "",
        f"共 {counts['total_verify_rounds']:,} 轮；首次全 KV {counts['cold_rounds']:,} 轮；"
        f"正常满窗口 {counts['full_window_rounds']:,} 轮；正常短窗口 {counts['shorter_normal_rounds']:,} 轮。", "",
        "当前 u 是原生 rejection 在 verify 时使用的实际随机数，不新增 RNG 调用。"
        "以下‘已知 u’结果是离线信号诊断，不代表已实现提前抽样或动态截断。", "",
        "![接受分布与 u 拒绝风险](u-and-acceptance.png)", "",
        "## 高 u 的直接信号", "",
        "只统计 rejection 实际到达的位置；首次拒绝以后的随机数不能标为本轮实际拒绝。"
        "高 u 与拒绝的关系部分来自接受判定本身；相关性不是上限预测收益的充分证据。", "",
        "|u 区间|到达位置数|拒绝率|", "|---|---:|---:|"]
    for value in deciles:
        rate = f"{value['rejection_rate']:.2%}" if value["rejection_rate"] is not None else "无数据"
        lines.append(f"|[{value['u_lower']:.1f},{value['u_upper']:.1f})|{value['reached_positions']:,}|{rate}|")
    lines += ["", "## 遇到第一个高 u 的简单上限", "",
        "候选 k 为第一个 u 超过阈值的位置（包含该位置），限制在 [4,k_max]；无命中则取 k_max。"
        "以下 2×低估 + 1×高估为代理损失，不能换算成吞吐收益。", "",
        "|阈值|平均 k|覆盖 A 的比例|少覆盖 token|多余 token|代理损失|", "|---:|---:|---:|---:|---:|---:|"]
    for threshold, value in summary["first_high_u_heuristics"].items():
        lines.append(f"|{threshold}|{value['mean_k']:.2f}|{value['coverage']:.2%}|"
                     f"{value['mean_under']:.2f}|{value['mean_over']:.2f}|{value['proxy_cost_2under_1over']:.3f}|")
    if not descriptive_only:
        lines += ["", "## 按请求隔离的 5 折预测", "",
            "基础特征只含轮前上下文、选集预算和该请求过去最多64个正常满窗口的接受数及 p/q 条件生存曲线。"
            "增强模型加入当前 u、前缀最大 u、首个高 u 位置，以及历史 alpha 在当前 u 下的通过比例。"
            "当前轮 alpha、p、q、实际接受长度均不作为输入。", "",
            "两种模型使用相同请求隔离划分、超参数和训练量；取整后 k 限制在 [4,k_max]。"
            "q=2/3 对应上述代理损失；q=.9 用于上限覆盖诊断，不能直接用它的均值比较代理最优。", "",
            "|目标|模型|平均 k|覆盖率|少覆盖 token|多余 token|代理损失|", "|---|---|---:|---:|---:|---:|---:|"]
        for tag, values in summary["request_held_out_models"].items():
            for name in ["constant", "state_history", "state_history_u"]:
                v = values[name]
                lines.append(f"|{tag}|{name}|{v['mean_k']:.2f}|{v['coverage']:.2%}|"
                             f"{v['mean_under']:.2f}|{v['mean_over']:.2f}|{v['proxy_cost_2under_1over']:.3f}|")
            gain = values["paired_u_gain"]
            lo, hi = gain["request_bootstrap_95ci"]
            lines += ["", f"{tag}：加入 u 的代理损失减少 {gain['cost_reduction']:.4f}，"
                      f"请求 bootstrap 95% 区间 [{lo:.4f}, {hi:.4f}]。", ""]
    lines += ["", "## 解释范围", "",
        "- k_max 的全部接受是右删失，不能确定其后的首次拒绝点。",
        "- 满窗口的预测分析仅代表该子集；压力下提前验证的短窗口单独计数，不补成满窗口标签。",
        "- u 小于1，u 接近1也不必拒绝；alpha=1的位置可以通过任意合法 u。",
        "- 纯 u 和当前接受长度的关系必须看独立请求上的增量预测收益，不能只看事后相关性。",
        "- 如后续实现依 u 选窗口，应另审查采样无偏性；本实验没有更改采样或提前停止规则。", ""]
    (output / "findings_zh.md").write_text("\n".join(lines))
    print(f"Actual-u analysis saved to {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptive-only", action="store_true")
    args = parser.parse_args()
    analyze(args.case.resolve(), args.output.resolve(), args.descriptive_only)


if __name__ == "__main__":
    main()
