"""
Publication figures + statistical ranking across all 4 SILSO resolutions.

Requires outputs/<resolution>/{metrics.csv, predictions.csv,
chronos_metrics.csv, chronos_predictions.csv} to already exist
(train_all_datasets.py + chronos_all.py).

Produces, under outputs/figures/:
  - box_errors.png/.pdf          abs-error distribution per model, 2x2 (one per resolution)
  - lines_actual_vs_pred.png/.pdf  actual vs predicted time series, 2x2
  - scatter_<resolution>.png/.pdf  actual-vs-predicted scatter, one figure per resolution
  - taylor_diagrams.png/.pdf      2x2 Taylor diagrams (one per resolution)
  - critical_difference.png/.pdf  Friedman/Nemenyi CD diagram across all 4 resolutions
  - summary_metrics.csv           full RMSE/MAE/R2/MAPE table, all models x all resolutions
  - friedman_results.json         Friedman statistic, p-value, average ranks, CD value
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

from data import RESOLUTIONS

OUT_ROOT = Path(r"C:\Users\amank\Downloads\sunspot_transformer\outputs")
FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RES_LABEL = {
    "daily": "Daily",
    "monthly_mean": "Monthly mean",
    "monthly_smoothed": "Monthly smoothed (13-mo)",
    "yearly": "Yearly",
}

EXCLUDE_MODELS = {"Proposed (LSTM+PatchTST)", "SSN-Hybrid (proposed)"}

MODEL_ORDER = ["Naive Persistence", "RNN", "GRU", "LSTM", "BiLSTM", "CNN1D",
               "Chronos-Bolt (zero-shot)", "PatchTST", "ResidualLSTM (Proposed)"]

MODEL_COLOR = {
    "Naive Persistence": "#9a978f",
    "RNN": "#c9891a",
    "GRU": "#1baf7a",
    "LSTM": "#4a3aa7",
    "BiLSTM": "#e87ba4",
    "CNN1D": "#eda100",
    "Chronos-Bolt (zero-shot)": "#e34948",
    "PatchTST": "#eb6834",
    "ResidualLSTM (Proposed)": "#2a78d6",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_all():
    data = {}
    for res in RESOLUTIONS:
        d = OUT_ROOT / res
        metrics = pd.read_csv(d / "metrics.csv")
        metrics = metrics[~metrics["Model"].isin(EXCLUDE_MODELS)]
        preds = pd.read_csv(d / "predictions.csv", parse_dates=["date"])
        cmet_path = d / "chronos_metrics.csv"
        cpred_path = d / "chronos_predictions.csv"
        if cmet_path.exists():
            cmet = pd.read_csv(cmet_path)
            metrics = pd.concat([metrics, cmet], ignore_index=True)
            cpred = pd.read_csv(cpred_path, parse_dates=["date"])
            preds = preds.merge(cpred[["date", "chronos_pred"]], on="date", how="left")
            preds = preds.rename(columns={"chronos_pred": "Chronos-Bolt (zero-shot)"})
        data[res] = {"metrics": metrics, "preds": preds}
    return data


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def present_models(metrics_df):
    return [m for m in MODEL_ORDER if m in set(metrics_df["Model"])]


# ---------------------------------------------------------------- Box plot
def plot_box(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, res in zip(axes.flat, RESOLUTIONS):
        preds = data[res]["preds"]
        models = [m for m in present_models(data[res]["metrics"]) if m in preds.columns and m != "Naive Persistence"]
        models = ["Naive Persistence"] + models if "Naive Persistence" in preds.columns else models
        errs = [np.abs(preds["actual"] - preds[m]).dropna().values for m in models]
        colors = [MODEL_COLOR.get(m, "#888") for m in models]
        bp = ax.boxplot(errs, patch_artist=True, showfliers=False, widths=0.6,
                         medianprops=dict(color="black", linewidth=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#333")
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=38, ha="right", fontsize=7.5)
        ax.set_ylabel("Absolute error")
        ax.set_title(RES_LABEL[res], fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Test-set absolute error distribution by model", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "box_errors")


# ---------------------------------------------------------------- Line plot
LINE_WINDOW = {"daily": 1500, "monthly_mean": 220, "monthly_smoothed": 220, "yearly": 50}
LINE_MODELS = ["Naive Persistence", "GRU", "ResidualLSTM (Proposed)"]


def plot_lines(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, res in zip(axes.flat, RESOLUTIONS):
        preds = data[res]["preds"].sort_values("date")
        w = LINE_WINDOW[res]
        sub = preds.tail(w)
        ax.plot(sub["date"], sub["actual"], color="#1a1a1a", lw=1.5, label="Actual")
        for m in LINE_MODELS:
            if m in sub.columns:
                style = dict(lw=1.1, alpha=0.9)
                if m == "Naive Persistence":
                    style.update(lw=1.0, alpha=0.6, linestyle="--")
                ax.plot(sub["date"], sub[m], color=MODEL_COLOR.get(m, "#888"), label=m, **style)
        ax.set_title(RES_LABEL[res], fontsize=10.5, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.suptitle("Held-out actual vs. predicted", fontsize=13, fontweight="bold", y=1.1)
    fig.tight_layout()
    save_fig(fig, "lines_actual_vs_pred")


# ---------------------------------------------------------------- Scatter plot
SCATTER_MODELS = ["Naive Persistence", "LSTM", "GRU", "Chronos-Bolt (zero-shot)", "ResidualLSTM (Proposed)"]


def plot_scatter(data):
    for res in RESOLUTIONS:
        preds = data[res]["preds"]
        models = [m for m in SCATTER_MODELS if m in preds.columns]
        fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.4), sharex=True, sharey=True)
        if len(models) == 1:
            axes = [axes]
        lo = float(min(preds["actual"].min(), preds[models].min().min()))
        hi = float(max(preds["actual"].max(), preds[models].max().max()))
        pad = (hi - lo) * 0.05
        for ax, m in zip(axes, models):
            sub = preds[["actual", m]].dropna()
            r2 = np.corrcoef(sub["actual"], sub[m])[0, 1] ** 2
            ax.scatter(sub["actual"], sub[m], s=9, alpha=0.45, color=MODEL_COLOR.get(m, "#888"),
                       edgecolors="none")
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999", lw=1, linestyle="--")
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{m}\nR\u00b2={r2:.3f}", fontsize=9)
            ax.set_xlabel("Actual");
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("Predicted")
        fig.suptitle(f"Actual vs. predicted \u2014 {RES_LABEL[res]}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, f"scatter_{res}")


# ---------------------------------------------------------------- Taylor diagram
def taylor_diagram(ax, refstd, stds, corrs, labels, colors):
    max_std = max(1.35 * refstd, 1.15 * max(stds))
    ax.set_xlim(0, max_std)
    ax.set_ylim(0, max_std)
    ax.set_aspect("equal")

    corr_ticks = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    for c in corr_ticks:
        theta = np.arccos(c)
        ax.plot([0, max_std * np.cos(theta)], [0, max_std * np.sin(theta)], color="#cfcfc7", lw=0.6, zorder=0)
        ax.text(1.02 * max_std * np.cos(theta), 1.02 * max_std * np.sin(theta), f"{c}",
                fontsize=6.5, color="#888", ha="left", va="bottom")

    for r in np.linspace(0.25, np.ceil(max_std * 4) / 4, int(np.ceil(max_std * 4))):
        theta = np.linspace(0, np.pi / 2, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#e8e6df", lw=0.6, zorder=0)

    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(refstd * np.cos(theta), refstd * np.sin(theta), color="#999", lw=1, linestyle=":", zorder=0)

    rgrid = np.linspace(0, max_std, 120)
    tgrid = np.linspace(0, np.pi / 2, 120)
    R, T = np.meshgrid(rgrid, tgrid)
    X, Y = R * np.cos(T), R * np.sin(T)
    CRMSE = np.sqrt(refstd**2 + R**2 - 2 * refstd * R * np.cos(T))
    cs = ax.contour(X, Y, CRMSE, levels=6, colors="#c9891a", linewidths=0.6, linestyles="--", alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

    ax.plot(refstd, 0, marker="*", color="#1a1a1a", markersize=14, zorder=5, label="Reference (actual)")
    for std, corr, label, color in zip(stds, corrs, labels, colors):
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot(std * np.cos(theta), std * np.sin(theta), marker="o", color=color,
                markersize=7, markeredgecolor="white", markeredgewidth=0.6, zorder=6, label=label)

    ax.set_xlabel("Standard deviation (normalized)")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)


def plot_taylor(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, res in zip(axes.flat, RESOLUTIONS):
        preds = data[res]["preds"]
        models = [m for m in present_models(data[res]["metrics"]) if m in preds.columns]
        actual_std = preds["actual"].std()
        stds, corrs, labels, colors = [], [], [], []
        for m in models:
            sub = preds[["actual", m]].dropna()
            stds.append(sub[m].std() / actual_std)
            corrs.append(np.corrcoef(sub["actual"], sub[m])[0, 1])
            labels.append(m)
            colors.append(MODEL_COLOR.get(m, "#888"))
        taylor_diagram(ax, 1.0, stds, corrs, labels, colors)
        ax.set_title(RES_LABEL[res], fontsize=11, fontweight="bold", pad=14)
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=8.5)
    fig.suptitle("Taylor diagrams \u2014 correlation, normalized std. dev., centered RMSE (dashed, gold)",
                 fontsize=12.5, fontweight="bold", y=1.09)
    fig.tight_layout()
    save_fig(fig, "taylor_diagrams")


# ---------------------------------------------------------------- Friedman + CD diagram
NEMENYI_Q_ALPHA05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                     8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def friedman_and_cd(data):
    summary = build_summary_table(data)
    pivot = summary.pivot_table(index="Resolution", columns="Model", values="RMSE_norm")
    pivot = pivot.dropna(axis=1)  # keep only models present in every resolution
    models = list(pivot.columns)
    k = len(models)
    n = len(pivot)

    stat, pval = friedmanchisquare(*[pivot[m].values for m in models])

    ranks = pivot.rank(axis=1, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()

    q_alpha = NEMENYI_Q_ALPHA05.get(k, 3.164)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    result = {
        "n_models": k, "n_resolutions": n,
        "friedman_statistic": float(stat), "friedman_pvalue": float(pval),
        "critical_difference": float(cd),
        "average_ranks": {m: float(r) for m, r in avg_rank.items()},
    }
    with open(FIG_DIR / "friedman_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    # CD diagram
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.42 * k))
    ranks_sorted = avg_rank.sort_values()
    y_positions = np.arange(len(ranks_sorted))[::-1]
    ax.hlines(y_positions, xmin=1, xmax=k, color="#ddd", lw=1, zorder=0)
    for y, (m, r) in zip(y_positions, ranks_sorted.items()):
        ax.plot(r, y, "o", color=MODEL_COLOR.get(m, "#2a78d6"), markersize=9, zorder=3)
        ax.text(k + 0.15, y, f"{m}  (rank {r:.2f})", va="center", fontsize=9)

    best_rank = ranks_sorted.iloc[0]
    ax.plot([1, 1 + cd], [len(ranks_sorted) + 0.6, len(ranks_sorted) + 0.6], color="#1a1a1a", lw=2)
    ax.text(1 + cd / 2, len(ranks_sorted) + 0.85, f"CD = {cd:.2f}", ha="center", fontsize=9)

    for i, (m, r) in enumerate(ranks_sorted.items()):
        if abs(r - best_rank) <= cd:
            ax.plot([min(r, best_rank), max(r, best_rank)], [y_positions[i] - 0.28] * 2,
                     color="#2a78d6", lw=3, alpha=0.35, zorder=1)

    ax.set_xlim(0.5, k + 3.6)
    ax.set_ylim(-1, len(ranks_sorted) + 1.6)
    ax.set_xlabel("Average rank across 4 resolutions (lower = better)")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"Friedman \u03c7\u00b2={stat:.2f}, p={pval:.4f}  |  Nemenyi CD={cd:.2f} (\u03b1=0.05, n={n} resolutions)\n"
                 "blue bar = not significantly different from the best model",
                 fontsize=9.5)
    fig.tight_layout()
    save_fig(fig, "critical_difference")
    return result


def build_summary_table(data):
    rows = []
    for res in RESOLUTIONS:
        m = data[res]["metrics"].copy()
        actual_std = data[res]["preds"]["actual"].std()
        m["Resolution"] = RES_LABEL[res]
        m["RMSE_norm"] = m["RMSE"] / actual_std  # NRMSE, comparable across resolutions
        rows.append(m)
    summary = pd.concat(rows, ignore_index=True)
    cols = ["Resolution", "Model", "RMSE", "RMSE_norm", "MAE", "R2", "MAPE"]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary.to_csv(FIG_DIR / "summary_metrics.csv", index=False)
    return summary


if __name__ == "__main__":
    data = load_all()
    print("Building box plot...")
    plot_box(data)
    print("Building line plot...")
    plot_lines(data)
    print("Building scatter plots...")
    plot_scatter(data)
    print("Building Taylor diagrams...")
    plot_taylor(data)
    print("Running Friedman test + CD diagram...")
    friedman_and_cd(data)
    print(f"\nAll figures saved to {FIG_DIR}")
