"""Build the report/slide figures from the CSV tables written by run_experiment.py."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, GRAY, INK, MUTED = "#2E86AB", "#E76F51", "#B8BEC6", "#1F2933", "#5F6B7A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#C9CED6",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left",
})


def risk_by_active_days(res, figs):
    d = pd.read_csv(res / "05_risk_by_active_days.csv")
    labels = ["0", "1–3", "4–6", "7–9", "10–12", "13–15", "16–18", "19–21", "22–24", "25–28"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, d["at_risk_rate"] * 100, color=ORANGE, width=0.7, edgecolor="white", linewidth=2)
    for i, v in enumerate(d["at_risk_rate"] * 100):
        if i in (0, len(d) - 1):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.axhline(44.1, color=MUTED, lw=1, ls="--")
    ax.text(len(d) - 0.6, 46, "overall 44%", color=MUTED, fontsize=10, ha="right")
    ax.set_ylim(0, 92)
    ax.set_ylabel("Students at risk (%)")
    ax.set_xlabel("Active days on the VLE in weeks 1–4")
    ax.grid(axis="y", color="#E6E9ED"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figs / "risk_by_active_days.png", dpi=220); plt.close(fig)


def model_comparison(res, figs):
    d = pd.read_csv(res / "04_model_comparison.csv")
    d = d[d.Model != "Dummy Baseline"].sort_values("F1")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = range(len(d))
    ax.barh([i + 0.19 for i in y], d["ROC-AUC"], height=0.36, color=GRAY, label="ROC-AUC",
            edgecolor="white", linewidth=1.5)
    ax.barh([i - 0.19 for i in y], d["F1"], height=0.36, color=BLUE, label="F1 (at-risk class)",
            edgecolor="white", linewidth=1.5)
    for i, (f1, auc) in enumerate(zip(d["F1"], d["ROC-AUC"])):
        ax.text(f1 + 0.005, i - 0.19, f"{f1:.3f}", va="center", fontsize=10, color=INK)
        ax.text(auc + 0.005, i + 0.19, f"{auc:.3f}", va="center", fontsize=10, color=MUTED)
    ax.set_yticks(list(y)); ax.set_yticklabels(d["Model"])
    ax.set_xlim(0.5, 0.8); ax.set_xlabel("Score on held-out test set (dummy baseline: F1 0.000, AUC 0.500)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "model_comparison.png", dpi=220); plt.close(fig)


def ablation(res, figs):
    a = pd.read_csv(res / "05a_ablation_feature_sets_cv.csv")
    order = ["Profile only", "Engagement only", "Profile + engagement"]
    lr = a[a.Model == "Logistic Regression"].set_index("Feature set").loc[order]
    gb = a[a.Model == "Gradient Boosting"].set_index("Feature set").loc[order]
    gap = (lr["F1 mean"] - gb["F1 mean"]) * 100
    sd = ((lr["F1 SD"] ** 2 + gb["F1 SD"] ** 2) ** 0.5) * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cols = [GRAY, BLUE, BLUE]
    ax.bar(["Profile only\n(no engagement)", "Engagement\nonly", "Profile +\nengagement"],
           gap, yerr=sd, color=cols, width=0.6, capsize=4, ecolor=MUTED,
           edgecolor="white", linewidth=2)
    for i, v in enumerate(gap):
        ax.text(i, v + sd.iloc[i] + 0.12, f"{v:+.1f}", ha="center", fontsize=12,
                fontweight="bold", color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("LR minus GB, F1 points\n(5-fold CV, ±1 SD)")
    ax.set_ylim(-1.2, 2.8)
    fig.tight_layout(); fig.savefig(figs / "ablation_lr_minus_gb.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    a = ap.parse_args()
    res, figs = Path(a.results_dir), Path(a.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)
    risk_by_active_days(res, figs)
    model_comparison(res, figs)
    ablation(res, figs)
    print("figures written to", figs.resolve())
