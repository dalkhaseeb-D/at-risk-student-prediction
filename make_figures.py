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


def cross_validation_table(res, figs):
    """Render results/07_cross_validation.csv as a table image."""
    d = pd.read_csv(res / "07_cross_validation.csv")
    metric_names = {"balanced_accuracy": "Balanced accuracy", "precision": "Precision",
                    "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
    d["Gap"] = d["Train mean"] - d["Validation mean"]

    header = ["Model", "Metric", "Train mean", "Validation mean ± SD", "Train − val. gap"]
    rows, models = [], []
    for model, grp in d.groupby("Model", sort=False):
        for i, (_, r) in enumerate(grp.iterrows()):
            rows.append([model if i == 0 else "", metric_names.get(r["Metric"], r["Metric"]),
                         f"{r['Train mean']:.3f}",
                         f"{r['Validation mean']:.3f} ± {r['Validation SD']:.3f}",
                         f"{r['Gap']:+.3f}"])
            models.append((model, r["Metric"]))

    fig, ax = plt.subplots(figsize=(10, 0.285 * (len(rows) + 1) + 0.9))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                   colWidths=[0.24, 0.20, 0.15, 0.23, 0.18])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.45)

    group_fill = ["#FFFFFF", "#F3F5F8"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DCE1E7")
        if r == 0:  # header
            cell.set_facecolor("#14213D")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        model, metric = models[r - 1]
        group_idx = list(dict.fromkeys(m for m, _ in models)).index(model)
        cell.set_facecolor("#DDEEF5" if model == "Logistic Regression" else group_fill[group_idx % 2])
        cell.get_text().set_color(INK)
        if c == 0:
            cell.get_text().set_fontweight("bold")
            cell._loc = "left"
        if metric == "f1" and c > 0:  # main metric
            cell.get_text().set_fontweight("bold")

    ax.set_title("5-fold stratified cross-validation (seed 42): train vs validation scores",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.text(0, -0.02, "Gap = train mean − validation mean; a large gap means overfitting\n"
            "(Logistic Regression ≈ 0; tree ensembles 0.05–0.09 in ROC-AUC). Bold = main metric (F1).",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(figs / "cross_validation_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def cross_validation_bars(res, figs):
    """Bar-chart version of 07_cross_validation.csv: train vs validation per model and metric."""
    d = pd.read_csv(res / "07_cross_validation.csv")
    metrics = [("f1", "F1 (main metric)"), ("roc_auc", "ROC-AUC"), ("recall", "Recall"),
               ("precision", "Precision"), ("balanced_accuracy", "Balanced accuracy")]
    models = list(dict.fromkeys(d["Model"]))
    short = {"Logistic Regression": "Logistic\nRegression", "Gradient Boosting": "Gradient\nBoosting",
             "Random Forest": "Random\nForest", "Decision Tree": "Decision\nTree"}
    x = range(len(models))
    w = 0.38

    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        m = d[d["Metric"] == key].set_index("Model").loc[models]
        ax.bar([i - w / 2 for i in x], m["Train mean"], width=w, color=GRAY,
               edgecolor="white", linewidth=1.5, label="Train (mean of 5 folds)")
        ax.bar([i + w / 2 for i in x], m["Validation mean"], width=w, color=BLUE,
               yerr=m["Validation SD"], capsize=3, ecolor=INK, error_kw={"lw": 1},
               edgecolor="white", linewidth=1.5, label="Validation (mean ± SD)")
        for i, (tr, va, sd) in enumerate(zip(m["Train mean"], m["Validation mean"],
                                             m["Validation SD"])):
            gap = tr - va
            ax.text(i, max(tr, va + sd) + 0.012, f"gap\n{gap:+.3f}", ha="center", va="bottom",
                    fontsize=8.5, linespacing=1.0,
                    color="#B8472C" if gap > 0.03 else MUTED,
                    fontweight="bold" if gap > 0.03 else "normal")
        ax.set_title(title)
        ax.set_xticks(list(x)); ax.set_xticklabels([short.get(n, n) for n in models], fontsize=9.5)
        ax.set_ylim(0.5, 0.9)
        ax.grid(axis="y", color="#E6E9ED"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("5-fold cross-validation: train vs validation scores", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, "Gap = train − validation, shown above each pair (red bold = over 0.03, i.e. overfitting). "
             "Logistic Regression's train and validation scores match; the tree models score "
             "much higher on training data than on unseen data. Exact values: cross_validation_table.png. Y-axis starts at 0.5.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "cross_validation_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def feature_importance(res, figs):
    """Permutation importance of the Logistic Regression model (drop in test F1)."""
    d = pd.read_csv(res / "07_permutation_importance.csv").sort_values("Importance")
    nice = {
        "active_days_wk1_4": "Active days (weeks 1–4)",
        "unique_sites_wk1_4": "Unique pages visited (weeks 1–4)",
        "total_clicks_wk1_4": "Total clicks (weeks 1–4)",
        "interaction_records_wk1_4": "Interaction records (weeks 1–4)",
        "code_module": "Module", "code_presentation": "Presentation (year/term)",
        "highest_education": "Highest education", "imd_band": "Deprivation (IMD band)",
        "region": "Region", "disability": "Disability", "gender": "Gender", "age_band": "Age band",
        "num_of_prev_attempts": "Previous attempts", "studied_credits": "Studied credits",
    }
    engagement = d["Feature"].str.endswith("_wk1_4")
    colors = [BLUE if e else GRAY for e in engagement]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.barh([nice.get(f, f) for f in d["Feature"]], d["Importance"], xerr=d["SD"],
            color=colors, height=0.66, capsize=3, ecolor=MUTED, error_kw={"lw": 1},
            edgecolor="white", linewidth=1.5)
    for i, (v, s) in enumerate(zip(d["Importance"], d["SD"])):
        ax.text(max(v, 0) + s + 0.002, i, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Drop in test F1 when the feature is shuffled\n(mean ± SD over 5 repeats)")
    fig.suptitle("Permutation feature importance (Logistic Regression)", x=0.02, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    ax.grid(axis="x", color="#E6E9ED"); ax.set_axisbelow(True)
    ax.set_xlim(min(-0.01, d["Importance"].min() - 0.005), d["Importance"].max() + 0.02)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Engagement (VLE clicks, days 0–27)"),
                       Patch(color=GRAY, label="Student profile")],
              loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    cross_validation_table(res, figs)
    cross_validation_bars(res, figs)
    feature_importance(res, figs)
    print("figures written to", figs.resolve())
