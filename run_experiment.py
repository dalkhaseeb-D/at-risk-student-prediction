"""
End-to-end experiment: early prediction of at-risk students on OULAD.

Runs every step of the CSBP711 A1 pipeline from a clean checkout:
    1. download + extract OULAD from the UCI repository (skipped if present)
    2. data audit (sizes, '?' missing markers, duplicates, class balance)
    3. build a day-28 prediction snapshot (no information after day 27)
    4. compare 5 algorithms with identical preprocessing and seed
       (task metrics + training time + model size)
    5. ablation across ALL models: profile-only vs engagement-only vs both
    6. leakage / robustness checks (student-grouped CV, train-2013 -> test-2014)
    7. permutation importance and 5-fold CV for the winner

Usage:
    python src/run_experiment.py                      # downloads data to ./data
    python src/run_experiment.py --data-dir /path/to/oulad_csvs --skip-download

All outputs go to ./results (CSV tables) and ./figures (PNG charts).
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import platform
import time
import zipfile
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CUTOFF_DAY = 28  # prediction is made at the start of day 28 (end of week 4)
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip"
)
KEYS = ["code_module", "code_presentation", "id_student"]

CATEGORICAL = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability",
]
PROFILE_NUMERIC = ["num_of_prev_attempts", "studied_credits"]
ENGAGEMENT = [
    "total_clicks_wk1_4", "interaction_records_wk1_4",
    "active_days_wk1_4", "unique_sites_wk1_4",
]
COLORS = {"good": "#2E86AB", "risk": "#E76F51"}


# --------------------------------------------------------------------------
# 1. Data acquisition
# --------------------------------------------------------------------------
def download_oulad(data_dir: Path) -> None:
    if (data_dir / "studentVle.csv").exists():
        print(f"[data] OULAD already present in {data_dir}")
        return
    print(f"[data] downloading OULAD from {UCI_URL}")
    resp = requests.get(UCI_URL, timeout=600)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(data_dir)
    # some mirrors wrap the CSVs in an inner zip
    for inner in data_dir.glob("*.zip"):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(data_dir)


# --------------------------------------------------------------------------
# 2. Audit
# --------------------------------------------------------------------------
def audit(data_dir: Path, out: Path) -> dict[str, pd.DataFrame]:
    files = ["courses", "assessments", "studentInfo", "studentRegistration",
             "studentAssessment", "vle"]
    tables, rows = {}, []
    for name in files:
        raw = pd.read_csv(data_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        tables[name] = pd.read_csv(data_dir / f"{name}.csv", na_values=["?"])
        rows.append({
            "File": f"{name}.csv",
            "Rows": len(raw),
            "Columns": raw.shape[1],
            "Duplicate_Rows": int(raw.duplicated().sum()),
            # OULAD encodes missing values as '?', which pandas does NOT treat
            # as NaN by default -> a naive isna() audit reports 0 missing.
            "NaN_naive_audit": int(pd.read_csv(data_dir / f"{name}.csv").isna().sum().sum()),
            "Missing_as_?": int((raw == "?").sum().sum()),
        })
    vle_path = str(data_dir / "studentVle.csv")
    sv = duckdb.sql(f"""
        SELECT COUNT(*) AS n,
               COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM read_csv_auto('{vle_path}'))) AS dups
        FROM read_csv_auto('{vle_path}')""").fetchone()
    rows.append({"File": "studentVle.csv", "Rows": sv[0], "Columns": 6,
                 "Duplicate_Rows": sv[1], "NaN_naive_audit": 0, "Missing_as_?": 0})
    df = pd.DataFrame(rows)
    df.to_csv(out / "01_audit.csv", index=False)
    print(df.to_string(index=False))

    miss_cols = []
    for name, t in tables.items():
        for c, n in t.isna().sum().items():
            if n:
                miss_cols.append({"File": name, "Column": c, "Missing": int(n)})
    pd.DataFrame(miss_cols).to_csv(out / "01_missing_by_column.csv", index=False)
    print(pd.DataFrame(miss_cols).to_string(index=False))
    return tables


# --------------------------------------------------------------------------
# 3. Day-28 snapshot
# --------------------------------------------------------------------------
def build_dataset(tables, data_dir: Path, out: Path) -> pd.DataFrame:
    info, reg = tables["studentInfo"], tables["studentRegistration"]
    data = info.merge(reg, on=KEYS, how="left", validate="one_to_one")

    # keep students still enrolled when the prediction is made
    eligible = data[data["date_unregistration"].isna()
                    | (data["date_unregistration"] > CUTOFF_DAY)].copy()
    eligible["target_at_risk"] = eligible["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    vle_path = str(data_dir / "studentVle.csv")
    eng = duckdb.sql(f"""
        SELECT code_module, code_presentation, id_student,
               SUM(sum_click)          AS total_clicks_wk1_4,
               COUNT(*)                AS interaction_records_wk1_4,
               COUNT(DISTINCT date)    AS active_days_wk1_4,
               COUNT(DISTINCT id_site) AS unique_sites_wk1_4
        FROM read_csv_auto('{vle_path}')
        WHERE date BETWEEN 0 AND {CUTOFF_DAY - 1}
        GROUP BY 1, 2, 3""").df()

    model_data = eligible.merge(eng, on=KEYS, how="left", validate="one_to_one")
    model_data[ENGAGEMENT] = model_data[ENGAGEMENT].fillna(0)
    model_data = model_data.reset_index(drop=True)

    summary = {
        "registrations": len(data),
        "withdrew_before_day28_excluded": len(data) - len(eligible),
        "eligible_at_day28": len(model_data),
        "at_risk": int(model_data.target_at_risk.sum()),
        "at_risk_pct": round(100 * model_data.target_at_risk.mean(), 2),
        "no_vle_activity_wk1_4": int((model_data.total_clicks_wk1_4 == 0).sum()),
        "unique_students": int(model_data.id_student.nunique()),
        "students_with_multiple_registrations": int(
            (model_data.id_student.value_counts() > 1).sum()),
        "final_result_counts": model_data.final_result.value_counts().to_dict(),
    }
    (out / "02_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    means = (model_data.groupby("target_at_risk")[ENGAGEMENT].mean()
             .rename(index={0: "Successful", 1: "At Risk"}))
    means.round(2).to_csv(out / "02_engagement_by_outcome.csv")
    return model_data


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def make_preprocessor(categorical, numeric):
    parts = []
    if categorical:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if numeric:
        parts.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    return ColumnTransformer(parts)


def make_models():
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", random_state=SEED),
    }


def count_parameters(model) -> int:
    """Learned parameters: weights for linear models, nodes for trees."""
    if isinstance(model, DummyClassifier):
        return len(np.atleast_1d(model.class_prior_))
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    if isinstance(model, DecisionTreeClassifier):
        return int(model.tree_.node_count)
    if isinstance(model, RandomForestClassifier):
        return int(sum(e.tree_.node_count for e in model.estimators_))
    if isinstance(model, HistGradientBoostingClassifier):
        return int(sum(len(p.nodes) for it in model._predictors for p in it))
    return -1


def scores(y_true, pred, proba):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
    }


def fit_and_score(X_train, X_test, y_train, y_test, categorical, numeric, repeats=3):
    rows, fitted = [], {}
    for name, model in make_models().items():
        times = []
        for _ in range(repeats):
            pipe = Pipeline([("pre", make_preprocessor(categorical, numeric)),
                             ("model", model)])
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            times.append(time.perf_counter() - t0)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append({"Model": name, **scores(y_test, pred, proba),
                     "Train time (s)": float(np.median(times)),
                     "Parameters / nodes": count_parameters(pipe.named_steps["model"]),
                     "Size (KB)": len(pickle.dumps(pipe.named_steps["model"])) / 1024})
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    data_dir, out, figs = map(Path, (args.data_dir, args.results_dir, args.figures_dir))
    for d in (data_dir, out, figs):
        d.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_oulad(data_dir)

    env = {"python": platform.python_version(), "sklearn": sklearn.__version__,
           "pandas": pd.__version__, "duckdb": duckdb.__version__,
           "machine": platform.machine(), "processor": platform.processor(),
           "seed": SEED}
    (out / "00_environment.json").write_text(json.dumps(env, indent=2))

    print("\n=== 2. Audit ===")
    tables = audit(data_dir, out)

    print("\n=== 3. Day-28 dataset ===")
    md = build_dataset(tables, data_dir, out)
    all_numeric = PROFILE_NUMERIC + ENGAGEMENT
    X = md[CATEGORICAL + all_numeric]
    y = md["target_at_risk"]
    idx_train, idx_test = train_test_split(
        md.index, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    split = {"train": len(idx_train), "test": len(idx_test),
             "train_at_risk_pct": round(100 * y_train.mean(), 2),
             "test_at_risk_pct": round(100 * y_test.mean(), 2),
             "students_in_both_train_and_test": int(len(
                 set(md.loc[idx_train, "id_student"]) & set(md.loc[idx_test, "id_student"])))}
    (out / "03_split.json").write_text(json.dumps(split, indent=2))
    print(split)

    print("\n=== 4. Model comparison (full features) ===")
    comp, fitted = fit_and_score(X_train, X_test, y_train, y_test, CATEGORICAL, all_numeric)
    comp = comp.sort_values("F1", ascending=False).reset_index(drop=True)
    comp.to_csv(out / "04_model_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))

    # soft-voting ensemble (as in the original notebook)
    ens_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    ens_proba = np.mean([fitted[n].predict_proba(X_test)[:, 1] for n in ens_names], axis=0)
    ens = pd.DataFrame([{"Model": "Soft Voting Ensemble (LR+RF+GB)",
                         **scores(y_test, (ens_proba >= 0.5).astype(int), ens_proba)}])
    ens.to_csv(out / "04_ensemble.csv", index=False)
    print(ens.round(3).to_string(index=False))

    # confusion matrix for winner
    ConfusionMatrixDisplay.from_predictions(
        y_test, fitted["Logistic Regression"].predict(X_test),
        display_labels=["Successful", "At Risk"], cmap="Blues", values_format="d")
    plt.title("Logistic Regression - test set")
    plt.tight_layout(); plt.savefig(figs / "confusion_matrix_lr.png", dpi=200); plt.close()

    print("\n=== 5. Explanation + ablations ===")
    # Data property behind the explanation: risk falls smoothly and monotonically
    # with early engagement (a dose-response), with no threshold or U-shape.
    bins = pd.cut(md["active_days_wk1_4"], [-1, 0, 3, 6, 9, 12, 15, 18, 21, 24, 28])
    shape = (md.groupby(bins, observed=True)["target_at_risk"].agg(["mean", "size"])
             .rename(columns={"mean": "at_risk_rate", "size": "students"}))
    shape.to_csv(out / "05_risk_by_active_days.csv")
    print(shape)

    # Ablation A: remove / isolate the monotone engagement signal, every model,
    # 5-fold stratified CV (single-split differences are inside the noise).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    feature_sets = {
        "Profile only": (CATEGORICAL, PROFILE_NUMERIC),
        "Engagement only": ([], ENGAGEMENT),
        "Profile + engagement": (CATEGORICAL, all_numeric),
    }
    abl_rows = []
    for fs_name, (cats, nums) in feature_sets.items():
        for name, model in make_models().items():
            pipe = Pipeline([("pre", make_preprocessor(cats, nums)), ("model", model)])
            cvr = cross_validate(pipe, md[cats + nums], y, cv=cv, n_jobs=1,
                                 scoring=["f1", "roc_auc"], return_train_score=True)
            abl_rows.append({"Feature set": fs_name, "Model": name,
                             "F1 mean": cvr["test_f1"].mean(), "F1 SD": cvr["test_f1"].std(),
                             "AUC mean": cvr["test_roc_auc"].mean(),
                             "AUC SD": cvr["test_roc_auc"].std(),
                             "Train AUC": cvr["train_roc_auc"].mean()})
    abl = pd.DataFrame(abl_rows)
    abl["Rank by F1"] = abl.groupby("Feature set")["F1 mean"].rank(ascending=False).astype(int)
    abl.to_csv(out / "05a_ablation_feature_sets_cv.csv", index=False)
    print(abl.round(4).to_string(index=False))

    # Ablation B: give Gradient Boosting the data property. If the signal really is
    # monotone, forcing GB to be monotone-decreasing in engagement should help it
    # (less variance) and close the gap to Logistic Regression.
    pre_named = make_preprocessor(CATEGORICAL, all_numeric).set_output(transform="pandas")
    mono = {f"num__{f}": -1 for f in ENGAGEMENT}
    gb_variants = {
        "Gradient Boosting (unconstrained)": make_models()["Gradient Boosting"],
        "Gradient Boosting (monotone in engagement)": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, class_weight="balanced",
            random_state=SEED, monotonic_cst=mono),
        "Logistic Regression": make_models()["Logistic Regression"],
    }
    mono_rows = []
    for name, model in gb_variants.items():
        pipe = Pipeline([("pre", pre_named), ("model", model)])
        cvr = cross_validate(pipe, X, y, cv=cv, n_jobs=1, scoring=["f1", "roc_auc"],
                             return_train_score=True)
        mono_rows.append({"Model": name, "F1 mean": cvr["test_f1"].mean(),
                          "F1 SD": cvr["test_f1"].std(),
                          "AUC mean": cvr["test_roc_auc"].mean(),
                          "AUC SD": cvr["test_roc_auc"].std(),
                          "Train AUC": cvr["train_roc_auc"].mean(),
                          "Overfit gap (train-val AUC)":
                              cvr["train_roc_auc"].mean() - cvr["test_roc_auc"].mean()})
    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(out / "05b_ablation_monotone_gb_cv.csv", index=False)
    print(mono_df.round(4).to_string(index=False))

    # Supporting evidence: learning curve on the fixed test set. A simple signal
    # should be learned from few rows by LR; flexible models need more data.
    lc_rows = []
    for frac in [0.05, 0.10, 0.25, 0.50, 1.00]:
        reps = 1 if frac == 1.0 else 3
        for name in ["Logistic Regression", "Gradient Boosting", "Random Forest"]:
            f1s, aucs = [], []
            for rep in range(reps):
                sub = idx_train if frac == 1.0 else pd.Index(
                    np.random.RandomState(rep).choice(idx_train, int(frac * len(idx_train)),
                                                      replace=False))
                pipe = Pipeline([("pre", make_preprocessor(CATEGORICAL, all_numeric)),
                                 ("model", make_models()[name])]).fit(X.loc[sub], y.loc[sub])
                p = pipe.predict_proba(X_test)[:, 1]
                f1s.append(f1_score(y_test, (p >= 0.5).astype(int)))
                aucs.append(roc_auc_score(y_test, p))
            lc_rows.append({"Train fraction": frac, "Train rows": int(frac * len(idx_train)),
                            "Model": name, "F1": np.mean(f1s), "ROC-AUC": np.mean(aucs)})
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(out / "05c_learning_curve.csv", index=False)
    print(lc.round(3).to_string(index=False))

    print("\n=== 6. Leakage / robustness checks (Logistic Regression + Gradient Boosting) ===")
    rob = []
    groups = md["id_student"]
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in ["Logistic Regression", "Gradient Boosting"]:
        pipe = Pipeline([("pre", make_preprocessor(CATEGORICAL, all_numeric)),
                         ("model", make_models()[name])])
        cvr = cross_validate(pipe, X, y, cv=sgk, groups=groups,
                             scoring=["f1", "roc_auc"], n_jobs=1)
        rob.append({"Check": "5-fold CV grouped by student", "Model": name,
                    "F1": cvr["test_f1"].mean(), "ROC-AUC": cvr["test_roc_auc"].mean()})
    # temporal: train on 2013 presentations, test on 2014 presentations
    tr = md["code_presentation"].str.startswith("2013")
    Xt = md[CATEGORICAL + all_numeric]
    for name in ["Logistic Regression", "Gradient Boosting"]:
        pipe = Pipeline([("pre", make_preprocessor(CATEGORICAL, all_numeric)),
                         ("model", make_models()[name])])
        pipe.fit(Xt[tr], y[tr])
        p = pipe.predict_proba(Xt[~tr])[:, 1]
        rob.append({"Check": "Train 2013 -> test 2014", "Model": name,
                    "F1": f1_score(y[~tr], (p >= 0.5).astype(int)),
                    "ROC-AUC": roc_auc_score(y[~tr], p)})
    rob = pd.DataFrame(rob)
    rob.to_csv(out / "06_robustness.csv", index=False)
    print(rob.round(3).to_string(index=False))

    print("\n=== 7. Winner: permutation importance + 5-fold CV ===")
    imp = permutation_importance(fitted["Logistic Regression"], X_test, y_test,
                                 scoring="f1", n_repeats=5, random_state=SEED, n_jobs=1)
    imp_df = (pd.DataFrame({"Feature": X_test.columns, "Importance": imp.importances_mean,
                            "SD": imp.importances_std})
              .sort_values("Importance", ascending=False))
    imp_df.to_csv(out / "07_permutation_importance.csv", index=False)
    print(imp_df.round(4).to_string(index=False))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name in ["Logistic Regression", "Gradient Boosting", "Random Forest", "Decision Tree"]:
        pipe = Pipeline([("pre", make_preprocessor(CATEGORICAL, all_numeric)),
                         ("model", make_models()[name])])
        cvr = cross_validate(pipe, X, y, cv=cv, return_train_score=True, n_jobs=1,
                             scoring=["balanced_accuracy", "precision", "recall", "f1", "roc_auc"])
        for m in ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
            cv_rows.append({"Model": name, "Metric": m,
                            "Train mean": cvr[f"train_{m}"].mean(),
                            "Validation mean": cvr[f"test_{m}"].mean(),
                            "Validation SD": cvr[f"test_{m}"].std()})
    cvdf = pd.DataFrame(cv_rows)
    cvdf.to_csv(out / "07_cross_validation.csv", index=False)
    print(cvdf.round(3).to_string(index=False))

    # figures -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    means = pd.read_csv(out / "02_engagement_by_outcome.csv", index_col=0)
    for f, t, ax in zip(ENGAGEMENT, ["Total clicks", "Interaction records", "Active days",
                                     "Unique learning sites"], axes.flatten()):
        means[f].plot(kind="bar", ax=ax, color=[COLORS["good"], COLORS["risk"]])
        ax.set_title(t); ax.set_xlabel(""); ax.set_ylabel("Mean, weeks 1-4")
        ax.tick_params(axis="x", rotation=0)
    plt.suptitle("Early VLE engagement: successful vs at-risk students")
    plt.tight_layout(); plt.savefig(figs / "engagement_by_outcome.png", dpi=200); plt.close()

    import make_figures  # same folder
    make_figures.risk_by_active_days(out, figs)
    make_figures.model_comparison(out, figs)
    make_figures.ablation(out, figs)

    print("\nDone. Tables in", out.resolve(), "| figures in", figs.resolve())


if __name__ == "__main__":
    main()
