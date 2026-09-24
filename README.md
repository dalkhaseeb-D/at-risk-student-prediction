# Early Prediction of At-risk Students (OULAD)

CSBP711 Advanced Artificial Intelligence, Fall 2026, Assignment 1: Datasets and Algorithm Comparison.

**Question.** At the end of week 4 of a course, can we tell which students will fail or withdraw, using only what the university already knows by then (enrolment profile + virtual learning environment (VLE) click logs)? Which algorithm does this best, and why?

**Short answer.** Logistic Regression wins (test F1 0.650, ROC-AUC 0.744; 5-fold CV F1 0.650 ± 0.007). It uses 53 parameters, trains in about 0.4 s, and is about 1 KB on disk. Gradient Boosting ranks cases just as well (AUC 0.744) but has a lower F1 (0.644 test, 0.635 CV), overfits (train AUC 0.79), and loses more when tested on a later year. **The reason is a property of the data**: risk falls with early engagement (81% at-risk with 0 active days → 19% with 25–28 active days). There is little non-linear structure for the flexible models to find, so their extra capacity mostly fits noise. one ablation test this (see [Results](#results)).

---

## Dataset

| | |
|---|---|
| Name | Open University Learning Analytics Dataset (OULAD) |
| Source URL | https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset |
| Original paper | Kuzilek J., Hlosta M., Zdrahal Z. (2017). *Open University Learning Analytics dataset.* Scientific Data 4, 170171. https://doi.org/10.1038/sdata.2017.171 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Version | UCI dataset id 349 (the only release; unchanged since 2017). Downloaded: **2026-09-24** |
| Collection | Anonymised records from The Open University (UK): 7 modules, 22 module-presentations (2013–2014), 32,593 student registrations, 10.6 M daily VLE click-summary rows. The Open University's Knowledge Media Institute collected it from its production systems and released it after anonymisation (IMD band and age were banded, some regions merged). |

## Reproduce

```bash
git clone https://github.com/dalkhaseeb-D/at-risk-student-prediction.git
cd at-risk-student-prediction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_experiment.py            # downloads OULAD from UCI into ./data, then runs everything
```

If you already have the CSVs: `python src/run_experiment.py --data-dir path/to/oulad --skip-download`.

The full run takes about 2 minutes on a 2-core laptop-class CPU. It writes every table to `results/` and every chart to `figures/`. `python src/make_figures.py` rebuilds only the charts. All randomness uses `random_state=42`. Environment details are logged to `results/00_environment.json`.

| Output | Content |
|---|---|
| `results/01_audit.csv`, `01_missing_by_column.csv` | size, duplicates, missing values per file |
| `results/02_dataset_summary.json` | final cohort, class balance, multi-registration students |
| `results/03_split.json` | train/test split sizes and balance |
| `results/04_model_comparison.csv` | **main table**: metrics + train time + parameters + size |
| `results/05_risk_by_active_days.csv` | the data property behind the explanation |
| `results/05a_ablation_feature_sets_cv.csv` | Ablation A (all models × 3 feature sets, 5-fold CV) |
| `results/05b_ablation_monotone_gb_cv.csv` | Ablation B (monotone-constrained Gradient Boosting) |
| `results/05c_learning_curve.csv` | supporting learning curve |
| `results/06_robustness.csv` | leakage checks: student-grouped CV and train-2013 → test-2014 |
| `results/07_*.csv` | permutation importance, 5-fold CV for the four real models |

## Method

**Prediction point.** Day 28 of the module (end of week 4).

**Cohort.** All registrations still enrolled at day 28. We drop students who unregistered on or before day 28 (5,055), because an early-warning system cannot act on students who have already left. That leaves 27,538 registrations.

**Target.** `at_risk = 1` if `final_result ∈ {Fail, Withdrawn}`, else 0 (Pass or Distinction). 44.1% are at risk, so the classes are moderately balanced.

**Features.** There are 14, all known at day 28.
- Profile (10): module, presentation, gender, region, highest education, IMD band, age band, disability, previous attempts, studied credits.
- Engagement, VLE days 0–27 only (4): total clicks, interaction records, active days, unique resources visited. Students with no activity get 0.

**Split.** Stratified 80/20 split, seed 42: 22,030 train and 5,508 test rows, 44.1% at risk in both.

**Preprocessing.** The same `ColumnTransformer` is used for every model, fitted on training data only:
- categorical: most-frequent imputation, then one-hot encoding (unknown categories ignored);
- numeric: median imputation, then standard scaling.

**Models.** Every model uses `random_state=42` and `class_weight="balanced"` where the model supports it:
- Dummy (most frequent class) as the baseline;
- Logistic Regression;
- Decision Tree (max depth 8, at least 20 samples per leaf);
- Random Forest (200 trees, max depth 12, at least 5 samples per leaf);
- Histogram Gradient Boosting (150 iterations, learning rate 0.08, 31 leaves).

A soft-voting ensemble (LR + RF + GB) is also reported.

**Metrics.** F1 on the at-risk class is the main metric. We also report ROC-AUC, balanced accuracy, precision, and recall.

## Data problems found

1. **Missing values are coded as `?`.** A naive `pandas.isna()` audit, which is what our first notebook did, reports **0 missing** in every file. Once `?` is read as missing:
   - `imd_band` is missing for 1,111 students (3.4%). It is imputed with the most frequent value inside the pipeline.
   - `date_unregistration` is missing for 22,521 students. This is not a data gap: it means the student never unregistered.
   - `date_registration` is missing for 45 students (not used as a feature).
   - `score` is missing for 173 assessment rows and `week_from`/`week_to` for 5,243 VLE rows (neither is used).

   The `date_unregistration` column also has to be converted to numeric before the day-28 filter. Otherwise the comparison fails on the `?` strings.
2. **Duplicates.** None in the six small tables. `studentVle.csv` has **787,170 exact duplicate rows** (7.4%). These are genuine repeated daily summaries for the same student, site, and day, not copy errors, so they are kept. `COUNT(DISTINCT date)` and `COUNT(DISTINCT id_site)` are unaffected by them.
3. **Same student in several registrations.** 2,578 students have more than one registration, and 844 of them end up with rows in both train and test. Re-running with CV grouped by student gives LR F1 0.647 vs 0.650 and AUC 0.743 vs 0.744, so no meaningful leakage.
4. **Leakage by timing.** VLE data is cut at day 27 and scores are excluded (see Method). Students who withdraw before day 28 are removed because their label is already known.
5. **Distribution shift over time.** Training on 2013 presentations and testing on 2014 lowers LR AUC to 0.723 (GB: 0.718). A deployed model should be re-trained each year.
6. 1,246 eligible students (4.5%) have **zero VLE activity** in weeks 1–4; 81% of them end up at risk.

## Results

### Main comparison (held-out test set, n = 5,508)

| Model | F1 | Recall | Precision | Bal. acc. | ROC-AUC | Train time (s) | Params / nodes | Size (KB) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.650** | **0.673** | 0.628 | 0.679 | 0.744 | 0.38 | 53 | 1.1 |
| Gradient Boosting | 0.644 | 0.645 | 0.642 | 0.681 | 0.744 | 0.60 | 4,148 | 247 |
| Random Forest | 0.630 | 0.617 | 0.643 | 0.673 | 0.739 | 1.45 | 241,842 | 18,957 |
| Decision Tree | 0.620 | 0.643 | 0.600 | 0.652 | 0.708 | 0.15 | 327 | 27 |
| Dummy baseline | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | 0.05 | 2 | 0.5 |
| *Soft-voting ensemble (LR+RF+GB)* | 0.644 | 0.649 | 0.638 | 0.679 | 0.747 | – | – | – |

Times were measured on a 2-core cloud CPU and will vary by machine.

With 5-fold CV the ranking is the same:

| Model | F1 | AUC | Train AUC |
|---|---|---|---|
| LR | 0.650 ± 0.007 | 0.743 | 0.745 |
| GB | 0.635 ± 0.008 | 0.743 | 0.794 |
| RF | 0.633 ± 0.009 | 0.740 | 0.825 |
| DT | 0.615 ± 0.011 | 0.715 | 0.751 |

### Why Logistic Regression wins: a property of the data

The at-risk rate falls **steadily and monotonically** as early activity rises. By active days in weeks 1–4, it goes 81% → 63% → 53% → 47% → 41% → 35% → 31% → 29% → 23% → 19% (`figures/risk_by_active_days.png`). There is no threshold, U-shape, or strong interaction. The signal is essentially "more regular early activity means lower risk", added on top of module-level base rates, and the labels are noisy because the outcome is decided weeks after day 28. That caps every model at about 0.74 AUC.

A linear model on scaled counts represents a monotone trend exactly, with 53 parameters and almost no variance (train AUC = validation AUC). The tree ensembles reach the same ranking quality (AUC) but use their extra capacity to fit noise: their train–validation AUC gap is 0.05–0.09. This shows up as a lower F1 at the operating threshold and a larger drop on the later year (2013 → 2014 F1: LR 0.637 vs GB 0.609).

### Ablations that test the explanation

**Remove the monotone engagement signal.**
*Prediction:* if LR wins *because of* the smooth engagement trend, its lead over GB should appear when engagement features are present and vanish without them.
*Result (5-fold CV, LR minus GB F1):* profile only **+0.1 pt** (0.578 vs 0.577, inside the noise); engagement only **+1.2 pt**; profile + engagement **+1.4 pt**. ✔

The constrained model moves up toward LR. 

**Supporting: learning curve.** With 5% of the training data (1,101 rows), LR already reaches F1 0.630 / AUC 0.720, while GB reaches 0.573 / 0.677. The gap decreases as data grows, which is what we expect when the true pattern is simple.

### Most important features 

<img width="800" height="500" alt="feature_importance" src="https://github.com/user-attachments/assets/8e22ab0b-5df7-4baf-add4-5b7b3b308331" />

## Libraries and references
- scikit-learn (Pedregosa et al., 2011, JMLR 12), pandas, NumPy, matplotlib, DuckDB (for aggregating the 10.6 M-row VLE file), requests.
- Kuzilek, Hlosta & Zdrahal (2017), OULAD, *Scientific Data* 4:170171.
- `data-edu/oulad` R package (mirror used when UCI was unreachable): https://github.com/data-edu/oulad


## Contributions

| Member | Contribution |
|---|---|
| Member 1 (Duaa Alkhaseeb) | Dataset selection, data download and audit notebook, ML-classifiers Applying, Feature extraction
| Member 2 (Arwa Al-shannaq)| Data Preprocessing, Comparison table of timing/size measurement, README, slides, reproducibility testing, Ablation analysis, figures creation


## Licence
Code: MIT (see `LICENSE`). Data: OULAD, CC BY 4.0. The data is **not** redistributed in this repository; the script downloads it from UCI.
