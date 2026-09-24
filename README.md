# At-Risk Student Prediction

A machine-learning project that predicts whether a student will fail or withdraw from a course using background information and engagement during the first four weeks of study.

The project uses the **Open University Learning Analytics Dataset (OULAD)** and compares six classification methods. It was developed for **CSBP711 Advanced Artificial Intelligence**.

## Research question

> How accurately can standard machine-learning algorithms identify students at risk of failing or withdrawing based on their background and early learning engagement?

## Overview

The code prepares a dataset at the student–module–presentation level, trains several models under the same evaluation conditions, and compares their predictive performance, training time, and saved model size.

It also runs an **ablation study**: all models are trained again after removing early engagement features. This tests whether those features improve prediction and change the model ranking.

This README describes the runnable `run_experiment.py` and `verify_results.py` files supplied in the assignment package. The pipeline builds on the original `01_data_audit.py` and its [source notebook](https://github.com/dalkhaseeb-D/at-risk-student-prediction/blob/main/notebooks/01_data_audit.ipynb).

## Project files

| File or directory | Purpose |
| --- | --- |
| `run_experiment.py` | Downloads or loads the data, audits it, creates features, trains models, evaluates performance, and runs the ablation. |
| `verify_results.py` | Checks saved metrics and student separation, then estimates uncertainty in the leading models' F1 difference. |
| `requirements.txt` | Lists the tested Python package versions. |
| `data/` | Stores downloaded or manually supplied OULAD files. Created when needed. |
| `results/` | Stores generated datasets, evaluation tables, audit information, and predictions. |
| `results/models/` | Stores the fitted preprocessing and classification pipelines. Created by the experiment. |
| `docs/RESULTS.md` | Explains the findings, data handling, ablation, and limitations. |
| `docs/feature_dictionary.md` | Defines each predictor and its source. |
| `docs/REQUIREMENTS_CHECKLIST.md` | Maps the assignment requirements to the deliverables. |
| `CSBP711_Presentation.pptx` | Contains the five-slide assignment presentation. |

## Installation

Use **Python 3.12**. The recorded experiment used Python 3.12.14.

Open a terminal in the project directory containing `run_experiment.py` and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses pandas and NumPy for data processing, scikit-learn for modelling, joblib for saving pipelines, and threadpoolctl for controlling computation threads. Exact versions are pinned in `requirements.txt`.

The VLE CSV is approximately 454 MB. The current implementation reads and audits the complete table, so allow several GB of available RAM.

## Run the project

### Download the data and run the experiment

```bash
python run_experiment.py --data-dir data --output-dir results --download
```

This command downloads the OULAD archive if the required files are unavailable, prepares the data, and runs both the full-feature and profile-only comparisons.

### Use data already downloaded

Place these three CSV files in `data/`:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`

Then run:

```bash
python run_experiment.py --data-dir data --output-dir results
```

If automatic downloading is blocked, download the archive from the [UCI dataset page](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) and extract the three files manually.

### Verify the outputs

After the experiment finishes:

```bash
python verify_results.py --results results
```

Run the experiment before verification. The verification script needs the individual predictions and split assignments generated locally; the distributed assignment package contains aggregate results only.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `data` | Directory containing the three source CSV files. |
| `--output-dir` | `results` | Directory where the experiment writes its outputs. |
| `--download` | Off | Allows downloading the source archive when required files are missing. |
| `--results` | `results` | Output directory to check when running `verify_results.py`. |

Use a different output directory to preserve an earlier run:

```bash
python run_experiment.py --data-dir data --output-dir results_new
python verify_results.py --results results_new
```

## Dataset

OULAD contains anonymised records from selected Open University modules taught in 2013 and 2014. This project uses three source tables:

| Source file | Information used |
| --- | --- |
| `studentInfo.csv` | Student background, course context, and final outcome. |
| `studentRegistration.csv` | Registration and withdrawal dates for determining eligibility at the cutoff. |
| `studentVle.csv` | Dated records of interaction with learning resources. |

Tables are joined using `code_module`, `code_presentation`, and `id_student`.

The source contains **32,593 enrolments**, representing **28,785 distinct students**. A student can have more than one enrolment, so enrolment rows are not interchangeable with unique student counts.

### Prediction time and target

The model screens students **at the end of day 28**, using VLE activity from **days 0–27 inclusive**. This represents a one-day logging lag.

An enrolment is eligible when its registration date is known and no later than day 28, and the student has not withdrawn on or before day 28.

| Target value | Meaning | Original outcomes |
| --- | --- | --- |
| `1` | At risk | `Fail`, `Withdrawn` |
| `0` | Successful | `Pass`, `Distinction` |

The resulting cohort contains **27,515 enrolments**, of which **44.11%** are at risk. Compared with the original script's 27,538-row cohort, the runnable pipeline additionally excludes 16 late registrations and 7 unknown registration dates.

### Input features

The full model uses **14 predictors**:

| Group | Features |
| --- | --- |
| Course context | `code_module`, `code_presentation` |
| Student background | `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability` |
| Study history and workload | `num_of_prev_attempts`, `studied_credits` |
| Early engagement | The four measures below |

All early engagement measures are calculated separately for each enrolment over days 0–27:

| Derived feature | Calculation |
| --- | --- |
| `total_clicks_wk1_4` | Sum of `sum_click`. |
| `interaction_records_wk1_4` | Number of log rows. This is not the number of sessions or clicks. |
| `active_days_wk1_4` | Number of distinct dates with a logged interaction, from 0 to 28. |
| `unique_sites_wk1_4` | Number of distinct learning resource IDs (`id_site`). |

An eligible enrolment with no early VLE records receives zero for all four measures. These are summaries of observed data; no synthetic predictors are added. The extra `risk_factor` column in the supplied student information file is ignored.

## How the code works

### 1. Load and audit the data

`ensure_data()` locates or downloads the input files. `prepare()` checks required columns, counts missing values and repeated rows, records file hashes, and verifies the student-table joins.

Repeated VLE rows are reported and retained because the source has no event identifier that establishes whether they are erroneous duplicates. This choice affects click and record totals, but not distinct-day or distinct-site counts.

### 2. Build the modelling dataset

`prepare()` applies the day-28 eligibility rules, creates the binary target, aggregates the four engagement measures, and saves `model_dataset.csv`.

### 3. Split students into training and test sets

`run()` uses the first split from `StratifiedGroupKFold` with five folds, shuffling, and seed `42`. Grouping by `id_student` keeps every enrolment belonging to the same student in one partition.

| Partition | Enrolments | Unique students | At risk |
| --- | ---: | ---: | ---: |
| Training | 22,011 | 19,850 | 44.11% |
| Test | 5,504 | 4,976 | 44.11% |

No student appears in both sets. Although the splitter is configured with five folds, this code uses **one approximately 80/20 holdout**, not five-fold cross-validation.

### 4. Preprocess the predictors

`preprocess()` creates a fresh `ColumnTransformer` for each model:

- Categorical features: most-frequent-value imputation followed by one-hot encoding. Unknown categories are ignored at prediction time.
- Numeric features: median imputation followed by standard scaling.

All preprocessing is fitted using training data only. In the recorded full-feature run, the 14 predictors become 52 encoded columns.

Student IDs, final outcomes, registration dates, and withdrawal dates are excluded from the predictors. IDs support joins and grouping; dates determine eligibility; final outcomes define the target.

### 5. Train and evaluate the models

`estimators()` defines the following methods:

| Method | Main settings |
| --- | --- |
| Dummy baseline | Predicts the most frequent training class. |
| Logistic regression | Balanced class weights, maximum 1,000 iterations. |
| Decision tree | Maximum depth 8, minimum 20 samples per leaf. |
| Random forest | 200 trees, maximum depth 12, minimum 5 samples per leaf. |
| Histogram gradient boosting | 150 iterations, learning rate 0.08, maximum 31 leaves, early stopping disabled. |
| Soft voting ensemble | Equal-weight probability averaging of logistic regression, random forest, and gradient boosting. |

The learned constituent models use balanced class weights. Every method receives the same split and preprocessing rules. Randomised components use seed `42`, and fits run sequentially with one computation thread.

`scores()` calculates accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC. **F1 for the at-risk class is the primary comparison metric.** The learned models use a fixed 0.5 probability threshold, with no test-driven threshold or hyperparameter tuning.

Training time covers preprocessing and model fitting. Model size is the uncompressed saved joblib pipeline, including preprocessing, measured in KiB.

### 6. Run the ablation and interpretation

`run()` repeats all six fits using only the ten background and course features. It saves the change in F1 and rank for each method. The hypothesis and expected direction are recorded before model fitting.

The script also calculates training engagement summaries and permutation importance for the full-feature F1 leader. It reloads the saved winning pipeline and checks that its predictions match the recorded predictions.

### 7. Verify the saved results

`verify_results.py` independently recalculates F1, accuracy, and ROC-AUC from the saved predictions, confirms that students do not cross the split, and runs 1,000 paired bootstrap resamples by student to estimate the F1 gap between the top two full-feature methods.

## Results

The following values come from the experiment run on **24 September 2026**:

| Model | Accuracy | Recall | F1 | ROC-AUC | Training seconds | Size KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.691 | 0.699 | **0.666** | 0.754 | 0.104 | 7.01 |
| Soft voting ensemble | 0.691 | 0.667 | 0.655 | **0.757** | 4.038 | 19,735.63 |
| Random forest | **0.692** | 0.644 | 0.648 | 0.749 | 2.716 | 19,189.12 |
| Gradient boosting | 0.686 | 0.652 | 0.647 | 0.750 | 1.325 | 547.50 |
| Decision tree | 0.658 | 0.626 | 0.618 | 0.720 | 0.127 | 34.18 |
| Dummy baseline | 0.559 | 0.000 | 0.000 | 0.500 | 0.060 | 6.31 |

Logistic regression has the highest observed F1. The ensemble leads on ROC-AUC, while random forest has the highest accuracy. The preferred method therefore depends on the evaluation objective. Timings are measurements from one local run and will vary by hardware.

### Effect of early engagement

Removing engagement reduced logistic regression's F1 from **0.6659 to 0.5806** and moved it from first to second place. Its F1 advantage over random forest shrank from **0.0176 to 0.0079**.

In the training data, the at-risk proportion declines across active-day bands, from 81.3% for zero active days to 20.6% for 22–28 active days. This pattern is consistent with engagement providing useful information that an additive model can capture. The ablation supports the value of these predictors, but does not establish causality or prove that a linear model will always perform best.

## Generated outputs

| Output under `results/` | Description |
| --- | --- |
| `data_audit.json` | Source dimensions, missing values, repeated rows, hashes, and cohort exclusions. |
| `model_dataset.csv` | Derived enrolment-level modelling dataset. |
| `split_summary.json`, `split_assignments.csv` | Split statistics and individual train/test membership. |
| `model_comparison.csv` | Full-feature performance, fit time, size, and confusion matrix counts. |
| `profile_comparison.csv` | Results after removing engagement. |
| `all_results.csv` | All 12 fits across both feature sets. |
| `ablation_comparison.csv` | F1 changes and rankings with and without engagement. |
| `ablation_hypothesis.json`, `ablation_test.json` | Expected ablation direction and observed outcome. |
| `test_predictions.csv` | Individual test predictions and probabilities. |
| `training_engagement_*.csv`, `training_risk_by_active_days.csv` | Training-data summaries used to interpret the results. |
| `feature_sources.csv`, `winner_permutation_importance.csv` | Feature provenance and permutation importance. |
| `models/*.joblib` | Saved fitted pipelines. |
| `run_metadata.json` | Run date, package versions, seed, platform, and measurement definitions. |
| `winner_gap_bootstrap.json` | Uncertainty estimate created by `verify_results.py`. |

Raw data, individual predictions, split assignments, and fitted models are excluded from version control by the supplied `.gitignore`. The experiment recreates them locally.

## Reproducibility and limitations

The recorded run uses previously supplied OULAD CSVs accessed on 24 September 2026. Their original download date is unknown; exact hashes are in `results/data_audit.json`. The automatic download path writes download metadata alongside the source files.

Use the same source data, pinned package versions, feature definitions, and seed to reproduce the comparison. The results apply to students still enrolled at the day-28 cutoff in this historical dataset. They do not cover earlier withdrawals or demonstrate performance on future cohorts. Background and workload values are assumed available at the cutoff, but the published snapshot does not provide a change history to verify that assumption.

The analysis uses one holdout split. The bootstrap interval is conditional on the already-trained models and does not include uncertainty from training or model selection. Future work can examine temporal validation, subgroup performance, calibration, and sensitivity to repeated VLE rows.

## Contributions and AI assistance

The existing repository history attributes the original data audit, engagement analysis, model comparison, ensemble, ablation, and evaluation to **dalkhaseeb-D**. The group should add each member's full name and actual contribution before submission.

ChatGPT/Codex assisted with code revision, experiment execution, documentation, and presentation preparation. The reported comparison values come from executed experiments. The student or group is responsible for reviewing the code and interpreting the results.

## References and licence

- **Dataset:** Kuzilek, J., Hlosta, M., and Zdrahal, Z. [Open University Learning Analytics Dataset](https://doi.org/10.24432/C5KK69), UCI Machine Learning Repository. The dataset uses the **CC BY 4.0** licence.
- **Dataset paper:** Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017). [Open University Learning Analytics dataset](https://doi.org/10.1038/sdata.2017.171). *Scientific Data*, 4, 170171.
- **Implementation libraries:** [scikit-learn](https://scikit-learn.org/1.8/), [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/), [joblib](https://joblib.readthedocs.io/), and [threadpoolctl](https://github.com/joblib/threadpoolctl).

Code in the existing repository is covered by its **MIT licence**. The dataset has its own licence, stated above.

## Assignment checklist

- [x] Document the dataset, licence, features, class balance, and data quality issues.
- [x] Apply a student-separated split and training-only preprocessing.
- [x] Compare at least four methods, including a baseline.
- [x] Report metrics, training times, and model sizes.
- [x] Explain the findings and run an ablation that compares rankings.
- [x] Provide runnable code, pinned dependencies, and five slides.
- [x] Acknowledge source code, libraries, and AI assistance.
- [ ] Upload the completed code and documentation to the public GitHub repository.
- [ ] Confirm all group members, actual contributions, and individual commit activity.
- [ ] Confirm the dataset claim with the instructor and submit the required files.

See [the detailed requirements checklist](docs/REQUIREMENTS_CHECKLIST.md) for the full assignment mapping.
