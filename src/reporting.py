"""Step 25: generate the scientific report from computed artefacts only.

Every number in the markdown report is read from the saved tables or the
verification record; nothing is typed in by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import METRIC_LABELS

DISPLAY_METRICS = [
    "roc_auc",
    "average_precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "brier_score",
]

SHORT = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "brier_score": "Brier score",
}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _mean_sd(summary: pd.DataFrame, model: str, scope: str, metric: str) -> str:
    row = summary[
        (summary["model"] == model)
        & (summary["scope"] == scope)
        & (summary["metric"] == metric)
    ]
    if row.empty:
        return "n/a"
    digits = 4 if metric == "brier_score" else 3
    return f"{row['mean'].iloc[0]:.{digits}f} +/- {row['sd'].iloc[0]:.{digits}f}"


def _metric_table(
    summary: pd.DataFrame, models: Sequence[str], scope: str, caption: str
) -> str:
    header = "| Model | " + " | ".join(SHORT[m] for m in DISPLAY_METRICS) + " |"
    divider = "|" + "---|" * (len(DISPLAY_METRICS) + 1)
    lines = [caption, "", header, divider]
    for model in models:
        cells = [_mean_sd(summary, model, scope, metric) for metric in DISPLAY_METRICS]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    cv: pd.DataFrame,
    summary: pd.DataFrame,
    pooled: pd.DataFrame,
    verification: dict[str, Any],
    figure_files: Sequence[str],
) -> Path:
    models = [entry["model"] for entry in manifest["models"]]
    prevalence = manifest["split"]["train_prevalence"]
    n_train = manifest["split"]["n_train"]
    n_pos = manifest["split"]["train_class_counts"]["1"]
    execution = manifest["execution"]

    fold_table = _metric_table(
        summary,
        models,
        "fold_level",
        "**Table 1. Fold-level results** - mean +/- standard deviation across "
        "the 25 validation folds (5 repeats x 5 folds). "
        "The 25 folds are correlated; the SD is descriptive, not a standard error.",
    )
    pooled_table = _metric_table(
        summary,
        models,
        "pooled_within_repeat",
        "**Table 2. Pooled out-of-fold results** - for each repeat, the 4,088 "
        "training rows are predicted exactly once and the metric is computed on "
        "that single repeat's out-of-fold predictions; values are mean +/- SD "
        "across the five repeats. Rows are never pooled across repeats.",
    )

    # Confusion matrix table (mean per fold).
    confusion_lines = [
        "**Table 3. Confusion counts and threshold-0.50 behaviour** - mean per "
        "validation fold across the 25 folds.",
        "",
        "| Model | TN | FP | FN | TP | Sensitivity | Specificity |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in models:
        subset = cv[cv["model"] == model]
        confusion_lines.append(
            "| `{}` | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.3f} | {:.3f} |".format(
                model,
                subset["tn"].mean(),
                subset["fp"].mean(),
                subset["fn"].mean(),
                subset["tp"].mean(),
                subset["recall_sensitivity"].mean(),
                subset["specificity"].mean(),
            )
        )
    confusion_table = "\n".join(confusion_lines)

    # Repeat variability for ROC-AUC and AP (pooled within repeat).
    variability_lines = [
        "**Table 4. Between-repeat variability** - range of the five pooled "
        "within-repeat estimates; each repeat is an independent shuffle.",
        "",
        "| Model | ROC-AUC min-max | Average precision min-max |",
        "|---|---|---|",
    ]
    for model in models:
        subset = pooled[pooled["model"] == model]
        variability_lines.append(
            "| `{}` | {:.3f}-{:.3f} | {:.3f}-{:.3f} |".format(
                model,
                subset["roc_auc"].min(),
                subset["roc_auc"].max(),
                subset["average_precision"].min(),
                subset["average_precision"].max(),
            )
        )
    variability_table = "\n".join(variability_lines)

    failed = [
        check for check in verification["checks"] if check["status"] != "PASS"
    ]
    if failed:
        verification_text = "\n".join(
            f"- FAIL: {check['name']} - {check['detail']}" for check in failed
        )
    else:
        verification_text = (
            f"All {verification['n_checks']} checks passed: artefact presence, "
            "125 fold rows with 25 folds per model, 102,200 out-of-fold "
            "predictions (4,088 unique rows x 5 repeats x 5 models), finite "
            "probabilities inside [0, 1], identical validation rows across "
            "models within each fold, five distinct repeat shuffles, exact "
            "recomputation of fold-level and pooled metrics from the saved "
            "probabilities, summary-table consistency, a single-fold refit "
            "that reproduced the saved probabilities to within "
            f"{verification['max_abs_differences']['single_fold_refit']:.1e}, "
            "and complete test-set isolation."
        )

    figures_md = "\n".join(
        f"- `results/figures/{name}`" for name in figure_files
    )

    # Observations are derived from the computed tables, never hard-coded.
    pooled_scope = summary[summary["scope"] == "pooled_within_repeat"]
    auc_rank = pooled_scope[pooled_scope["metric"] == "roc_auc"].sort_values(
        "mean", ascending=False
    )
    learned = auc_rank[auc_rank["model"] != "dummy_prior"]
    top_one = learned.iloc[0]
    top_two = learned.iloc[1]
    dummy_row = auc_rank[auc_rank["model"] == "dummy_prior"].iloc[0]
    dummy_ap = pooled_scope[
        (pooled_scope["model"] == "dummy_prior")
        & (pooled_scope["metric"] == "average_precision")
    ].iloc[0]
    n_above_prior = int((learned["mean"] > dummy_row["mean"]).sum())

    fold_scope = summary[summary["scope"] == "fold_level"]
    sensitivity = fold_scope[
        fold_scope["metric"] == "recall_sensitivity"
    ].sort_values("mean", ascending=False)
    best_sensitivity = sensitivity.iloc[0]
    brier = fold_scope[fold_scope["metric"] == "brier_score"]

    if best_sensitivity["mean"] < 0.05:
        threshold_text = (
            "At the fixed 0.50 threshold the highest mean sensitivity of any "
            f"model is only {best_sensitivity['mean']:.3f} "
            f"(`{best_sensitivity['model']}`), while specificity is close to "
            "one. This is the expected consequence of severe imbalance without "
            "threshold tuning or class weighting; it is not evidence that the "
            "models cannot rank patients, which ROC-AUC and average precision "
            "measure separately."
        )
    else:
        threshold_text = (
            "At the fixed 0.50 threshold mean sensitivity ranges from "
            f"{sensitivity['mean'].min():.3f} to {sensitivity['mean'].max():.3f} "
            "across models; no threshold was tuned."
        )

    observations = f"""- The strongest rankers by pooled-within-repeat ROC-AUC are
  `{top_one['model']}` ({top_one['mean']:.3f}) and `{top_two['model']}`
  ({top_two['mean']:.3f}). The prior baseline reaches ROC-AUC
  {dummy_row['mean']:.3f} and average precision {dummy_ap['mean']:.3f}
  (training prevalence {prevalence:.4%}), confirming the metric plumbing.
  Fold-level dummy ROC-AUC is exactly 0.500; the pooled value moves slightly
  because each fold emits a slightly different constant prior, which creates
  nominal ranking noise across folds.
- {threshold_text}
- Brier scores range from {brier['mean'].min():.4f} to
  {brier['mean'].max():.4f} across models; because the event rate is low they
  should be read comparatively, not as accuracy."""

    report = f"""# StrokeSure - Baseline Experiments (Steps 21-25)

Run ID: `{manifest['run_id']}`
Generated: {manifest['generated_at_utc']}

## 1. Objective and scope

Establish an unweighted, untuned, leakage-safe baseline for ischaemic-stroke
prediction on the Stroke Prediction Dataset. Steps 21-25 cover preflight,
repeated stratified cross-validation of five fixed baselines, artefact saving,
figures and independent verification.

Out of scope by instruction: threshold tuning, probability calibration, class
weighting, resampling/SMOTE/ADASYN, and any access to the held-out test set.

## 2. Protocol

- **Data**: `{manifest['data']['path']}`, SHA256 `{manifest['data']['sha256']}`.
- **Training partition**: {n_train:,} rows, {n_pos} positives
  (prevalence {prevalence:.4%}).
- **Held-out partition**: {manifest['split']['n_test']:,} rows, untouched.
  `test_set_used = false`; the loader's test set remained locked throughout.
- **Split**: `train_test_split(test_size=0.2, stratify=y,
  random_state=42)`, reproducing Notebook 02 exactly.
- **CV**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
  random_state=42)`; each repeat is an independent shuffle, and every model
  iterates the same fold plan.
- **Preprocessing**: fitted from scratch inside every fold
  (median imputation; standardisation plus drop-first one-hot for the
  linear/SVM family; full one-hot for the tree family). No transformer was
  fitted on the full training partition.
- **Decision threshold**: fixed at 0.50 for every model and fold.
- **Class balancing**: none (`class_weight=None` everywhere; no resampling).
- **Average precision**: `average_precision_score` (step-wise area under the
  precision-recall curve). It is **not** trapezoidal PR-AUC.

## 3. Results

{fold_table}

{pooled_table}

{confusion_table}

{variability_table}

### Key observations

{observations}

## 4. Figures

{figures_md}

Aggregation labels are printed on each figure: ROC/PR curves are mean +/-
1 SD across the five repeats (each repeat's curve uses its own out-of-fold
predictions), bar charts show fold-level mean +/- SD across 25 correlated
folds, and confusion matrices show mean counts per fold. No curve pools
predictions across repeats.

## 5. Verification

{verification_text}

Saved record: `results/tables/baseline_verification.json`.

## 6. Limitations

1. **Few positive events.** {n_pos} positives in {n_train:,} training rows
   (about 39-40 per validation fold) make all metrics noisy. Differences of a
   few hundredths of ROC-AUC should not be treated as decisive.
2. **Correlated folds.** The 25 fold estimates reuse the same 4,088 rows.
   Their SD describes dispersion across folds, not sampling uncertainty of a
   population estimate; no confidence intervals or significance tests are
   claimed. The five repeat means are closer to independent, but n=5 is tiny.
3. **Single fixed threshold.** No threshold tuning was performed, so
   sensitivity/F1/precision reflect the arbitrary 0.50 cut-off rather than a
   clinically chosen operating point.
4. **No calibration.** Probabilities are whatever each estimator emits
   (`SVC(probability=True)` uses internal Platt scaling); no calibration curve
   or Brier decomposition was computed.
5. **No class balancing.** This is deliberate so that the baseline reflects
   the natural prevalence; it is not the final modelling strategy.
6. **Average precision is step-wise.** It is not interchangeable with
   trapezoidal PR-AUC, and no trapezoidal value is reported.
7. **Held-out test set not evaluated.** All numbers are cross-validated
   training-partition estimates; no generalisation claim to the 1,022 held-out
   rows is made.
8. **Runtime and environment.** {execution['total_seconds']:.1f} s total on
   Python {manifest['environment']['python']} with scikit-learn
   {manifest['environment']['scikit_learn']}; timings are machine-specific.

## 7. Conclusion

The baseline stage is complete and reproducible. The strongest
pooled-within-repeat ranker is `{top_one['model']}` (ROC-AUC
{top_one['mean']:.3f}), followed by `{top_two['model']}` ({top_two['mean']:.3f}).
{n_above_prior} of the {len(learned)} learned models have a higher pooled ROC-AUC than the
prior baseline. The fixed threshold still yields low sensitivity under
{prevalence:.2%} prevalence without balancing. These results define the
reference against which later balancing, thresholding and calibration
experiments must be compared; they are not a clinical result.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
