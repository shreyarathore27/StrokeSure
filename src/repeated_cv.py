"""Steps 21-25: repeated stratified CV of the five StrokeSure baselines.

Run with::

    python -m src.repeated_cv

What happens
------------
1. Preflight (`src.preflight`): data hash, split, lock, unfitted pipelines,
   unweighted estimators and the 5 x 5 fold plan.
2. Fit every baseline on the full 4,088-row training partition using
   ``RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)``.
   All models iterate the identical fold plan, so folds are paired.
3. Save fold-level metrics, a long-format summary, out-of-fold predictions,
   pooled-within-repeat metrics and a machine-readable manifest.
4. Render dependency-free SVG figures.
5. Independently verify the saved artefacts (`src.verification`).
6. Write a scientific report from the computed numbers.

The held-out test set is never accessed, no threshold tuning or calibration is
performed, and no class balancing is applied.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import (
    BASELINE_RUN_PREFIX,
    CV_N_REPEATS,
    CV_N_SPLITS,
    DECISION_THRESHOLD,
    EXPECTED_FEATURE_COUNTS,
    FIGURES_DIR,
    LOGS_DIR,
    METRIC_LABELS,
    METRIC_NAMES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TABLES_DIR,
    TEST_SIZE,
    ensure_output_directories,
)
from src.data_loading import load_split
from src.evaluation import (
    build_run_id,
    compute_metrics,
    configure_run_logger,
    environment_snapshot,
    utc_timestamp,
    write_json,
)
from src.figures import (
    pr_curve_bundle,
    roc_curve_bundle,
    save_brier_comparison,
    save_confusion_matrix_grid,
    save_metric_comparison,
    save_pr_figure,
    save_roc_figure,
)
from src.models import BASELINE_MODEL_SPECS, build_baseline_pipelines, model_names
from src.preflight import build_fold_plan, run_preflight, validate_fold_plan
from src.reporting import write_report

TABLES = [
    "baseline_cv_results.csv",
    "baseline_summary.csv",
    "baseline_oof_predictions.csv",
    "baseline_pooled_metrics_by_repeat.csv",
    "baseline_fold_plan.csv",
]

FIGURES = [
    "fig_roc_curves.svg",
    "fig_precision_recall_curves.svg",
    "fig_metric_comparison.svg",
    "fig_brier_comparison.svg",
    "fig_confusion_matrices.svg",
]

FOLD_LEVEL_NOTE = (
    "Descriptive mean/SD over 25 correlated fold estimates "
    "(5 repeats x 5 folds); not a standard error."
)
REPEAT_MEAN_NOTE = (
    "Mean/SD over 5 per-repeat means; each repeat is an independent shuffle "
    "with 5 folds."
)
POOLED_NOTE = (
    "Mean/SD over 5 pooled out-of-fold estimates; within one repeat every "
    "training row is predicted exactly once."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def build_summary(
    cv: pd.DataFrame, pooled: pd.DataFrame, models: list[str]
) -> pd.DataFrame:
    """Long-format summary separating fold-level from pooled aggregation."""
    rows: list[dict[str, Any]] = []
    for model in models:
        subset = cv[cv["model"] == model]

        for metric in METRIC_NAMES:
            values = subset[metric].astype(float)
            rows.append(
                {
                    "model": model,
                    "scope": "fold_level",
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_units": int(len(values)),
                    "note": FOLD_LEVEL_NOTE,
                }
            )

        per_repeat_means = subset.groupby("repeat")[METRIC_NAMES].mean()
        for metric in METRIC_NAMES:
            values = per_repeat_means[metric].astype(float)
            rows.append(
                {
                    "model": model,
                    "scope": "repeat_mean",
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_units": int(len(values)),
                    "note": REPEAT_MEAN_NOTE,
                }
            )

        pooled_subset = pooled[pooled["model"] == model]
        for metric in METRIC_NAMES:
            values = pooled_subset[metric].astype(float)
            rows.append(
                {
                    "model": model,
                    "scope": "pooled_within_repeat",
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_units": int(len(values)),
                    "note": POOLED_NOTE,
                }
            )

    summary = pd.DataFrame(rows)
    scope_order = {
        "fold_level": 0,
        "repeat_mean": 1,
        "pooled_within_repeat": 2,
    }
    metric_order = {metric: index for index, metric in enumerate(METRIC_NAMES)}
    summary["_scope_order"] = summary["scope"].map(scope_order)
    summary["_metric_order"] = summary["metric"].map(metric_order)
    summary = summary.sort_values(
        ["model", "_scope_order", "_metric_order"], kind="stable"
    ).drop(columns=["_scope_order", "_metric_order"])
    return summary.reset_index(drop=True)


def run_experiment() -> dict[str, Any]:
    ensure_output_directories()
    run_id = build_run_id(BASELINE_RUN_PREFIX)
    logger = configure_run_logger(run_id, LOGS_DIR / f"{run_id}.log")

    wall_start = time.perf_counter()
    logger.info("=" * 76)
    logger.info("StrokeSure Steps 21-25 - repeated stratified CV")
    logger.info("run_id=%s", run_id)
    logger.info("=" * 76)

    # ------------------------------------------------------------------
    # Step 21 - preflight
    # ------------------------------------------------------------------
    preflight = run_preflight()
    logger.info(
        "preflight passed: train=%d, test=%d, folds=%d",
        preflight["n_train"],
        preflight["n_test"],
        preflight["fold_plan"]["n_folds_total"],
    )

    split = load_split()
    X = split.X_train
    y = split.y_train
    models = model_names()
    pipelines = build_baseline_pipelines()
    plan = build_fold_plan(X, y)
    validate_fold_plan(plan, len(X))

    if split.test.unlocked:
        raise RuntimeError("Held-out test set was unlocked; aborting.")

    logger.info(
        "training partition: %d rows, %d positives, prevalence %.4f",
        len(X),
        int(y.sum()),
        float(y.mean()),
    )
    logger.info(
        "CV: RepeatedStratifiedKFold(n_splits=%d, n_repeats=%d, random_state=%d)",
        CV_N_SPLITS,
        CV_N_REPEATS,
        RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Step 22 - run all five baselines over the shared fold plan
    # ------------------------------------------------------------------
    fold_records: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    per_model_seconds: dict[str, float] = {}

    for model in models:
        pipeline = pipelines[model]
        expected_features = EXPECTED_FEATURE_COUNTS[
            BASELINE_MODEL_SPECS[model].family
        ]
        model_start = time.perf_counter()
        logger.info("model %s: starting 25 folds", model)

        for spec in plan:
            X_tr = X.iloc[spec.train_positions]
            X_va = X.iloc[spec.valid_positions]
            y_tr = y.iloc[spec.train_positions]
            y_va = y.iloc[spec.valid_positions]

            if int(y_tr.sum()) == 0 or int(y_va.sum()) == 0:
                raise RuntimeError(
                    f"{model} repeat {spec.repeat} fold {spec.fold}: "
                    "a fold contains only one class."
                )

            fold_start = time.perf_counter()
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_tr, y_tr)

            classes = list(fold_pipeline.classes_)
            proba = fold_pipeline.predict_proba(X_va)[:, classes.index(POSITIVE_LABEL)]
            elapsed = time.perf_counter() - fold_start

            n_features = len(
                fold_pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
            if n_features != expected_features:
                raise RuntimeError(
                    f"{model} repeat {spec.repeat} fold {spec.fold}: "
                    f"{n_features} features, expected {expected_features}."
                )

            metrics = compute_metrics(
                y_va.to_numpy(), proba, threshold=DECISION_THRESHOLD
            )
            fold_records.append(
                {
                    "model": model,
                    "repeat": spec.repeat,
                    "fold": spec.fold,
                    "n_train": int(len(X_tr)),
                    "n_validation": int(len(X_va)),
                    "n_positive_validation": int((y_va == POSITIVE_LABEL).sum()),
                    "n_features": n_features,
                    "execution_seconds": float(elapsed),
                    **metrics,
                }
            )
            oof_parts.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "repeat": spec.repeat,
                        "fold": spec.fold,
                        "row_index": X.index.values[spec.valid_positions],
                        "y_true": y.values[spec.valid_positions],
                        "y_proba": proba,
                    }
                )
            )

        model_seconds = time.perf_counter() - model_start
        per_model_seconds[model] = float(model_seconds)
        logger.info("model %s: finished in %.2f s", model, model_seconds)

    cv = pd.DataFrame(fold_records)
    oof = pd.concat(oof_parts, ignore_index=True)

    if len(cv) != len(models) * CV_N_SPLITS * CV_N_REPEATS:
        raise RuntimeError(f"Unexpected fold count: {len(cv)}.")
    if np.isnan(oof["y_proba"]).any():
        raise RuntimeError("Out-of-fold probabilities contain missing values.")

    # Pooled metrics are computed *within* each repeat, never across repeats.
    pooled_rows: list[dict[str, Any]] = []
    for (model, repeat), group in oof.groupby(["model", "repeat"], sort=True):
        metrics = compute_metrics(
            group["y_true"].to_numpy(),
            group["y_proba"].to_numpy(),
            threshold=DECISION_THRESHOLD,
        )
        pooled_rows.append({"model": model, "repeat": int(repeat), **metrics})
    pooled = pd.DataFrame(pooled_rows)

    summary = build_summary(cv, pooled, models)

    # ------------------------------------------------------------------
    # Step 23 - save tables
    # ------------------------------------------------------------------
    table_paths = {
        "baseline_cv_results.csv": TABLES_DIR / "baseline_cv_results.csv",
        "baseline_summary.csv": TABLES_DIR / "baseline_summary.csv",
        "baseline_oof_predictions.csv": TABLES_DIR / "baseline_oof_predictions.csv",
        "baseline_pooled_metrics_by_repeat.csv": (
            TABLES_DIR / "baseline_pooled_metrics_by_repeat.csv"
        ),
        "baseline_fold_plan.csv": TABLES_DIR / "baseline_fold_plan.csv",
    }
    cv.to_csv(table_paths["baseline_cv_results.csv"], index=False)
    summary.to_csv(table_paths["baseline_summary.csv"], index=False)
    oof.to_csv(table_paths["baseline_oof_predictions.csv"], index=False)

    pooled = pooled.copy()
    pooled["average_precision_definition"] = (
        "average_precision_score (step-wise PR area, not trapezoidal PR-AUC)"
    )
    pooled.to_csv(table_paths["baseline_pooled_metrics_by_repeat.csv"], index=False)

    fold_plan_frame = pd.DataFrame(preflight["fold_plan_specs"])
    fold_plan_frame.to_csv(table_paths["baseline_fold_plan.csv"], index=False)

    # ------------------------------------------------------------------
    # Step 24 - figures (SVG, no new dependency)
    # ------------------------------------------------------------------
    roc_bundle = roc_curve_bundle(oof, models)
    pr_bundle = pr_curve_bundle(oof, models)
    prevalence = float(y.mean())

    figure_paths = {
        "fig_roc_curves.svg": FIGURES_DIR / "fig_roc_curves.svg",
        "fig_precision_recall_curves.svg": FIGURES_DIR
        / "fig_precision_recall_curves.svg",
        "fig_metric_comparison.svg": FIGURES_DIR / "fig_metric_comparison.svg",
        "fig_brier_comparison.svg": FIGURES_DIR / "fig_brier_comparison.svg",
        "fig_confusion_matrices.svg": FIGURES_DIR / "fig_confusion_matrices.svg",
    }
    save_roc_figure(roc_bundle, models, figure_paths["fig_roc_curves.svg"])
    save_pr_figure(
        pr_bundle, models, figure_paths["fig_precision_recall_curves.svg"], prevalence
    )
    save_metric_comparison(summary, models, figure_paths["fig_metric_comparison.svg"])
    save_brier_comparison(summary, models, figure_paths["fig_brier_comparison.svg"])
    save_confusion_matrix_grid(
        cv, models, figure_paths["fig_confusion_matrices.svg"]
    )

    # ------------------------------------------------------------------
    # Experiment manifest
    # ------------------------------------------------------------------
    total_seconds = time.perf_counter() - wall_start
    environment = environment_snapshot()

    model_entries = []
    for model in models:
        spec = BASELINE_MODEL_SPECS[model]
        estimator = pipelines[model].named_steps["classifier"]
        params = {
            key: _jsonable(value)
            for key, value in estimator.get_params(deep=False).items()
        }
        model_entries.append(
            {
                "model": model,
                "classifier": type(estimator).__name__,
                "preprocessing_family": spec.family,
                "expected_features": EXPECTED_FEATURE_COUNTS[spec.family],
                "hyperparameters": params,
                "class_weight": params.get("class_weight", "not applicable"),
                "rationale": spec.rationale,
                "execution_seconds": per_model_seconds[model],
            }
        )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "project": "StrokeSure",
        "steps": "21-25",
        "generated_at_utc": utc_timestamp(),
        "environment": environment,
        "data": {
            "path": str(RAW_DATA_PATH),
            "sha256": split.metadata["data_sha256"],
            "n_rows": int(split.metadata["n_rows"]),
            "n_features": int(split.metadata["n_features"]),
            "target": "stroke",
            "excluded_columns": ["id"],
        },
        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": "stroke",
            "n_train": int(split.metadata["n_train"]),
            "n_test": int(split.metadata["n_test"]),
            "n_overlap": int(split.metadata["n_overlap"]),
            "train_class_counts": split.metadata["train_class_counts"],
            "test_class_counts": split.metadata["test_class_counts"],
            "train_prevalence": float(split.metadata["train_prevalence"]),
            "test_prevalence": float(split.metadata["test_prevalence"]),
        },
        "test_set": {
            "test_set_used": False,
            "test_locked": bool(not split.test.unlocked),
            "note": (
                "No model was fitted or evaluated on held-out rows; only label "
                "counts were inspected to verify stratification."
            ),
        },
        "cv": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": CV_N_SPLITS,
            "n_repeats": CV_N_REPEATS,
            "random_state": RANDOM_STATE,
            "shuffle_per_repeat": True,
            "threshold": DECISION_THRESHOLD,
            "threshold_tuning": False,
            "n_folds_total": int(len(cv)),
            "fold_plan": preflight["fold_plan_specs"],
            "repeat_assignment_fingerprints": preflight["fold_plan"][
                "repeat_assignment_fingerprints"
            ],
            "pairing": "All models used the identical fold plan.",
        },
        "models": model_entries,
        "class_balancing": {
            "applied": False,
            "class_weight": "None for every estimator",
            "resampling": "none",
            "smote_adasyn": "none",
        },
        "metrics": {
            "names": METRIC_NAMES,
            "labels": METRIC_LABELS,
            "threshold": DECISION_THRESHOLD,
            "average_precision_definition": (
                "average_precision_score: step-wise area under the PR curve; "
                "explicitly not trapezoidal PR-AUC"
            ),
            "execution_seconds": "fit + predict_proba per validation fold",
        },
        "seeds": {
            "global_random_state": RANDOM_STATE,
            "split_random_state": RANDOM_STATE,
            "cv_random_state": RANDOM_STATE,
            "estimator_random_state": RANDOM_STATE,
        },
        "aggregation": {
            "fold_level": FOLD_LEVEL_NOTE,
            "repeat_mean": REPEAT_MEAN_NOTE,
            "pooled_within_repeat": POOLED_NOTE,
            "across_repeat_pooling": (
                "Not performed: pooling OOF predictions from different repeats "
                "would duplicate every row five times."
            ),
        },
        "outputs": {
            "tables": [_relative(path) for path in table_paths.values()],
            "figures": [_relative(path) for path in figure_paths.values()],
            "manifest": _relative(TABLES_DIR / "baseline_experiment_manifest.json"),
            "verification": _relative(TABLES_DIR / "baseline_verification.json"),
            "report": _relative(PROJECT_ROOT / "results" / "step21_25_report.md"),
        },
        "execution": {
            "total_seconds": float(total_seconds),
            "per_model_seconds": per_model_seconds,
            "n_folds": int(len(cv)),
            "n_oof_predictions": int(len(oof)),
        },
    }

    manifest_path = TABLES_DIR / "baseline_experiment_manifest.json"
    write_json(manifest, manifest_path)
    logger.info("tables and manifest written to %s", TABLES_DIR)

    # ------------------------------------------------------------------
    # Step 25 - independent verification and report
    # ------------------------------------------------------------------
    from src.verification import verify_run

    verification = verify_run(tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR)
    if verification["overall"] != "PASSED":
        logger.error("verification failed; report will still be written")

    report_path = write_report(
        PROJECT_ROOT / "results" / "step21_25_report.md",
        manifest=manifest,
        cv=cv,
        summary=summary,
        pooled=pooled,
        verification=verification,
        figure_files=FIGURES,
    )

    logger.info("report written to %s", report_path)
    logger.info("total wall time: %.2f s", total_seconds)

    print()
    print("=" * 78)
    print("STEP 21-25 COMPLETE")
    print("=" * 78)
    print(f"run id            : {run_id}")
    print(f"fold rows         : {len(cv)} (5 models x 25 folds)")
    print(f"OOF predictions   : {len(oof):,}")
    print(f"verification      : {verification['overall']} "
          f"({verification['n_passed']}/{verification['n_checks']} checks)")
    print(f"total wall time   : {total_seconds:.1f} s")
    print()
    print("fold-level mean +/- SD (ROC-AUC | average precision):")
    for model in models:
        auc = summary[
            (summary["model"] == model)
            & (summary["scope"] == "fold_level")
            & (summary["metric"] == "roc_auc")
        ].iloc[0]
        ap = summary[
            (summary["model"] == model)
            & (summary["scope"] == "fold_level")
            & (summary["metric"] == "average_precision")
        ].iloc[0]
        print(
            f"  {model:22s} {auc['mean']:.3f} +/- {auc['sd']:.3f} | "
            f"{ap['mean']:.3f} +/- {ap['sd']:.3f}"
        )

    return {
        "run_id": run_id,
        "cv": cv,
        "summary": summary,
        "pooled": pooled,
        "oof": oof,
        "manifest": manifest,
        "verification": verification,
        "report_path": report_path,
    }


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as exc:  # noqa: BLE001 - surface the failure and exit non-zero
        print(f"EXPERIMENT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
