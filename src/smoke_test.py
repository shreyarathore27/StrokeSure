"""Step 20 smoke test: cheap end-to-end validation on the training partition.

Run with::

    python -m src.smoke_test

What is checked
---------------
1. Raw data loads, validates and keeps its checksum (no writes).
2. The 80/20 stratified split reproduces Notebook 02 exactly and the
   partitions are disjoint.
3. The held-out test partition stays locked; any access raises.
4. Class distributions in the full data, train partition and smoke subsample.
5. Both preprocessing families produce the expected feature counts, contain no
   missing values and keep ``smoking_status="Unknown"`` as its own category.
6. Preprocessing is fitted inside each CV fold: the fold imputer statistic
   equals that fold's median, not the global median.
7. Stratified 3-fold CV runs for all five baselines without errors and the
   probabilities are valid.
8. Metric implementations agree with direct sklearn/manual calculations.

Outputs
-------
Everything is written under ``results/smoke_test/`` and never under
``results/tables`` or ``results/figures``, so smoke artefacts cannot be
mistaken for publication results.

The test deliberately does **not** train final models, run the repeated CV
experiment, tune thresholds, calibrate probabilities or touch the test set.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DECISION_THRESHOLD,
    EXPECTED_FEATURE_COUNTS,
    MODEL_FAMILIES,
    MODEL_FAMILY_LINEAR_SVM,
    RANDOM_STATE,
    SMOKE_CV_SPLITS,
    SMOKE_FIGURES_DIR,
    SMOKE_RUN_PREFIX,
    SMOKE_SAMPLE_SIZE,
    SMOKE_TABLES_DIR,
    ensure_output_directories,
)
from src.data_loading import (
    TestSetAccessError,
    build_features_and_target,
    load_raw_dataframe,
    load_split,
    stratified_subsample,
    stratified_train_test_split,
)
from src.evaluation import (
    build_manifest,
    build_run_id,
    compute_metrics,
    confusion_matrix_table,
    configure_run_logger,
    cross_validate_pipeline,
    environment_snapshot,
    make_stratified_cv,
    save_cv_result,
    write_json,
)
from src.models import (
    build_baseline_pipelines,
    make_logistic_regression,
    model_family,
    model_parameter_table,
)
from src.preprocessing import (
    CATEGORY_VALUES,
    build_model_pipeline,
    feature_names_out,
    has_explicit_unknown_smoking_status,
)


class SmokeTestFailure(AssertionError):
    """Raised when a smoke-test expectation is not met."""


def check(condition: bool, message: str) -> None:
    """Assert with a descriptive message."""
    if not condition:
        raise SmokeTestFailure(message)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_split_reproduction(split, logger) -> None:
    """Independently reproduce the Notebook 02 split and compare indices."""
    section("2. Split reproduction and integrity")

    df = load_raw_dataframe()
    X, y = build_features_and_target(df)
    X_tr_ref, X_te_ref, y_tr_ref, y_te_ref = stratified_train_test_split(X, y)

    check(
        X_tr_ref.index.equals(split.X_train.index),
        "Training indices differ from an independent Notebook 02 reproduction.",
    )
    check(
        X_te_ref.index.equals(split.test_index),
        "Test indices differ from an independent Notebook 02 reproduction.",
    )
    check(
        y_tr_ref.equals(split.y_train),
        "Training labels differ from an independent reproduction.",
    )
    check(
        y_te_ref.equals(split.test.y),
        "Test labels differ from an independent reproduction.",
    )

    overlap = set(split.train_index) & set(split.test_index)
    check(len(overlap) == 0, f"Train/test overlap detected: {len(overlap)} rows.")
    check(
        len(split.train_index) + len(split.test_index) == len(df),
        "Train and test partitions do not cover the entire dataset.",
    )
    check(
        len(split.X_train) == 4088 and len(split.test_index) == 1022,
        "Unexpected 80/20 partition sizes.",
    )

    print(f"  train rows / features : {split.X_train.shape}")
    print(f"  test rows / features  : {split.test.shape} (locked)")
    print(f"  overlapping rows      : {len(overlap)}")
    print(f"  full prevalence       : {split.metadata['full_prevalence']:.6f}")
    print(f"  train prevalence      : {split.metadata['train_prevalence']:.6f}")
    print(f"  test prevalence       : {split.metadata['test_prevalence']:.6f}")
    logger.info(
        "Split reproduced: train=%d test=%d overlap=%d",
        len(split.train_index),
        len(split.test_index),
        len(overlap),
    )


def check_test_partition_lock(split, logger) -> None:
    section("3. Held-out test partition lock")
    check(not split.test.unlocked, "Test set is unexpectedly unlocked.")

    raised = False
    try:
        split.test.get()
    except TestSetAccessError:
        raised = True
    check(raised, "Locked test set did not raise TestSetAccessError on access.")

    print("  state                : LOCKED")
    print("  access attempt       : raised TestSetAccessError (expected)")
    print(f"  test class counts    : {split.test.class_counts} (metadata only)")
    logger.info("Test partition remained locked; access correctly refused.")


def check_class_distributions(split, X_smoke, y_smoke, logger) -> None:
    section("4. Class distributions")
    full = split.metadata
    print(
        "  full data            : "
        f"{full['n_rows']} rows, positives={full['train_class_counts']['1'] + full['test_class_counts']['1']}, "
        f"prevalence={full['full_prevalence']:.4%}"
    )
    print(
        "  training partition   : "
        f"{split.n_train} rows, positives={full['train_class_counts']['1']}, "
        f"prevalence={full['train_prevalence']:.4%}"
    )
    print(
        "  smoke subsample      : "
        f"{len(y_smoke)} rows, positives={int(y_smoke.sum())}, "
        f"prevalence={y_smoke.mean():.4%}"
    )

    check(int(y_smoke.sum()) > 0, "Smoke subsample contains no positive cases.")
    check(
        abs(float(y_smoke.mean()) - float(split.y_train.mean())) < 0.01,
        "Smoke subsample class prevalence deviates by more than 1 percentage point.",
    )
    logger.info(
        "Smoke subsample: n=%d positives=%d prevalence=%.4f",
        len(y_smoke),
        int(y_smoke.sum()),
        float(y_smoke.mean()),
    )


def check_preprocessing(X_smoke: pd.DataFrame, y_smoke: pd.Series, logger) -> dict[str, Any]:
    section("5. Preprocessing families")
    results: dict[str, Any] = {}

    for feature, declared in CATEGORY_VALUES.items():
        observed = set(X_smoke[feature].dropna().unique().tolist())
        undeclared = observed - set(declared)
        check(
            not undeclared,
            f"{feature}: encoder categories miss observed values {undeclared}.",
        )

    for family in MODEL_FAMILIES:
        pipeline = build_model_pipeline(make_logistic_regression(), family)
        pipeline.fit(X_smoke, y_smoke)
        names = feature_names_out(pipeline)
        expected = EXPECTED_FEATURE_COUNTS[family]

        transformed = pipeline.named_steps["preprocessor"].transform(X_smoke)
        check(
            transformed.shape[1] == expected,
            f"{family}: expected {expected} features, got {transformed.shape[1]}.",
        )
        check(
            not np.isnan(transformed).any(),
            f"{family}: preprocessing left missing values behind.",
        )
        check(
            has_explicit_unknown_smoking_status(pipeline),
            f"{family}: smoking_status='Unknown' is not an explicit category.",
        )

        numeric_mean = None
        if family == MODEL_FAMILY_LINEAR_SVM:
            numeric_block = transformed[:, :3]
            numeric_mean = float(np.abs(numeric_block.mean(axis=0)).max())
            numeric_std = float(np.abs(numeric_block.std(axis=0) - 1).max())
            check(
                numeric_mean < 1e-8,
                f"{family}: scaled numeric features are not centred.",
            )
            check(
                numeric_std < 1e-8,
                f"{family}: scaled numeric features are not unit variance.",
            )

        print(
            f"  {family:11s} features={len(names):2d} (expected {expected}), "
            f"no NaN, Unknown category present"
            + (
                f", |max numeric mean|={numeric_mean:.2e}"
                if numeric_mean is not None
                else ""
            )
        )
        results[family] = {"n_features": len(names), "names": names}

    return results


def check_fold_local_preprocessing(
    X_smoke: pd.DataFrame, y_smoke: pd.Series, logger
) -> dict[str, Any]:
    """Prove that preprocessing statistics come from the fold, not the full data."""
    section("6. Fold-local preprocessing (leakage check)")

    cv = StratifiedKFold(
        n_splits=SMOKE_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_train, fold_valid = next(cv.split(X_smoke, y_smoke))

    pipeline = build_model_pipeline(make_logistic_regression(), MODEL_FAMILY_LINEAR_SVM)
    pipeline.fit(X_smoke.iloc[fold_train], y_smoke.iloc[fold_train])

    preprocessor = pipeline.named_steps["preprocessor"]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    bmi_position = 2  # NUMERIC_FEATURES = age, avg_glucose_level, bmi
    fold_median = float(X_smoke.iloc[fold_train]["bmi"].median())
    global_median = float(X_smoke["bmi"].median())
    fitted_median = float(imputer.statistics_[bmi_position])

    check(
        abs(fitted_median - fold_median) < 1e-12,
        "Imputer statistic does not match the fold-training median "
        f"({fitted_median} vs {fold_median}).",
    )

    fold_age_mean = float(X_smoke.iloc[fold_train]["age"].mean())
    check(
        abs(float(scaler.mean_[0]) - fold_age_mean) < 1e-9,
        "Scaler mean does not match the fold-training mean.",
    )

    print(f"  fold training rows   : {len(fold_train)}")
    print(f"  fold BMI median      : {fitted_median:.6f}")
    print(f"  full-train BMI median: {global_median:.6f}")
    print(f"  imputer used fold value: {abs(fitted_median - fold_median) < 1e-12}")
    if abs(fold_median - global_median) < 1e-12:
        print(
            "  note                 : fold and global medians coincide here, "
            "but the check asserts the fold value, not the global value."
        )
    logger.info(
        "Fold-local preprocessing verified: fold median %.6f vs global %.6f",
        fold_median,
        global_median,
    )
    return {
        "fold_bmi_median": fold_median,
        "global_bmi_median": global_median,
        "fitted_imputer_statistic": fitted_median,
    }


def check_metric_calculations() -> None:
    section("7. Metric calculation checks")

    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.35, 0.80, 0.60, 0.90])
    y_pred = (p >= DECISION_THRESHOLD).astype(int)

    metrics = compute_metrics(y, p)

    expected_specificity = 2 / 3  # TN=2, FP=1
    expected_balanced = (1.0 + expected_specificity) / 2
    expected_brier = float(np.mean((p - y) ** 2))

    checks = {
        "roc_auc": (metrics["roc_auc"], roc_auc_score(y, p)),
        "average_precision": (
            metrics["average_precision"],
            average_precision_score(y, p),
        ),
        "recall_sensitivity": (metrics["recall_sensitivity"], 1.0),
        "precision": (metrics["precision"], 2 / 3),
        "f1": (metrics["f1"], f1_score(y, y_pred, zero_division=0)),
        "specificity": (metrics["specificity"], expected_specificity),
        "balanced_accuracy": (metrics["balanced_accuracy"], expected_balanced),
        "brier_score": (metrics["brier_score"], expected_brier),
    }
    for name, (actual, expected) in checks.items():
        check(
            abs(actual - expected) < 1e-12,
            f"Metric {name}: {actual} != expected {expected}.",
        )

    check(
        metrics["tn"] == 2 and metrics["fp"] == 1 and metrics["fn"] == 0
        and metrics["tp"] == 2,
        f"Confusion counts are wrong: {metrics}.",
    )
    check(
        metrics["balanced_accuracy"] == balanced_accuracy_score(y, y_pred),
        "Balanced accuracy disagrees with sklearn.",
    )
    check(
        abs(metrics["brier_score"] - brier_score_loss(y, p)) < 1e-12,
        "Brier score disagrees with sklearn.",
    )
    check(
        metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
        == metrics["n_samples"],
        "Confusion matrix does not sum to the number of samples.",
    )

    perfect = compute_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    check(
        abs(perfect["roc_auc"] - 1.0) < 1e-12
        and abs(perfect["average_precision"] - 1.0) < 1e-12
        and abs(perfect["brier_score"]) < 1e-12,
        f"Perfect predictions did not score as expected: {perfect}.",
    )

    for bad in ([-0.01, 0.5], [0.5, 1.01]):
        raised = False
        try:
            compute_metrics([0, 1], bad)
        except ValueError:
            raised = True
        check(raised, "compute_metrics accepted out-of-range probabilities.")

    print("  toy example          : all metrics match manual/sklearn values")
    print(f"  specificity          : {metrics['specificity']:.6f} (2/3)")
    print(f"  average precision    : {metrics['average_precision']:.6f} "
          "(step-wise, not trapezoidal)")
    print(f"  brier score          : {metrics['brier_score']:.6f} "
          "(mean squared probability error)")
    print("  perfect predictions  : ROC-AUC=1, AP=1, Brier=0")
    print("  invalid probabilities: correctly rejected")


def check_unknown_category_handling(
    X_smoke: pd.DataFrame, y_smoke: pd.Series
) -> None:
    section("8. Unseen-category handling")

    pipeline = build_model_pipeline(make_logistic_regression(), MODEL_FAMILY_LINEAR_SVM)
    pipeline.fit(X_smoke, y_smoke)

    preprocessor = pipeline.named_steps["preprocessor"]
    names = list(preprocessor.get_feature_names_out())
    smoking_columns = [
        i for i, name in enumerate(names) if name.startswith("cat__smoking_status_")
    ]

    unseen = X_smoke.iloc[[0]].copy()
    unseen["smoking_status"] = "e-cigarette"
    with warnings.catch_warnings():
        # The all-zero encoding emits an expected UserWarning; this test
        # deliberately exercises that path.
        warnings.simplefilter("ignore", UserWarning)
        transformed = preprocessor.transform(unseen)

    check(
        float(transformed[0, smoking_columns].sum()) == 0.0,
        "An unseen smoking category produced a non-zero one-hot block.",
    )
    check(
        "cat__smoking_status_Unknown" in names,
        "The explicit 'Unknown' smoking category is missing.",
    )

    known = X_smoke.iloc[[0]].copy()
    known_transformed = preprocessor.transform(known)
    print(f"  unseen category      : encoded as all-zero block ({len(smoking_columns)} columns)")
    print("  'Unknown' category   : retained as explicit column")
    print(
        "  observed row category: "
        f"{known['smoking_status'].iloc[0]!r}"
    )


def run_models(
    X_smoke: pd.DataFrame, y_smoke: pd.Series, logger
) -> tuple[list[Any], dict[str, float], pd.DataFrame]:
    section("9. Stratified cross-validation smoke run")
    cv = make_stratified_cv(n_splits=SMOKE_CV_SPLITS)

    pipelines = build_baseline_pipelines()
    results: list[Any] = []
    timings: dict[str, float] = {}
    summary_rows: list[pd.Series] = []

    for name, pipeline in pipelines.items():
        started = time.perf_counter()
        result = cross_validate_pipeline(
            pipeline, X_smoke, y_smoke, model_name=name, cv=cv
        )
        elapsed = time.perf_counter() - started
        timings[name] = elapsed

        proba = result.oof_probabilities
        check(
            proba.shape == (len(X_smoke),),
            f"{name}: unexpected OOF probability shape {proba.shape}.",
        )
        check(
            np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all(),
            f"{name}: probabilities are invalid.",
        )
        check(
            result.n_features == EXPECTED_FEATURE_COUNTS[model_family(name)],
            f"{name}: unexpected number of preprocessing features.",
        )

        row = result.summaries.set_index("metric")["mean"]
        summary_rows.append(
            pd.Series(
                {
                    "model": name,
                    "roc_auc": row["roc_auc"],
                    "average_precision": row["average_precision"],
                    "recall_sensitivity": row["recall_sensitivity"],
                    "specificity": row["specificity"],
                    "precision": row["precision"],
                    "f1": row["f1"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "brier_score": row["brier_score"],
                    "n_features": result.n_features,
                    "seconds": elapsed,
                }
            )
        )
        results.append(result)

        print(
            f"  {name:22s} folds={SMOKE_CV_SPLITS} features={result.n_features:2d} "
            f"ROC-AUC={row['roc_auc']:.3f} AP={row['average_precision']:.3f} "
            f"sens={row['recall_sensitivity']:.3f} spec={row['specificity']:.3f} "
            f"Brier={row['brier_score']:.3f} ({elapsed:.1f}s)"
        )
        logger.info(
            "CV ok: %s folds=%d features=%d roc_auc=%.4f average_precision=%.4f",
            name,
            SMOKE_CV_SPLITS,
            result.n_features,
            row["roc_auc"],
            row["average_precision"],
        )

    return results, timings, pd.DataFrame(summary_rows)


def save_outputs(
    run_id: str,
    results: list[Any],
    timings: dict[str, float],
    split,
    fold_stats: dict[str, Any],
    n_smoke: int,
    logger,
) -> dict[str, Path]:
    section("11. Writing smoke-test artefacts")

    tables = SMOKE_TABLES_DIR
    figures = SMOKE_FIGURES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    all_folds = pd.concat([r.fold_metrics for r in results], ignore_index=True)
    fold_path = tables / f"{run_id}_fold_metrics.csv"
    all_folds.to_csv(fold_path, index=False)
    paths["fold_metrics"] = fold_path

    summary = pd.concat(
        [
            r.summaries.assign(model=r.model_name, scope="fold_mean_sd")
            for r in results
        ],
        ignore_index=True,
    )
    summary_path = tables / f"{run_id}_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    pooled = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "scope": "pooled_out_of_fold",
                **r.pooled_metrics,
                "average_precision_definition": (
                    "average_precision_score (step-wise PR area, not trapezoidal)"
                ),
            }
            for r in results
        ]
    )
    pooled_path = tables / f"{run_id}_pooled_metrics.csv"
    pooled.to_csv(pooled_path, index=False)
    paths["pooled_metrics"] = pooled_path

    cm_table = confusion_matrix_table(results)
    cm_path = tables / f"{run_id}_confusion_matrices.csv"
    cm_table.to_csv(cm_path, index=False)
    paths["confusion_matrices"] = cm_path

    for result in results:
        save_cv_result(result, run_id, tables, figures)

    extra = {
        "scope": "smoke test only - not publication results",
        "sample_size": int(n_smoke),
        "cv_splits": SMOKE_CV_SPLITS,
        "timings_seconds": timings,
        "fold_local_preprocessing": fold_stats,
        "test_set_used": False,
        "model_parameters": model_parameter_table(),
        "notes": [
            "Smoke test ran on a stratified subsample of the training partition.",
            "The held-out test partition was never unlocked or evaluated.",
            "No threshold tuning, calibration or class balancing was performed.",
        ],
    }
    manifest = build_manifest(
        run_id=run_id,
        step="step20_smoke_test",
        results=results,
        data_metadata=split.metadata,
        extra=extra,
    )
    manifest_path = tables / f"{run_id}_manifest.json"
    write_json(manifest, manifest_path)
    paths["manifest"] = manifest_path

    for path in paths.values():
        print(f"  wrote {path.relative_to(Path.cwd())}")
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_smoke_test() -> int:
    ensure_output_directories()
    run_id = build_run_id(SMOKE_RUN_PREFIX)
    log_path = SMOKE_TABLES_DIR.parent / f"{run_id}.log"
    logger = configure_run_logger(run_id, log_path)

    started = time.perf_counter()

    section("StrokeSure - Step 20 smoke test")
    print(f"  run id               : {run_id}")
    print(f"  log file             : {log_path}")
    env = environment_snapshot()
    print(f"  python               : {env['python']} ({env['python_executable']})")
    print(f"  numpy/scipy/pandas   : {env['numpy']} / {env['scipy']} / {env['pandas']}")
    print(f"  scikit-learn         : {env['scikit_learn']}")
    print(f"  git commit           : {env['git_commit']}")

    section("1. Data loading and validation")
    split = load_split()
    print(f"  raw rows / columns   : {split.metadata['n_rows']} / {split.metadata['n_features'] + 2}")
    print(f"  feature columns      : {split.metadata['n_features']} (id and target excluded)")
    print(f"  raw data sha256      : {split.metadata['data_sha256'][:16]}...")
    logger.info("Data validated: %s", split.metadata)

    check_split_reproduction(split, logger)
    check_test_partition_lock(split, logger)

    X_smoke, y_smoke = stratified_subsample(
        split.X_train, split.y_train, n_samples=SMOKE_SAMPLE_SIZE
    )
    check_class_distributions(split, X_smoke, y_smoke, logger)

    check_preprocessing(X_smoke, y_smoke, logger)
    fold_stats = check_fold_local_preprocessing(X_smoke, y_smoke, logger)
    check_metric_calculations()
    check_unknown_category_handling(X_smoke, y_smoke)

    results, timings, summary_table = run_models(X_smoke, y_smoke, logger)

    section("10. Cross-validation summary (smoke subsample, 3 folds)")
    print(summary_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    save_outputs(run_id, results, timings, split, fold_stats, len(X_smoke), logger)

    total = time.perf_counter() - started

    section("RESULT")
    check(not split.test.unlocked, "Test set was unlocked during the smoke test.")
    check(
        not any(r.pooled_metrics["n_samples"] == split.n_test for r in results),
        "A result set matches the test-set size; investigate before proceeding.",
    )

    print(f"  status               : PASSED")
    print(f"  total runtime        : {total:.1f}s")
    print(f"  models exercised     : {len(results)}")
    print(f"  test partition       : never unlocked, never evaluated")
    print(f"  artefacts            : {SMOKE_TABLES_DIR} and {SMOKE_FIGURES_DIR}")
    logger.info("SMOKE TEST PASSED in %.1fs", total)

    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
