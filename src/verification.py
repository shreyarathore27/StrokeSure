"""Step 25: independent verification of the saved repeated-CV artefacts.

Run with::

    python -m src.verification

The verifier never refits the whole experiment.  It re-reads the saved CSVs,
recomputes metrics directly with scikit-learn from the out-of-fold
probabilities, checks fold pairing and repeat independence, confirms the
held-out test set was never touched, and refits exactly one fold to confirm
bit-level determinism.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DECISION_THRESHOLD,
    METRIC_NAMES,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TABLES_DIR,
    FIGURES_DIR,
)
from src.data_loading import load_split
from src.evaluation import utc_timestamp, write_json
from src.models import model_names, build_baseline_pipelines
from src.preflight import build_fold_plan, validate_fold_plan

TOLERANCE = 1e-9

REQUIRED_TABLE_FILES = [
    "baseline_cv_results.csv",
    "baseline_summary.csv",
    "baseline_oof_predictions.csv",
    "baseline_pooled_metrics_by_repeat.csv",
    "baseline_fold_plan.csv",
    "baseline_experiment_manifest.json",
]

REQUIRED_FIGURE_FILES = [
    "fig_roc_curves.svg",
    "fig_precision_recall_curves.svg",
    "fig_metric_comparison.svg",
    "fig_brier_comparison.svg",
    "fig_confusion_matrices.svg",
]


def _metrics_from_predictions(
    y_true: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Recompute the metric suite with scikit-learn only (no project helper)."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


class CheckLog:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )
        if not passed:
            print(f"[FAIL] {name}: {detail}")
        else:
            print(f"[PASS] {name}: {detail}")

    def require(self, name: str, condition: bool, detail: str) -> None:
        self.add(name, bool(condition), detail)


def _max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    if len(joined) != max(len(left), len(right)):
        return float("inf")
    return float((joined["a"] - joined["b"]).abs().max())


def verify_run(
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run every verification check and write ``baseline_verification.json``."""
    log = CheckLog()

    # ------------------------------------------------------------------
    # 1. Artefacts exist
    # ------------------------------------------------------------------
    missing_tables = [f for f in REQUIRED_TABLE_FILES if not (tables_dir / f).is_file()]
    log.require(
        "artefacts: tables present",
        not missing_tables,
        "all required tables present" if not missing_tables else f"missing {missing_tables}",
    )
    missing_figures = [f for f in REQUIRED_FIGURE_FILES if not (figures_dir / f).is_file()]
    log.require(
        "artefacts: figures present",
        not missing_figures,
        "all required figures present" if not missing_figures else f"missing {missing_figures}",
    )

    empty_svgs = [
        f
        for f in REQUIRED_FIGURE_FILES
        if (figures_dir / f).is_file()
        and (figures_dir / f).stat().st_size < 500
    ]
    log.require(
        "artefacts: figures non-empty",
        not empty_svgs,
        "every SVG exceeds 500 bytes" if not empty_svgs else f"suspicious {empty_svgs}",
    )

    cv = pd.read_csv(tables_dir / "baseline_cv_results.csv")
    summary = pd.read_csv(tables_dir / "baseline_summary.csv")
    oof = pd.read_csv(tables_dir / "baseline_oof_predictions.csv")
    pooled = pd.read_csv(tables_dir / "baseline_pooled_metrics_by_repeat.csv")
    fold_plan = pd.read_csv(tables_dir / "baseline_fold_plan.csv")
    manifest = json.loads(
        (tables_dir / "baseline_experiment_manifest.json").read_text(encoding="utf-8")
    )

    models = model_names()

    # ------------------------------------------------------------------
    # 2. Model and fold counts
    # ------------------------------------------------------------------
    log.require(
        "cv: row count",
        len(cv) == 125,
        f"{len(cv)} fold rows (expected 125 = 5 models x 25 folds)",
    )
    log.require(
        "cv: model set",
        set(cv["model"]) == set(models),
        f"models = {sorted(cv['model'].unique())}",
    )
    per_model = cv.groupby("model").size()
    log.require(
        "cv: 25 folds per model",
        bool((per_model == 25).all()),
        f"counts = {per_model.to_dict()}",
    )
    combos = cv.groupby("model")[["repeat", "fold"]].apply(
        lambda frame: frame.drop_duplicates().shape[0]
    )
    log.require(
        "cv: unique (repeat, fold) per model",
        bool((combos == 25).all()),
        f"unique fold identifiers = {combos.to_dict()}",
    )
    expected_repeats = {1, 2, 3, 4, 5}
    log.require(
        "cv: repeat identifiers",
        set(cv["repeat"]) == expected_repeats,
        f"repeats = {sorted(cv['repeat'].unique())}",
    )
    log.require(
        "cv: fold identifiers",
        set(cv["fold"]) == {1, 2, 3, 4, 5},
        f"folds = {sorted(cv['fold'].unique())}",
    )
    log.require(
        "cv: stratified fold sizes",
        bool(
            cv.groupby(["repeat", "fold"])["n_positive_validation"]
            .nunique()
            .eq(1)
            .all()
        )
        and int(cv["n_positive_validation"].max())
        - int(cv["n_positive_validation"].min())
        <= 1,
        (
            f"positives per fold range "
            f"{int(cv['n_positive_validation'].min())}-"
            f"{int(cv['n_positive_validation'].max())}, identical across models"
        ),
    )

    # ------------------------------------------------------------------
    # 3. OOF completeness, probabilities and pairing
    # ------------------------------------------------------------------
    split = load_split()
    X_train = split.X_train
    y_train = split.y_train
    train_index = np.asarray(X_train.index)

    log.require(
        "oof: row count",
        len(oof) == 5 * 5 * len(X_train),
        f"{len(oof)} rows (expected 102,200)",
    )

    coverage_failures = []
    for (model, repeat), group in oof.groupby(["model", "repeat"]):
        if len(group) != len(X_train) or group["row_index"].nunique() != len(X_train):
            coverage_failures.append((model, repeat, len(group), group["row_index"].nunique()))
    log.require(
        "oof: each row predicted once per repeat",
        not coverage_failures,
        (
            "every model x repeat covers 4,088 unique rows exactly once"
            if not coverage_failures
            else f"failures: {coverage_failures[:5]}"
        ),
    )

    label_mismatch = 0
    for row_index, group in oof.groupby("row_index"):
        expected = int(y_train.loc[row_index])
        if not bool((group["y_true"] == expected).all()):
            label_mismatch += 1
    log.require(
        "oof: labels match training partition",
        label_mismatch == 0,
        f"{label_mismatch} mismatching row indices",
    )

    proba = oof["y_proba"].to_numpy()
    log.require(
        "oof: probabilities finite",
        bool(np.isfinite(proba).all()),
        f"{len(proba)} probabilities, no NaN/inf",
    )
    log.require(
        "oof: probabilities within [0, 1]",
        bool(((proba >= 0.0) & (proba <= 1.0)).all()),
        f"range [{proba.min():.6f}, {proba.max():.6f}]",
    )

    # Paired folds: identical row sets per (repeat, fold) for every model.
    pairing_failures = []
    for (repeat, fold), group in oof.groupby(["repeat", "fold"]):
        sets = {
            model: frozenset(sub["row_index"].tolist())
            for model, sub in group.groupby("model")
        }
        if len(set(sets.values())) != 1:
            pairing_failures.append((repeat, fold))
    log.require(
        "cv: folds paired across models",
        not pairing_failures,
        (
            "all five models used identical validation rows in every fold"
            if not pairing_failures
            else f"unpaired folds: {pairing_failures[:5]}"
        ),
    )

    # Repeat independence: fold assignment vectors must differ between repeats.
    reference = oof[oof["model"] == models[0]]
    assignments = (
        reference.pivot_table(
            index="row_index", columns="repeat", values="fold", aggfunc="first"
        )
        .sort_index()
    )
    fingerprints = {
        int(repeat): hashlib.sha256(
            assignments[repeat].astype(np.int8).to_numpy().tobytes()
        ).hexdigest()[:16]
        for repeat in assignments.columns
    }
    log.require(
        "cv: repeats independently shuffled",
        len(set(fingerprints.values())) == 5,
        f"repeat fingerprints = {fingerprints}",
    )

    # ------------------------------------------------------------------
    # 4. Independent recomputation of fold-level metrics
    # ------------------------------------------------------------------
    recomputed_fold = []
    for (model, repeat, fold), group in oof.groupby(["model", "repeat", "fold"]):
        metrics = _metrics_from_predictions(group["y_true"], group["y_proba"])
        recomputed_fold.append(
            {"model": model, "repeat": repeat, "fold": fold, **metrics}
        )
    recomputed_fold_df = pd.DataFrame(recomputed_fold)
    merged = cv.merge(
        recomputed_fold_df,
        on=["model", "repeat", "fold"],
        suffixes=("_saved", "_recomputed"),
    )
    fold_diffs = {
        metric: _max_abs_diff(
            merged[f"{metric}_saved"], merged[f"{metric}_recomputed"]
        )
        for metric in METRIC_NAMES
    }
    worst_fold = max(fold_diffs.values())
    log.require(
        "metrics: fold-level recomputation",
        worst_fold <= TOLERANCE,
        f"max |difference| over 125 folds and 8 metrics = {worst_fold:.3e}",
    )

    # ------------------------------------------------------------------
    # 5. Independent recomputation of pooled-within-repeat metrics
    # ------------------------------------------------------------------
    pooled = pooled.copy()
    recomputed_pooled = []
    for (model, repeat), group in oof.groupby(["model", "repeat"]):
        metrics = _metrics_from_predictions(group["y_true"], group["y_proba"])
        recomputed_pooled.append(
            {"model": model, "repeat": repeat, **metrics}
        )
    recomputed_pooled_df = pd.DataFrame(recomputed_pooled)
    pooled_merge = pooled.merge(
        recomputed_pooled_df,
        on=["model", "repeat"],
        suffixes=("_saved", "_recomputed"),
    )
    pooled_diffs = {
        metric: _max_abs_diff(
            pooled_merge[f"{metric}_saved"], pooled_merge[f"{metric}_recomputed"]
        )
        for metric in METRIC_NAMES
    }
    worst_pooled = max(pooled_diffs.values())
    log.require(
        "metrics: pooled-within-repeat recomputation",
        worst_pooled <= TOLERANCE,
        f"max |difference| over 25 model-repeat pools = {worst_pooled:.3e}",
    )

    # ------------------------------------------------------------------
    # 6. Summary table consistent with fold and pooled results
    # ------------------------------------------------------------------
    expected_rows = []
    for model in models:
        fold_subset = cv[cv["model"] == model]
        pooled_subset = pooled[pooled["model"] == model]
        for metric in METRIC_NAMES:
            fold_values = fold_subset[metric].astype(float)
            expected_rows.append(
                {
                    "model": model,
                    "scope": "fold_level",
                    "metric": metric,
                    "mean": float(fold_values.mean()),
                    "sd": float(fold_values.std(ddof=1)),
                    "n_units": 25,
                }
            )
            repeat_means = (
                fold_subset.groupby("repeat")[metric].mean().astype(float)
            )
            expected_rows.append(
                {
                    "model": model,
                    "scope": "repeat_mean",
                    "metric": metric,
                    "mean": float(repeat_means.mean()),
                    "sd": float(repeat_means.std(ddof=1)),
                    "n_units": 5,
                }
            )
            pooled_values = pooled_subset[metric].astype(float)
            expected_rows.append(
                {
                    "model": model,
                    "scope": "pooled_within_repeat",
                    "metric": metric,
                    "mean": float(pooled_values.mean()),
                    "sd": float(pooled_values.std(ddof=1)),
                    "n_units": 5,
                }
            )
    expected = pd.DataFrame(expected_rows)
    compared = summary.merge(
        expected,
        on=["model", "scope", "metric"],
        suffixes=("_saved", "_expected"),
    )
    log.require(
        "summary: all rows matched",
        len(compared) == len(summary) == len(expected),
        f"summary rows = {len(summary)}, expected = {len(expected)}",
    )
    summary_diffs = {
        column: _max_abs_diff(
            compared[f"{column}_saved"], compared[f"{column}_expected"]
        )
        for column in ("mean", "sd")
    }
    worst_summary = max(summary_diffs.values())
    log.require(
        "summary: mean and SD recomputation",
        worst_summary <= TOLERANCE,
        f"max |difference| across 120 summary rows = {worst_summary:.3e}",
    )
    log.require(
        "summary: no missing metric values",
        not summary[["mean", "sd"]].isna().any().any(),
        "mean and sd are complete",
    )

    # ------------------------------------------------------------------
    # 7. Fold plan consistency
    # ------------------------------------------------------------------
    rebuilt_plan = build_fold_plan(X_train, y_train)
    plan_info = validate_fold_plan(rebuilt_plan, len(X_train))
    log.require(
        "fold plan: coverage and disjointness",
        plan_info["n_folds_total"] == 25
        and plan_info["n_repeats"] == 5
        and plan_info["n_splits"] == 5,
        "25 folds, 5 repeats, every repeat partitions the training rows exactly once",
    )
    plan_csv_ok = (
        len(fold_plan) == 25
        and set(fold_plan["repeat"]) == expected_repeats
        and set(fold_plan["fold"]) == {1, 2, 3, 4, 5}
    )
    log.require(
        "fold plan: saved table",
        plan_csv_ok,
        f"{len(fold_plan)} saved fold specifications",
    )

    # ------------------------------------------------------------------
    # 8. Test-set isolation
    # ------------------------------------------------------------------
    log.require(
        "isolation: test set locked",
        (not split.test.unlocked),
        "HeldOutTestSet remains locked",
    )
    oof_indices = set(oof["row_index"].unique())
    log.require(
        "isolation: OOF indices within training partition",
        oof_indices.issubset(set(int(i) for i in train_index)),
        f"{len(oof_indices)} unique row indices, all in the 4,088-row training partition",
    )
    log.require(
        "isolation: no test rows in OOF",
        len(oof_indices & set(int(i) for i in split.test_index)) == 0,
        "intersection between OOF row indices and held-out test indices is empty",
    )
    log.require(
        "isolation: manifest declares no test use",
        manifest.get("test_set", {}).get("test_set_used") is False,
        f"manifest test_set_used = {manifest.get('test_set', {}).get('test_set_used')}",
    )

    # ------------------------------------------------------------------
    # 9. Determinism: refit a single fold and compare probabilities
    # ------------------------------------------------------------------
    target_model, target_repeat, target_fold = "logistic_regression", 3, 2
    spec = next(
        item
        for item in rebuilt_plan
        if item.repeat == target_repeat and item.fold == target_fold
    )
    pipelines = build_baseline_pipelines()
    single = clone(pipelines[target_model])
    single.fit(
        X_train.iloc[spec.train_positions], y_train.iloc[spec.train_positions]
    )
    classes = list(single.classes_)
    refit_proba = single.predict_proba(
        X_train.iloc[spec.valid_positions]
    )[:, classes.index(POSITIVE_LABEL)]

    saved = (
        oof[
            (oof["model"] == target_model)
            & (oof["repeat"] == target_repeat)
            & (oof["fold"] == target_fold)
        ]
        .set_index("row_index")["y_proba"]
        .sort_index()
    )
    refit_series = pd.Series(
        refit_proba, index=X_train.index.values[spec.valid_positions]
    ).sort_index()
    determinism_diff = _max_abs_diff(saved, refit_series)
    log.require(
        "determinism: single-fold refit",
        determinism_diff <= 1e-12,
        (
            f"{target_model} repeat {target_repeat} fold {target_fold}: "
            f"max |probability difference| = {determinism_diff:.3e}"
        ),
    )

    # ------------------------------------------------------------------
    # 10. Manifest and data hash
    # ------------------------------------------------------------------
    raw_path = Path(manifest["data"]["path"])
    actual_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    log.require(
        "manifest: data hash unchanged",
        actual_hash == manifest["data"]["sha256"],
        f"sha256 = {actual_hash[:16]}...",
    )
    log.require(
        "manifest: split sizes",
        manifest["split"]["n_train"] == 4088
        and manifest["split"]["n_test"] == 1022,
        f"train={manifest['split']['n_train']}, test={manifest['split']['n_test']}",
    )
    log.require(
        "manifest: CV protocol",
        manifest["cv"]["n_splits"] == 5
        and manifest["cv"]["n_repeats"] == 5
        and manifest["cv"]["random_state"] == RANDOM_STATE
        and manifest["cv"]["threshold"] == DECISION_THRESHOLD,
        "5 folds x 5 repeats, random_state=42, threshold=0.50 fixed",
    )
    log.require(
        "manifest: no class balancing",
        manifest["class_balancing"]["applied"] is False,
        "no class weights, resampling, SMOTE or ADASYN",
    )

    passed = sum(1 for check in log.checks if check["status"] == "PASS")
    failed = len(log.checks) - passed
    result = {
        "verified_at_utc": utc_timestamp(),
        "run_id": manifest.get("run_id"),
        "n_checks": len(log.checks),
        "n_passed": passed,
        "n_failed": failed,
        "overall": "PASSED" if failed == 0 else "FAILED",
        "tolerance": TOLERANCE,
        "checks": log.checks,
        "max_abs_differences": {
            "fold_metrics": fold_diffs,
            "pooled_metrics": pooled_diffs,
            "summary": summary_diffs,
            "single_fold_refit": determinism_diff,
        },
    }
    write_json(result, tables_dir / "baseline_verification.json")

    print()
    print(f"VERIFICATION: {result['overall']} ({passed}/{len(log.checks)} checks)")
    return result


if __name__ == "__main__":
    result = verify_run()
    raise SystemExit(0 if result["overall"] == "PASSED" else 1)
