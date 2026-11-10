"""Evaluation utilities: metrics, stratified CV and reproducible logging.

Metric definitions
------------------
* ``roc_auc``            - ``sklearn.metrics.roc_auc_score``
* ``average_precision``  - ``sklearn.metrics.average_precision_score``.
  This is **average precision**, i.e. the step-wise (non-interpolated)
  area under the precision-recall curve. It is intentionally *not* the
  trapezoidal ``auc(recall, precision)`` that some libraries mislabel as
  "PR-AUC"; the two are not interchangeable.
* ``recall_sensitivity`` - TP / (TP + FN)
* ``specificity``        - TN / (TN + FP)
* ``precision``          - TP / (TP + FP)
* ``f1``                 - harmonic mean of precision and recall
* ``balanced_accuracy``  - mean of sensitivity and specificity
* ``brier_score``        - mean squared error of predicted probabilities
  (lower is better)

Cross-validation
----------------
``cross_validate_pipeline`` clones the *whole* pipeline for every fold and
fits it on that fold's training rows only. In-fold fitting is what keeps
median imputation, scaling and one-hot encoding free of validation leakage.

Results are written with an explicit ``scope`` column distinguishing
fold-level aggregates from pooled out-of-fold metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    CV_N_SPLITS,
    CV_SHUFFLE,
    DECISION_THRESHOLD,
    METRIC_LABELS,
    METRIC_NAMES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
)

LOGGER_NAME = "strokesure.evaluation"


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Return TN/FP/FN/TP counts with labels fixed to (0, 1)."""
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
    ).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def specificity_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """True-negative rate, TN / (TN + FP); 0.0 if there are no negatives."""
    counts = confusion_counts(y_true, y_pred)
    denominator = counts["tn"] + counts["fp"]
    return float(counts["tn"] / denominator) if denominator else 0.0


def compute_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Compute the full metric suite for one set of predictions.

    Parameters
    ----------
    y_true
        Binary ground-truth labels (0/1).
    y_proba
        Predicted probability of the positive class.
    threshold
        Fixed decision threshold; no threshold tuning is performed in the
        baseline stage.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_proba_arr = np.asarray(y_proba, dtype=float).ravel()

    if y_true_arr.shape != y_proba_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs "
            f"y_proba {y_proba_arr.shape}."
        )
    if not np.isfinite(y_proba_arr).all():
        raise ValueError("Predicted probabilities contain NaN or infinity.")
    if ((y_proba_arr < 0.0) | (y_proba_arr > 1.0)).any():
        raise ValueError("Predicted probabilities fall outside [0, 1].")

    y_pred = (y_proba_arr >= threshold).astype(int)
    counts = confusion_counts(y_true_arr, y_pred)

    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        # Average precision, NOT trapezoidal PR-AUC (see module docstring).
        "average_precision": float(
            average_precision_score(y_true_arr, y_proba_arr)
        ),
        "recall_sensitivity": float(
            recall_score(y_true_arr, y_pred, zero_division=0)
        ),
        "specificity": specificity_score(y_true_arr, y_pred),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "brier_score": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "threshold": float(threshold),
        "n_samples": int(y_true_arr.size),
        "n_positive": int(y_true_arr.sum()),
        "n_negative": int((y_true_arr == NEGATIVE_LABEL).sum()),
        "prevalence": float(y_true_arr.mean()),
        **counts,
    }
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    """Container for one model's cross-validation outcome."""

    model_name: str
    fold_metrics: pd.DataFrame
    oof_probabilities: np.ndarray
    oof_index: pd.Index
    summaries: pd.DataFrame
    pooled_metrics: dict[str, Any]
    fold_assignments: list[dict[str, Any]] = field(default_factory=list)
    n_features: int = 0
    cv_config: dict[str, Any] = field(default_factory=dict)


def make_stratified_cv(
    n_splits: int = CV_N_SPLITS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Deterministic stratified k-fold splitter."""
    return StratifiedKFold(
        n_splits=n_splits, shuffle=CV_SHUFFLE, random_state=random_state
    )


def _index_fingerprint(index: Iterable[Any]) -> str:
    """Stable short hash of a row-index set (for audit logs)."""
    payload = ",".join(str(i) for i in sorted(index))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: StratifiedKFold | None = None,
    threshold: float = DECISION_THRESHOLD,
) -> CVResult:
    """Run stratified CV, fitting preprocessing independently per fold.

    The pipeline is cloned for every fold, so the estimator *and* the
    preprocessor start unfitted each time. Predictions on the validation fold
    are always made by a model that has never seen those rows.
    """
    if cv is None:
        cv = make_stratified_cv()

    if not X.index.equals(y.index):
        raise ValueError("X and y indices are not aligned.")

    y = y.loc[X.index]

    oof_probabilities = np.full(len(X), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    n_features_seen: int | None = None

    for fold, (train_positions, valid_positions) in enumerate(
        cv.split(X, y), start=1
    ):
        X_tr = X.iloc[train_positions]
        X_va = X.iloc[valid_positions]
        y_tr = y.iloc[train_positions]
        y_va = y.iloc[valid_positions]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)

        classes = list(fold_pipeline.classes_)
        positive_column = classes.index(POSITIVE_LABEL)
        proba = fold_pipeline.predict_proba(X_va)[:, positive_column]

        if proba.shape[0] != len(X_va) or not np.isfinite(proba).all():
            raise RuntimeError(
                f"{model_name}: invalid validation probabilities in fold {fold}."
            )

        oof_probabilities[valid_positions] = proba

        preprocessor = fold_pipeline.named_steps["preprocessor"]
        fold_features = len(preprocessor.get_feature_names_out())
        if n_features_seen is None:
            n_features_seen = fold_features
        elif fold_features != n_features_seen:
            raise RuntimeError(
                f"{model_name}: inconsistent feature count across folds "
                f"({n_features_seen} vs {fold_features})."
            )

        metrics = compute_metrics(y_va, proba, threshold=threshold)

        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "n_train": int(len(X_tr)),
                "n_validation": int(len(X_va)),
                "n_positive_validation": int((y_va == POSITIVE_LABEL).sum()),
                "n_features": fold_features,
                **metrics,
            }
        )
        assignments.append(
            {
                "fold": fold,
                "n_train": int(len(X_tr)),
                "n_validation": int(len(X_va)),
                "train_index_fingerprint": _index_fingerprint(X_tr.index),
                "validation_index_fingerprint": _index_fingerprint(X_va.index),
            }
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            f"{model_name}: some rows received no out-of-fold prediction."
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summaries = summarise_folds(fold_metrics)
    pooled = compute_metrics(y, oof_probabilities, threshold=threshold)
    pooled["n_samples"] = int(len(y))

    return CVResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        oof_probabilities=oof_probabilities,
        oof_index=X.index,
        summaries=summaries,
        pooled_metrics=pooled,
        fold_assignments=assignments,
        n_features=int(n_features_seen or 0),
        cv_config={
            "n_splits": cv.get_n_splits(),
            "shuffle": CV_SHUFFLE,
            "random_state": RANDOM_STATE,
            "threshold": threshold,
        },
    )


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max of each metric across folds."""
    rows = []
    for metric in METRIC_NAMES:
        values = fold_metrics[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_folds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures (dependency-free SVG)
# ---------------------------------------------------------------------------
def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def save_confusion_matrix_svg(
    counts: dict[str, int], path: Path, title: str
) -> Path:
    """Render a 2x2 confusion matrix as a standalone SVG.

    Written by hand so that the project needs no plotting dependency.
    """
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    matrix = [[tn, fp], [fn, tp]]
    maximum = max(max(row) for row in matrix) or 1

    width, height = 380, 340
    cell, left, top = 120, 120, 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="Segoe UI, Arial" font-size="15" font-weight="600">'
        f"{_svg_escape(title)}</text>",
        f'<text x="{width / 2}" y="52" text-anchor="middle" '
        f'font-family="Segoe UI, Arial" font-size="11" fill="#555">'
        "threshold 0.50</text>",
    ]

    row_labels = ["Actual 0", "Actual 1"]
    col_labels = ["Pred 0", "Pred 1"]

    for col, label in enumerate(col_labels):
        x = left + col * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 10}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial" font-size="12">{label}</text>'
        )
    for row, label in enumerate(row_labels):
        y = top + row * cell + cell / 2 + 4
        parts.append(
            f'<text x="{left - 10}" y="{y}" text-anchor="end" '
            f'font-family="Segoe UI, Arial" font-size="12">{label}</text>'
        )

    for row in range(2):
        for col in range(2):
            value = matrix[row][col]
            intensity = value / maximum
            shade = int(244 - 150 * intensity)
            x = left + col * cell
            y = top + row * cell
            text_colour = "#ffffff" if intensity > 0.55 else "#222222"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" '
                f'fill="rgb({shade},{shade + 8},{shade + 18})" '
                f'stroke="#ffffff" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 4) / 2}" y="{y + (cell - 4) / 2 + 6}" '
                f'text-anchor="middle" font-family="Segoe UI, Arial" '
                f'font-size="20" font-weight="600" fill="{text_colour}">'
                f"{value}</text>"
            )

    parts.append(
        f'<text x="{width / 2}" y="{top + 2 * cell + 45}" '
        f'text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" '
        f'fill="#333">Sensitivity {tp}/{tp + fn} &nbsp; '
        f"Specificity {tn}/{tn + fp}</text>"
    )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_id(prefix: str) -> str:
    """Timestamped run identifier, e.g. ``smoke_20261102T144501Z``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def environment_snapshot() -> dict[str, Any]:
    """Versions and identifiers needed to reproduce a run."""
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    import threadpoolctl

    return {
        "timestamp_utc": utc_timestamp(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def configure_run_logger(
    run_id: str, log_path: Path, level: int = logging.INFO
) -> logging.Logger:
    """Create a file+console logger for one run, without duplicate handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"strokesure.run.{run_id}")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _json_default(value: Any) -> Any:
    """Convert numpy/pandas scalars so manifests stay strongly typed."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_cv_result(
    result: CVResult,
    run_id: str,
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    """Write fold metrics, summaries, pooled metrics, OOF predictions and SVG."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{run_id}_{result.model_name}"
    paths: dict[str, Path] = {}

    fold_path = tables_dir / f"{stem}_fold_metrics.csv"
    result.fold_metrics.to_csv(fold_path, index=False)
    paths["fold_metrics"] = fold_path

    summary = result.summaries.copy()
    summary.insert(0, "model", result.model_name)
    summary.insert(1, "scope", "fold_mean_sd")
    summary_path = tables_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    pooled_row = {
        "model": result.model_name,
        "scope": "pooled_out_of_fold",
        **{k: v for k, v in result.pooled_metrics.items()},
        "average_precision_definition": (
            "average_precision_score (step-wise PR area, not trapezoidal)"
        ),
    }
    pooled_path = tables_dir / f"{stem}_pooled_metrics.csv"
    pd.DataFrame([pooled_row]).to_csv(pooled_path, index=False)
    paths["pooled_metrics"] = pooled_path

    oof_path = tables_dir / f"{stem}_oof_predictions.csv"
    pd.DataFrame(
        {
            "row_index": result.oof_index,
            "y_proba": result.oof_probabilities,
        }
    ).to_csv(oof_path, index=False)
    paths["oof_predictions"] = oof_path

    counts = {
        "tn": result.pooled_metrics["tn"],
        "fp": result.pooled_metrics["fp"],
        "fn": result.pooled_metrics["fn"],
        "tp": result.pooled_metrics["tp"],
    }
    svg_path = figures_dir / f"{stem}_confusion_matrix_pooled.svg"
    save_confusion_matrix_svg(
        counts, svg_path, f"{result.model_name} - pooled OOF"
    )
    paths["confusion_figure"] = svg_path

    return paths


def confusion_matrix_table(results: Sequence[CVResult]) -> pd.DataFrame:
    """Long-format confusion counts for pooled and per-fold scopes."""
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "scope": "pooled_out_of_fold",
                "fold": "all",
                **{
                    key: int(result.pooled_metrics[key])
                    for key in ("tn", "fp", "fn", "tp")
                },
                "sensitivity": result.pooled_metrics["recall_sensitivity"],
                "specificity": result.pooled_metrics["specificity"],
            }
        )
        for _, row in result.fold_metrics.iterrows():
            rows.append(
                {
                    "model": result.model_name,
                    "scope": "fold",
                    "fold": int(row["fold"]),
                    "tn": int(row["tn"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tp": int(row["tp"]),
                    "sensitivity": float(row["recall_sensitivity"]),
                    "specificity": float(row["specificity"]),
                }
            )
    return pd.DataFrame(rows)


def build_manifest(
    run_id: str,
    step: str,
    results: Sequence[CVResult],
    data_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a JSON manifest describing exactly how a run was produced."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "step": step,
        "created_utc": utc_timestamp(),
        "environment": environment_snapshot(),
        "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD,
        "metric_definitions": METRIC_LABELS,
        "average_precision_note": (
            "average_precision_score computes a step-wise estimate of the area "
            "under the precision-recall curve. It is not trapezoidal PR-AUC "
            "and must not be compared with auc(recall, precision)."
        ),
        "data": data_metadata,
        "models": [],
    }
    for result in results:
        manifest["models"].append(
            {
                "model": result.model_name,
                "n_features": result.n_features,
                "cv": result.cv_config,
                "folds": result.fold_assignments,
                "fold_metrics": result.fold_metrics.to_dict(orient="records"),
                "pooled_metrics": result.pooled_metrics,
            }
        )
    if extra:
        manifest["extra"] = extra
    return manifest
