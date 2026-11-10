"""Fold-local repeated-CV runner for the imbalance experiment.

This module is intentionally *not* executed on the full dataset in the current
step.  It provides the reusable machinery and is exercised by the automated
tests on small synthetic data.

Guarantees
----------
* One :class:`src.preflight.FoldSpec` plan is materialised once and iterated by
  every model/strategy pipeline, so comparisons use identical folds.
* Each pipeline is cloned and fitted only on ``train_positions``.
* The sampler is invoked only during pipeline fitting.  Validation rows are
  never passed to ``fit_resample``; the imblearn pipeline skips sampler steps
  for prediction.
* Every fold record contains the sampler seed, CV seed, estimator seed, and the
  actual class counts before and after resampling.
* The locked test partition is never accepted by this API.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config import (
    CV_N_REPEATS,
    CV_N_SPLITS,
    DECISION_THRESHOLD,
    POSITIVE_LABEL,
    RANDOM_STATE,
)
from src.evaluation import compute_metrics
from src.imbalance_pipelines import RecordingSampler, class_counts
from src.preflight import FoldSpec, build_fold_plan


def _index_fingerprint(values: Sequence[Any]) -> str:
    """Stable short hash for a set of row positions or row indices."""
    array = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def validate_shared_fold_plan(
    plan: Sequence[FoldSpec],
    n_rows: int,
    *,
    n_splits: int | None = None,
    n_repeats: int | None = None,
) -> dict[str, Any]:
    """Validate coverage, disjointness and repeat structure of a fold plan.

    Unlike :func:`src.preflight.validate_fold_plan`, this helper accepts an
    arbitrary number of splits/repeats, which keeps the unit tests small while
    preserving the same invariants used by the full 5x5 protocol.
    """
    if not plan:
        raise ValueError("Fold plan is empty.")

    positions = np.arange(n_rows)
    repeats: dict[int, list[FoldSpec]] = {}
    seen: set[tuple[int, int]] = set()

    for spec in plan:
        key = (int(spec.repeat), int(spec.fold))
        if key in seen:
            raise ValueError(f"Duplicate fold identifier {key}.")
        seen.add(key)
        repeats.setdefault(int(spec.repeat), []).append(spec)

        train = np.asarray(spec.train_positions, dtype=int)
        valid = np.asarray(spec.valid_positions, dtype=int)
        if len(train) == 0 or len(valid) == 0:
            raise ValueError(f"{key}: empty train or validation partition.")
        if len(np.intersect1d(train, valid)):
            raise ValueError(f"{key}: train/validation overlap.")
        if len(train) + len(valid) != n_rows:
            raise ValueError(f"{key}: train + validation != n_rows.")
        combined = np.sort(np.concatenate([train, valid]))
        if not np.array_equal(combined, positions):
            raise ValueError(f"{key}: train/validation do not partition rows.")

    repeat_ids = sorted(repeats)
    expected_repeats = list(range(1, len(repeats) + 1))
    if n_repeats is not None:
        expected_repeats = list(range(1, n_repeats + 1))
    if repeat_ids != expected_repeats:
        raise ValueError(
            f"Expected repeats {expected_repeats}, found {repeat_ids}."
        )

    fold_counts = {repeat: len(specs) for repeat, specs in repeats.items()}
    if len(set(fold_counts.values())) != 1:
        raise ValueError(
            f"Repeats have inconsistent fold counts: {fold_counts}."
        )

    fingerprints: dict[int, str] = {}
    for repeat, specs in repeats.items():
        if n_splits is not None and len(specs) != n_splits:
            raise ValueError(
                f"Repeat {repeat} has {len(specs)} folds, expected {n_splits}."
            )
        covered = np.concatenate(
            [np.asarray(spec.valid_positions, dtype=int) for spec in specs]
        )
        if len(covered) != n_rows or not np.array_equal(
            np.sort(covered), positions
        ):
            raise ValueError(
                f"Repeat {repeat}: validation folds do not partition rows "
                "exactly once."
            )
        assignment = np.empty(n_rows, dtype=int)
        for spec in specs:
            assignment[np.asarray(spec.valid_positions, dtype=int)] = int(
                spec.fold
            )
        fingerprints[repeat] = hashlib.sha256(
            assignment.astype(np.int8).tobytes()
        ).hexdigest()[:16]

    if len(fingerprints) > 1 and len(set(fingerprints.values())) != len(
        fingerprints
    ):
        raise ValueError("Repeats do not produce distinct fold assignments.")

    return {
        "n_rows": int(n_rows),
        "n_folds_total": int(len(plan)),
        "n_splits": int(
            n_splits
            if n_splits is not None
            else max(len(specs) for specs in repeats.values())
        ),
        "n_repeats": int(
            n_repeats if n_repeats is not None else len(repeats)
        ),
        "repeat_assignment_fingerprints": fingerprints,
        "pairing": (
            "All pipelines iterate the identical FoldSpec list; per-fold "
            "train/validation index fingerprints are recorded."
        ),
    }


def run_imbalance_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Mapping[str, Pipeline],
    fold_plan: Sequence[FoldSpec] | None = None,
    *,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    random_state: int = RANDOM_STATE,
    threshold: float = DECISION_THRESHOLD,
    fold_observer: Callable[..., None] | None = None,
    validate_plan: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit every pipeline over one shared fold plan and record fold metadata.

    Parameters
    ----------
    X, y
        Training partition only.  Do not pass the held-out test set.
    pipelines
        Mapping from ``"model__strategy"`` to an unfitted pipeline.
    fold_plan
        Optional pre-materialised plan.  If omitted, one is built with
        :func:`src.preflight.build_fold_plan` using ``n_splits``, ``n_repeats``
        and ``random_state``.
    n_splits, n_repeats, random_state
        Used only when ``fold_plan`` is not supplied.
    fold_observer
        Optional callable invoked after each successful fold fit and
        prediction.  It receives keyword arguments for the pipeline key,
        model/strategy names, :class:`FoldSpec`, fitted pipeline, sampler step,
        train/validation data, validation probabilities and metrics.  This
        hook exists for auditable pilot checks; exceptions propagate.

    Returns
    -------
    cv : pandas.DataFrame
        One row per pipeline/fold with metrics, seeds and sampling counts.
    oof : pandas.DataFrame
        Out-of-fold predictions restricted to the supplied training rows.
    metadata : dict
        Fold-plan validation, seeds and pairing information.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}.")
    if not pipelines:
        raise ValueError("At least one pipeline is required.")

    if fold_plan is None:
        fold_plan = build_fold_plan(
            X,
            y,
            n_splits=CV_N_SPLITS if n_splits is None else n_splits,
            n_repeats=CV_N_REPEATS if n_repeats is None else n_repeats,
            random_state=random_state,
        )
        plan_info = validate_shared_fold_plan(
            fold_plan,
            n_rows=len(X),
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
    elif validate_plan:
        plan_info = validate_shared_fold_plan(
            fold_plan,
            n_rows=len(X),
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
    else:
        # Per-fold capture (src.imbalance_outputs) validates the complete
        # shared plan once and then re-uses individual FoldSpec slices; the
        # structural partition check is skipped here on purpose.
        single_specs = fold_plan if len(fold_plan) == 1 else None
        if single_specs is None:
            raise ValueError(
                "validate_plan=False is only supported for one-fold plans."
            )
        plan_info = {
            "n_rows": int(len(X)),
            "n_folds_total": 1,
            "n_splits": 1,
            "n_repeats": 1,
            "repeat_assignment_fingerprints": {},
            "pairing": (
                "Single fold re-used from an already validated shared plan."
            ),
        }

    fold_records: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    reference_validation_counts: dict[tuple[int, int], dict[int, int]] = {}

    for pipeline_key, pipeline in pipelines.items():
        model_name, separator, strategy_name = pipeline_key.partition("__")
        if not separator:
            model_name, strategy_name = pipeline_key, ""

        for spec in fold_plan:
            X_train = X.iloc[spec.train_positions]
            X_validation = X.iloc[spec.valid_positions]
            y_train = y.iloc[spec.train_positions]
            y_validation = y.iloc[spec.valid_positions]

            if y_train.nunique() < 2 or y_validation.nunique() < 2:
                raise ValueError(
                    f"{pipeline_key} repeat {spec.repeat} fold {spec.fold}: "
                    "a fold contains only one class."
                )

            fit_pipeline = clone(pipeline)
            fit_started = time.perf_counter()
            fit_pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_started

            sampler_step = fit_pipeline.named_steps.get("sampler")
            if sampler_step is not None and not isinstance(
                sampler_step, RecordingSampler
            ):
                raise TypeError(
                    f"{pipeline_key}: resampling pipelines must use "
                    "RecordingSampler so before/after class counts can be "
                    "audited."
                )

            if sampler_step is not None:
                sampler_meta = sampler_step.sampling_metadata()
                before_counts = dict(sampler_meta["class_counts_before"])
                after_counts = dict(sampler_meta["class_counts_after"])
                n_train_before = int(sampler_meta["n_samples_before"])
                n_train_after = int(sampler_meta["n_samples_after"])
                sampler_name = str(sampler_meta["sampler"])
                sampler_random_state = sampler_meta["sampler_random_state"]
                n_fit_resample_calls = int(
                    sampler_meta["n_fit_resample_calls"]
                )
            else:
                before_counts = class_counts(y_train)
                after_counts = dict(before_counts)
                n_train_before = int(len(y_train))
                n_train_after = int(len(y_train))
                sampler_name = None
                sampler_random_state = None
                n_fit_resample_calls = 0

            if before_counts != class_counts(y_train):
                raise RuntimeError(
                    f"{pipeline_key} repeat {spec.repeat} fold {spec.fold}: "
                    "sampler did not receive exactly the training labels."
                )
            if n_train_before != len(y_train):
                raise RuntimeError(
                    f"{pipeline_key} repeat {spec.repeat} fold {spec.fold}: "
                    "sampler saw a row count different from the training fold."
                )

            validation_counts_before = class_counts(y_validation)
            previous = reference_validation_counts.setdefault(
                (int(spec.repeat), int(spec.fold)),
                dict(validation_counts_before),
            )
            if previous != validation_counts_before:
                raise RuntimeError(
                    f"{pipeline_key} repeat {spec.repeat} fold {spec.fold}: "
                    "validation class distribution differs across pipelines."
                )

            classes = list(fit_pipeline.classes_)
            predict_started = time.perf_counter()
            probabilities = fit_pipeline.predict_proba(X_validation)[
                :, classes.index(POSITIVE_LABEL)
            ]
            predict_seconds = time.perf_counter() - predict_started

            # A prediction call must never invoke the sampler again.  The
            # wrapper's call counter and recorded counts are the audit trail.
            if sampler_step is not None:
                if (
                    sampler_step.n_fit_resample_calls_
                    != n_fit_resample_calls
                ):
                    raise RuntimeError(
                        f"{pipeline_key} repeat {spec.repeat} fold "
                        f"{spec.fold}: sampler was invoked after fitting."
                    )
                if (
                    sampler_step.n_samples_before_ != len(y_train)
                    or sampler_step.n_samples_before_
                    != n_train_before
                ):
                    raise RuntimeError(
                        f"{pipeline_key} repeat {spec.repeat} fold "
                        f"{spec.fold}: sampler input changed during prediction."
                    )

            validation_counts_after = class_counts(y_validation)
            if validation_counts_after != validation_counts_before:
                raise RuntimeError(
                    f"{pipeline_key} repeat {spec.repeat} fold {spec.fold}: "
                    "validation labels were altered."
                )

            metrics = compute_metrics(
                y_validation.to_numpy(),
                probabilities,
                threshold=threshold,
            )
            preprocessor = fit_pipeline.named_steps["preprocessor"]
            n_features = len(preprocessor.get_feature_names_out())
            estimator = fit_pipeline.named_steps["classifier"]

            if fold_observer is not None:
                fold_observer(
                    pipeline_key=pipeline_key,
                    model_name=model_name,
                    strategy_name=strategy_name,
                    spec=spec,
                    pipeline=fit_pipeline,
                    sampler_step=sampler_step,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    probabilities=probabilities,
                    metrics=metrics,
                    n_features=int(n_features),
                )

            fold_records.append(
                {
                    "pipeline": pipeline_key,
                    "model": model_name,
                    "strategy": strategy_name,
                    "repeat": int(spec.repeat),
                    "fold": int(spec.fold),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "n_train_before_sampling": n_train_before,
                    "n_train_after_sampling": n_train_after,
                    "n_validation": int(len(y_validation)),
                    "sampling_applied": sampler_step is not None,
                    "sampler": sampler_name,
                    "sampler_random_state": sampler_random_state,
                    "estimator_random_state": estimator.get_params(
                        deep=False
                    ).get("random_state", None),
                    "cv_random_state": int(random_state),
                    "n_fit_resample_calls": n_fit_resample_calls,
                    "sampling_class_counts_before": before_counts,
                    "sampling_class_counts_after": after_counts,
                    "validation_class_counts_before": validation_counts_before,
                    "validation_class_counts_after": validation_counts_after,
                    "train_index_fingerprint": _index_fingerprint(
                        X.index.values[spec.train_positions]
                    ),
                    "validation_index_fingerprint": _index_fingerprint(
                        X.index.values[spec.valid_positions]
                    ),
                    "n_features": int(n_features),
                    **metrics,
                }
            )

            oof_parts.append(
                pd.DataFrame(
                    {
                        "pipeline": pipeline_key,
                        "repeat": int(spec.repeat),
                        "fold": int(spec.fold),
                        "row_index": X.index.values[spec.valid_positions],
                        "y_true": y.values[spec.valid_positions],
                        "y_proba": probabilities,
                    }
                )
            )

    cv = pd.DataFrame(fold_records)
    oof = pd.concat(oof_parts, ignore_index=True)

    if len(cv) != len(pipelines) * len(fold_plan):
        raise RuntimeError(
            f"Unexpected fold record count: {len(cv)} != "
            f"{len(pipelines)} x {len(fold_plan)}."
        )
    if not set(oof["row_index"]).issubset(set(X.index)):
        raise RuntimeError("Out-of-fold predictions contain non-training rows.")

    metadata = {
        "n_pipelines": int(len(pipelines)),
        "n_folds_per_pipeline": int(len(fold_plan)),
        "n_fold_records": int(len(cv)),
        "n_oof_predictions": int(len(oof)),
        "cv_random_state": int(random_state),
        "threshold": float(threshold),
        "fold_plan": plan_info,
        "pairing": plan_info["pairing"],
        "validation_resampling": False,
        "test_set_used": False,
        "seeds": {
            "cv_random_state": int(random_state),
            "sampler_random_state_values": sorted(
                {
                    row["sampler_random_state"]
                    for row in fold_records
                    if row["sampler_random_state"] is not None
                },
                key=str,
            ),
            "estimator_random_state_values": sorted(
                {
                    row["estimator_random_state"]
                    for row in fold_records
                    if row["estimator_random_state"] is not None
                },
                key=str,
            ),
        },
    }
    return cv, oof, metadata


__all__ = ["run_imbalance_cv", "validate_shared_fold_plan"]
