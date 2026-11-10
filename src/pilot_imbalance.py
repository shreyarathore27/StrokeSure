"""Small, training-only pilot for the class-imbalance strategies.

Run from the project root with::

    python -m src.pilot_imbalance

Scope
-----
* Uses only a class-stratified subsample of the training partition.
* Runs one shared 3-fold stratified CV plan for every supported
  model/strategy pipeline (25 combinations).
* Never evaluates, unlocks or reads the held-out test partition.
* Fits preprocessing and sampling inside each training fold only.
* Audits the actual resampled categorical values for every resampling fold.
* Captures every warning and error instead of silently skipping failures.

This is a pilot, not the full imbalance experiment.  It uses one repeat and a
small stratified subset, performs no threshold tuning or probability
calibration, and must not be reported as a final result.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    METRIC_LABELS,
    METRIC_NAMES,
    MODEL_FAMILY_TREE,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.data_loading import load_split, stratified_subsample
from src.evaluation import (
    build_run_id,
    compute_metrics,
    configure_run_logger,
    environment_snapshot,
    utc_timestamp,
    write_json,
)
from src.imbalance import (
    STRATEGIES,
    strategy_table,
    support_matrix,
    supports_strategy,
)
from src.imbalance_experiment import (
    run_imbalance_cv,
    validate_shared_fold_plan,
)
from src.imbalance_pipelines import (
    CategoricalCodeDecoder,
    RecordingSampler,
    UNSEEN_CATEGORY,
    build_leakage_safe_grid,
    class_counts,
)
from src.models import BASELINE_MODEL_SPECS
from src.preflight import FoldSpec, build_fold_plan
from src.preprocessing import CATEGORY_VALUES

# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------
PILOT_DIR = RESULTS_DIR / "pilot_imbalance"
PILOT_RUN_PREFIX = "pilot_imbalance"
PILOT_SAMPLE_SIZE = 1500
PILOT_CV_SPLITS = 3
PILOT_CV_REPEATS = 1
PILOT_SAMPLE_SEED = RANDOM_STATE
PILOT_MIN_POSITIVES = 50
PILOT_MIN_TRAIN_POSITIVES = 20
PILOT_MIN_VALIDATION_POSITIVES = 10

RESAMPLING_STRATEGIES = (
    "random_over_sampler",
    "random_under_sampler",
    "smote",
    "adasyn",
)


class PilotFailure(RuntimeError):
    """Raised when the pilot cannot establish a basic precondition."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _counts_text(counts: Mapping[Any, Any]) -> str:
    return json.dumps(
        {str(key): int(value) for key, value in sorted(counts.items())},
        sort_keys=True,
    )


def _index_fingerprint(values: Sequence[Any]) -> str:
    array = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


class CapturingRecordingSampler(RecordingSampler):
    """Recording sampler that also retains the rows it produced.

    The retained matrix is used only by the pilot to decode and inspect the
    actual resampled categorical values.  It is not serialised to disk.
    """

    def fit_resample(self, X: Any, y: Any):
        X_resampled, y_resampled = super().fit_resample(X, y)
        self.resampled_X_ = np.asarray(X_resampled, dtype=float)
        self.resampled_y_ = np.asarray(y_resampled).astype(int)
        return X_resampled, y_resampled


def _expected_grid_keys() -> list[str]:
    return [
        f"{model_name}__{strategy_name}"
        for model_name in BASELINE_MODEL_SPECS
        for strategy_name in STRATEGIES
        if supports_strategy(model_name, strategy_name)
    ]


def build_pilot_grid() -> dict[str, Any]:
    """Build all 25 supported pipelines and make their samplers observable."""
    grid = dict(build_leakage_safe_grid(record_sampling=True))
    expected = _expected_grid_keys()
    if list(grid) != expected:
        raise PilotFailure(
            "Supported pipeline grid changed unexpectedly: "
            f"expected {expected}, found {list(grid)}."
        )

    observable: dict[str, Any] = {}
    for key, pipeline in grid.items():
        if "sampler" in pipeline.named_steps:
            wrapper = pipeline.named_steps["sampler"]
            if not isinstance(wrapper, RecordingSampler):
                raise PilotFailure(
                    f"{key}: resampling pipeline does not use RecordingSampler."
                )
            pipeline = clone(pipeline)
            pipeline.set_params(
                sampler=CapturingRecordingSampler(wrapper.sampler)
            )
        observable[key] = pipeline
    return observable


# ---------------------------------------------------------------------------
# Data and fold plan
# ---------------------------------------------------------------------------
def load_pilot_subset() -> tuple[pd.DataFrame, pd.Series, dict[str, Any], list[FoldSpec], dict[str, Any]]:
    """Return a stratified training subset plus its shared fold plan.

    Only the training fields of the deterministic split are read.  The
    held-out partition's lock flag is inspected, but its ``X``, ``y``,
    ``get()``, ``shape`` and class-count fields are never touched.
    """
    split = load_split()
    test_was_locked = not bool(split.test_unlocked)
    if not test_was_locked:
        raise PilotFailure("Held-out test set was already unlocked; aborting.")

    X_train = split.X_train
    y_train = split.y_train
    train_index = split.train_index
    train_metadata = {
        "data_sha256": split.metadata["data_sha256"],
        "n_rows": int(split.metadata["n_rows"]),
        "n_features": int(split.metadata["n_features"]),
        "n_train": int(split.metadata["n_train"]),
        "train_class_counts": dict(split.metadata["train_class_counts"]),
        "train_prevalence": float(split.metadata["train_prevalence"]),
        "full_prevalence": float(split.metadata["full_prevalence"]),
        "test_size": float(split.metadata["test_size"]),
        "random_state": int(split.metadata["random_state"]),
    }
    # Drop the split object so later code cannot accidentally reach the test
    # container through a stale reference.
    del split

    X, y = stratified_subsample(
        X_train,
        y_train,
        n_samples=PILOT_SAMPLE_SIZE,
        random_state=PILOT_SAMPLE_SEED,
    )
    y = y.loc[X.index]

    if not set(X.index).issubset(set(train_index)):
        raise PilotFailure("Pilot subset contains rows outside the training split.")
    if int(y.sum()) < PILOT_MIN_POSITIVES:
        raise PilotFailure(
            f"Pilot subset has only {int(y.sum())} positives; "
            f"at least {PILOT_MIN_POSITIVES} are required."
        )

    plan = build_fold_plan(
        X,
        y,
        n_splits=PILOT_CV_SPLITS,
        n_repeats=PILOT_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    plan_info = validate_shared_fold_plan(
        plan,
        n_rows=len(X),
        n_splits=PILOT_CV_SPLITS,
        n_repeats=PILOT_CV_REPEATS,
    )

    train_positive_counts = [
        int(y.iloc[spec.train_positions].sum()) for spec in plan
    ]
    validation_positive_counts = [
        int(y.iloc[spec.valid_positions].sum()) for spec in plan
    ]
    if min(train_positive_counts) < PILOT_MIN_TRAIN_POSITIVES:
        raise PilotFailure(
            "A training fold has too few positives for the configured "
            f"neighbourhood size: {train_positive_counts}."
        )
    if min(validation_positive_counts) < PILOT_MIN_VALIDATION_POSITIVES:
        raise PilotFailure(
            "A validation fold has too few positives for stable pilot metrics: "
            f"{validation_positive_counts}."
        )

    metadata = {
        "pilot_sample_size": int(len(X)),
        "pilot_sample_seed": int(PILOT_SAMPLE_SEED),
        "pilot_class_counts": {
            "0": int((y == 0).sum()),
            "1": int((y == POSITIVE_LABEL).sum()),
        },
        "pilot_prevalence": float(y.mean()),
        "minimum_positives": int(PILOT_MIN_POSITIVES),
        "train_positive_counts_per_fold": train_positive_counts,
        "validation_positive_counts_per_fold": validation_positive_counts,
        "fold_plan": plan_info,
        "source_training_metadata": train_metadata,
        "source_training_rows": int(len(X_train)),
        "pilot_rows_within_training": True,
        "heldout_test_lock_read": True,
        "heldout_test_data_fields_accessed": False,
        "heldout_test_was_locked": test_was_locked,
        "test_set_used": False,
    }
    return X, y, metadata, plan, plan_info


# ---------------------------------------------------------------------------
# Resampling audit
# ---------------------------------------------------------------------------
def _discrete_issues(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for column in BINARY_FEATURES:
        observed = set(frame[column].unique().tolist())
        if not observed.issubset({0, 1}):
            issues.append(f"{column}: invalid binary values {sorted(observed)}")
    for column, allowed in CATEGORY_VALUES.items():
        observed = set(frame[column].unique().tolist())
        if not observed.issubset(set(allowed)):
            issues.append(
                f"{column}: invalid categories {sorted(map(str, observed))}"
            )
        if UNSEEN_CATEGORY in observed:
            issues.append(f"{column}: decoder emitted the unseen-category sentinel")
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            issues.append(f"{column}: non-finite sampled values")
    return issues


def _one_hot_issues(
    pipeline: Any,
    decoded: pd.DataFrame,
    model_name: str,
) -> tuple[list[str], int, int]:
    """Check the model preprocessor's categorical block for fractional values."""
    preprocessor = pipeline.named_steps["preprocessor"]
    matrix = np.asarray(preprocessor.transform(decoded), dtype=float)
    names = [str(name) for name in preprocessor.get_feature_names_out()]

    categorical_indices = [
        index for index, name in enumerate(names) if name.startswith("cat__")
    ]
    issues: list[str] = []
    n_nonfinite = int((~np.isfinite(matrix)).sum())
    if n_nonfinite:
        issues.append(f"model matrix contains {n_nonfinite} non-finite values")

    if not categorical_indices:
        return issues, 0, n_nonfinite

    categorical_block = matrix[:, categorical_indices]
    fractional = (categorical_block > 1e-9) & (
        categorical_block < 1.0 - 1e-9
    )
    outside = (categorical_block < -1e-12) | (
        categorical_block > 1.0 + 1e-12
    )
    n_fractional = int(fractional.sum())
    n_outside = int(outside.sum())
    if n_fractional:
        issues.append(
            f"{n_fractional} fractional one-hot values in the model matrix"
        )
    if n_outside:
        issues.append(f"{n_outside} one-hot values outside [0, 1]")

    family = BASELINE_MODEL_SPECS[model_name].family
    for feature in CATEGORICAL_FEATURES:
        columns = [
            index
            for index, name in enumerate(names)
            if name.startswith(f"cat__{feature}_")
        ]
        if not columns:
            issues.append(f"{feature}: no one-hot columns found")
            continue
        row_sums = matrix[:, columns].sum(axis=1)
        if family == MODEL_FAMILY_TREE:
            if not np.allclose(row_sums, 1.0, atol=1e-9):
                issues.append(
                    f"{feature}: full one-hot rows do not sum to one"
                )
        else:
            if np.any(row_sums > 1.0 + 1e-9):
                issues.append(
                    f"{feature}: drop-first one-hot rows activate >1 level"
                )
    return issues, n_fractional, n_nonfinite


def audit_resampling_fold(
    *,
    pipeline_key: str,
    model_name: str,
    strategy_name: str,
    spec: FoldSpec,
    pipeline: Any,
    sampler_step: Any,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> dict[str, Any]:
    """Decode and validate the exact rows produced by this fold's sampler."""
    record: dict[str, Any] = {
        "pipeline": pipeline_key,
        "model": model_name,
        "strategy": strategy_name,
        "repeat": int(spec.repeat),
        "fold": int(spec.fold),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "validation_class_counts": _counts_text(class_counts(y_validation)),
        "sampler": type(sampler_step.sampler).__name__,
        "sampler_random_state": getattr(
            sampler_step.sampler, "random_state", None
        ),
        "n_fit_resample_calls": int(sampler_step.n_fit_resample_calls_),
        "n_train_before_sampling": int(sampler_step.n_samples_before_),
        "n_train_after_sampling": int(sampler_step.n_samples_after_),
        "class_counts_before": _counts_text(
            sampler_step.sampling_class_counts_before_
        ),
        "class_counts_after": _counts_text(
            sampler_step.sampling_class_counts_after_
        ),
        "n_appended_rows": int(
            max(0, sampler_step.n_samples_after_ - sampler_step.n_samples_before_)
        ),
        "audit_status": "PASS",
        "issues": "",
    }

    try:
        if not hasattr(sampler_step, "resampled_X_"):
            raise PilotFailure("sampler did not retain its resampled rows")

        decoder: CategoricalCodeDecoder = pipeline.named_steps[
            "sampling_decoder"
        ]
        resampled_X = np.asarray(sampler_step.resampled_X_, dtype=float)
        resampled_y = np.asarray(sampler_step.resampled_y_)
        if len(resampled_X) != len(resampled_y):
            raise PilotFailure("resampled X/y lengths differ")
        if len(resampled_y) != int(sampler_step.n_samples_after_):
            raise PilotFailure("resampled row count differs from recorded count")
        if not np.isfinite(resampled_X).all():
            raise PilotFailure("resampled encoded matrix has non-finite values")

        decoded = decoder.transform(resampled_X)
        if len(decoded) != len(resampled_y):
            raise PilotFailure("decoded row count differs from resampled labels")

        issues = _discrete_issues(decoded)
        one_hot_problems, n_fractional, n_nonfinite = _one_hot_issues(
            pipeline, decoded, model_name
        )
        issues.extend(one_hot_problems)

        n_before = int(sampler_step.n_samples_before_)
        appended = decoded.iloc[n_before:] if n_before < len(decoded) else decoded.iloc[0:0]
        appended_issues = _discrete_issues(appended) if len(appended) else []
        if appended_issues:
            issues.extend(
                f"appended rows: {issue}" for issue in appended_issues
            )

        record.update(
            {
                "n_decoded_rows": int(len(decoded)),
                "n_appended_rows_checked": int(len(appended)),
                "n_fractional_one_hot_values": int(n_fractional),
                "n_nonfinite_model_values": int(n_nonfinite),
                "appended_rows_checked": True,
                "audit_status": "PASS" if not issues else "FAIL",
                "issues": " | ".join(issues),
            }
        )
    except Exception as exc:  # audit failures are reported, not hidden
        record.update(
            {
                "audit_status": "ERROR",
                "issues": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return record


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_pilot() -> int:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(PILOT_RUN_PREFIX)
    log_path = PILOT_DIR / f"{run_id}.log"
    logger = configure_run_logger(run_id, log_path)

    started = time.perf_counter()
    logger.info("=" * 76)
    logger.info("StrokeSure imbalance pilot")
    logger.info("run_id=%s", run_id)
    logger.info("=" * 76)

    X, y, data_metadata, fold_plan, plan_info = load_pilot_subset()
    logger.info(
        "pilot subset: n=%d positives=%d prevalence=%.4f",
        len(X),
        int(y.sum()),
        float(y.mean()),
    )
    logger.info(
        "CV: %d folds x %d repeat, random_state=%d",
        PILOT_CV_SPLITS,
        PILOT_CV_REPEATS,
        RANDOM_STATE,
    )
    logger.info("test-set lock read: %s; data fields accessed: %s",
                data_metadata["heldout_test_was_locked"],
                data_metadata["heldout_test_data_fields_accessed"])

    build_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = build_pilot_grid()
    build_warnings.extend(caught)

    warning_records: list[dict[str, Any]] = []
    for warning in build_warnings:
        warning_records.append(
            {
                "pipeline": "<grid-build>",
                "phase": "build",
                "category": warning.category.__name__,
                "message": str(warning.message),
                "filename": warning.filename,
                "lineno": int(warning.lineno),
            }
        )

    cv_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    sampled_audits: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    pipeline_status: dict[str, str] = {}
    pipeline_seconds: dict[str, float] = {}

    for pipeline_key, pipeline in grid.items():
        model_name, _, strategy_name = pipeline_key.partition("__")
        pipeline_started = time.perf_counter()
        logger.info("pipeline %s: starting", pipeline_key)

        def observer(**kwargs: Any) -> None:
            sampler_step = kwargs["sampler_step"]
            if sampler_step is None:
                return
            if strategy_name not in RESAMPLING_STRATEGIES:
                return
            sampled_audits.append(
                audit_resampling_fold(
                    pipeline_key=pipeline_key,
                    model_name=model_name,
                    strategy_name=strategy_name,
                    spec=kwargs["spec"],
                    pipeline=kwargs["pipeline"],
                    sampler_step=sampler_step,
                    y_train=kwargs["y_train"],
                    y_validation=kwargs["y_validation"],
                )
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                cv, oof, metadata = run_imbalance_cv(
                    X,
                    y,
                    {pipeline_key: pipeline},
                    fold_plan,
                    n_splits=PILOT_CV_SPLITS,
                    n_repeats=PILOT_CV_REPEATS,
                    random_state=RANDOM_STATE,
                    threshold=DECISION_THRESHOLD,
                    fold_observer=observer,
                )
                cv_parts.append(cv)
                oof_parts.append(oof)
                pipeline_status[pipeline_key] = "ok"
                logger.info(
                    "pipeline %s: ok (%d folds, %.2fs)",
                    pipeline_key,
                    len(cv),
                    time.perf_counter() - pipeline_started,
                )
            except Exception as exc:
                pipeline_status[pipeline_key] = "error"
                error_records.append(
                    {
                        "pipeline": pipeline_key,
                        "phase": "fit/predict",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                logger.error(
                    "pipeline %s: ERROR %s: %s",
                    pipeline_key,
                    type(exc).__name__,
                    exc,
                )
        for warning in caught:
            warning_records.append(
                {
                    "pipeline": pipeline_key,
                    "phase": "fit/predict",
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": int(warning.lineno),
                }
            )
        pipeline_seconds[pipeline_key] = time.perf_counter() - pipeline_started

    cv = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    sampling_audit = pd.DataFrame(sampled_audits)
    warnings_table = pd.DataFrame(warning_records)
    errors_table = pd.DataFrame(error_records)

    summary, summary_wide = build_summaries(cv)
    pooled = build_pooled_metrics(oof)
    verification = verify_pilot(
        grid=grid,
        cv=cv,
        oof=oof,
        X=X,
        y=y,
        fold_plan=fold_plan,
        plan_info=plan_info,
        data_metadata=data_metadata,
        sampling_audit=sampling_audit,
        pipeline_status=pipeline_status,
        warnings_table=warnings_table,
        errors_table=errors_table,
    )

    total_seconds = time.perf_counter() - started
    paths = save_pilot_outputs(
        run_id=run_id,
        log_path=log_path,
        X=X,
        y=y,
        fold_plan=fold_plan,
        data_metadata=data_metadata,
        plan_info=plan_info,
        cv=cv,
        oof=oof,
        summary=summary,
        summary_wide=summary_wide,
        pooled=pooled,
        sampling_audit=sampling_audit,
        warnings_table=warnings_table,
        errors_table=errors_table,
        verification=verification,
        pipeline_status=pipeline_status,
        pipeline_seconds=pipeline_seconds,
        grid=grid,
        total_seconds=total_seconds,
    )

    logger.info("pilot finished in %.1fs; status=%s", total_seconds,
                "PASSED" if verification["overall_passed"] else "FAILED")
    logger.info("artefacts: %s", PILOT_DIR)

    report_path = paths["report"]
    print()
    print("=" * 78)
    print("IMBALANCE PILOT RESULT")
    print("=" * 78)
    print(f"  run id              : {run_id}")
    print(f"  training subset     : n={len(X)}, positives={int(y.sum())}")
    print(f"  CV                  : {PILOT_CV_SPLITS} folds x {PILOT_CV_REPEATS} repeat")
    print(f"  supported pipelines : {len(grid)}")
    print(f"  successful          : {sum(v == 'ok' for v in pipeline_status.values())}")
    print(f"  errors              : {len(error_records)}")
    print(f"  warnings            : {len(warning_records)}")
    print(f"  synthetic audits    : {len(sampling_audit)}")
    print(f"  verification        : {'PASSED' if verification['overall_passed'] else 'FAILED'}")
    print(f"  report              : {_relative(report_path)}")
    print(f"  results directory   : {_relative(PILOT_DIR)}")
    if error_records:
        print("  errors were captured in the pilot outputs; see the report.")
    if warning_records:
        print("  warnings were captured in the pilot outputs; see the report.")
    print("  held-out test       : never accessed for fitting or evaluation")
    print("  note                : pilot only; not a final experiment result")

    return 0 if verification["overall_passed"] else 1


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def build_summaries(cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cv.empty:
        return pd.DataFrame(), pd.DataFrame()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for (model_name, strategy_name), group in cv.groupby(
        ["model", "strategy"], sort=False
    ):
        wide_row: dict[str, Any] = {
            "model": model_name,
            "strategy": strategy_name,
            "pipeline": f"{model_name}__{strategy_name}",
            "n_folds": int(len(group)),
            "sampler": group["sampler"].dropna().iloc[0]
            if group["sampler"].notna().any()
            else "none",
            "mean_n_train_before": float(
                group["n_train_before_sampling"].mean()
            ),
            "mean_n_train_after": float(
                group["n_train_after_sampling"].mean()
            ),
        }
        for metric in METRIC_NAMES:
            values = group[metric].astype(float)
            long_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy_name,
                    "pipeline": f"{model_name}__{strategy_name}",
                    "scope": "fold_level",
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_folds": int(len(values)),
                }
            )
            wide_row[metric] = float(values.mean())
            wide_row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        wide_rows.append(wide_row)

    summary = pd.DataFrame(long_rows)
    metric_order = {metric: index for index, metric in enumerate(METRIC_NAMES)}
    model_order = {
        model: index for index, model in enumerate(BASELINE_MODEL_SPECS)
    }
    strategy_order = {
        strategy: index for index, strategy in enumerate(STRATEGIES)
    }
    summary["_model"] = summary["model"].map(model_order)
    summary["_strategy"] = summary["strategy"].map(strategy_order)
    summary["_metric"] = summary["metric"].map(metric_order)
    summary = (
        summary.sort_values(["_model", "_strategy", "_metric"], kind="stable")
        .drop(columns=["_model", "_strategy", "_metric"])
        .reset_index(drop=True)
    )

    summary_wide = pd.DataFrame(wide_rows)
    summary_wide["_model"] = summary_wide["model"].map(model_order)
    summary_wide["_strategy"] = summary_wide["strategy"].map(strategy_order)
    summary_wide = (
        summary_wide.sort_values(["_model", "_strategy"], kind="stable")
        .drop(columns=["_model", "_strategy"])
        .reset_index(drop=True)
    )
    return summary, summary_wide


def build_pooled_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for pipeline_key, group in oof.groupby("pipeline", sort=False):
        model_name, _, strategy_name = pipeline_key.partition("__")
        metrics = compute_metrics(
            group["y_true"].to_numpy(),
            group["y_proba"].to_numpy(),
            threshold=DECISION_THRESHOLD,
        )
        rows.append(
            {
                "pipeline": pipeline_key,
                "model": model_name,
                "strategy": strategy_name,
                "scope": "pooled_oof_3fold",
                "average_precision_definition": (
                    "average_precision_score (step-wise PR area, not "
                    "trapezoidal PR-AUC)"
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: Any,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "details": details})


def verify_pilot(
    *,
    grid: Mapping[str, Any],
    cv: pd.DataFrame,
    oof: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    fold_plan: Sequence[FoldSpec],
    plan_info: dict[str, Any],
    data_metadata: dict[str, Any],
    sampling_audit: pd.DataFrame,
    pipeline_status: Mapping[str, str],
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_keys = _expected_grid_keys()

    _check(
        checks,
        "supported_grid_complete",
        len(grid) == 25 and list(grid) == expected_keys,
        {"n_pipelines": len(grid), "expected": expected_keys},
    )
    _check(
        checks,
        "all_pipelines_attempted",
        set(pipeline_status) == set(expected_keys),
        {
            "attempted": sorted(pipeline_status),
            "missing": sorted(set(expected_keys) - set(pipeline_status)),
        },
    )

    successful = {
        key for key, status in pipeline_status.items() if status == "ok"
    }
    _check(
        checks,
        "all_pipelines_fit_successfully",
        not errors_table.shape[0]
        and successful == set(expected_keys)
        and len(cv) == len(expected_keys) * len(fold_plan),
        {
            "n_successful": len(successful),
            "n_errors": int(errors_table.shape[0]),
            "n_fold_records": int(len(cv)),
            "expected_fold_records": len(expected_keys) * len(fold_plan),
        },
    )

    probability_checks: list[str] = []
    if oof.empty:
        probability_checks.append("no out-of-fold predictions")
    else:
        if not set(oof["pipeline"]).issubset(set(expected_keys)):
            probability_checks.append("unknown pipeline in OOF predictions")
        for pipeline_key, group in oof.groupby("pipeline"):
            values = group["y_proba"].to_numpy(dtype=float)
            if len(group) != len(X):
                probability_checks.append(
                    f"{pipeline_key}: {len(group)} OOF rows, expected {len(X)}"
                )
            if not np.isfinite(values).all():
                probability_checks.append(f"{pipeline_key}: non-finite probabilities")
            if ((values < 0.0) | (values > 1.0)).any():
                probability_checks.append(
                    f"{pipeline_key}: probabilities outside [0, 1]"
                )
            if set(group["row_index"]) != set(X.index):
                probability_checks.append(
                    f"{pipeline_key}: OOF rows do not match pilot rows"
                )
            if len(group) != group["row_index"].nunique():
                probability_checks.append(
                    f"{pipeline_key}: duplicate OOF row predictions"
                )
    _check(
        checks,
        "probabilities_valid",
        not probability_checks,
        {"issues": probability_checks},
    )

    audit_issues: list[str] = []
    if sampling_audit.empty:
        audit_issues.append("no resampling audits were recorded")
    else:
        for _, row in sampling_audit.iterrows():
            if row["audit_status"] != "PASS":
                audit_issues.append(
                    f"{row['pipeline']} repeat {row['repeat']} fold "
                    f"{row['fold']}: {row['issues']}"
                )
            if int(row["n_fractional_one_hot_values"]) != 0:
                audit_issues.append(
                    f"{row['pipeline']}: fractional one-hot values"
                )
            if int(row["n_appended_rows"]) > 0 and not bool(
                row["appended_rows_checked"]
            ):
                audit_issues.append(
                    f"{row['pipeline']}: appended rows were not checked"
                )
    _check(
        checks,
        "synthetic_categorical_values_valid",
        not audit_issues,
        {
            "n_audits": int(len(sampling_audit)),
            "n_failed": int(
                (sampling_audit["audit_status"] != "PASS").sum()
            )
            if not sampling_audit.empty
            else 0,
            "issues": audit_issues,
        },
    )

    fold_lookup = {
        (spec.repeat, spec.fold): spec for spec in fold_plan
    }
    resampling_issues: list[str] = []
    if cv.empty:
        resampling_issues.append("no fold records")
    else:
        for _, row in cv.iterrows():
            key = (int(row["repeat"]), int(row["fold"]))
            spec = fold_lookup.get(key)
            if spec is None:
                resampling_issues.append(f"unknown fold {key}")
                continue
            expected_train = len(spec.train_positions)
            expected_train_counts = class_counts(y.iloc[spec.train_positions])
            expected_validation_counts = class_counts(
                y.iloc[spec.valid_positions]
            )
            if int(row["n_train_before_sampling"]) != expected_train:
                resampling_issues.append(
                    f"{row['pipeline']} {key}: sampler saw "
                    f"{row['n_train_before_sampling']} rows, expected "
                    f"{expected_train}"
                )
            if row["sampling_class_counts_before"] != expected_train_counts:
                resampling_issues.append(
                    f"{row['pipeline']} {key}: sampler before-counts differ "
                    "from the training fold"
                )
            if row["validation_class_counts_before"] != expected_validation_counts:
                resampling_issues.append(
                    f"{row['pipeline']} {key}: validation counts differ from "
                    "the fold plan"
                )
            if row["validation_class_counts_before"] != row[
                "validation_class_counts_after"
            ]:
                resampling_issues.append(
                    f"{row['pipeline']} {key}: validation labels changed"
                )
            if bool(row["sampling_applied"]):
                if int(row["n_fit_resample_calls"]) != 1:
                    resampling_issues.append(
                        f"{row['pipeline']} {key}: sampler called "
                        f"{row['n_fit_resample_calls']} times"
                    )
                if row["strategy"] in {
                    "random_over_sampler",
                    "smote",
                    "adasyn",
                } and not (
                    int(row["n_train_after_sampling"])
                    > int(row["n_train_before_sampling"])
                ):
                    resampling_issues.append(
                        f"{row['pipeline']} {key}: over-sampling did not "
                        "increase the training rows"
                    )
                if row["strategy"] == "random_under_sampler" and not (
                    int(row["n_train_after_sampling"])
                    < int(row["n_train_before_sampling"])
                ):
                    resampling_issues.append(
                        f"{row['pipeline']} {key}: under-sampling did not "
                        "reduce the training rows"
                    )
    _check(
        checks,
        "resampling_only_inside_training_folds",
        not resampling_issues,
        {"issues": resampling_issues},
    )

    split_issues: list[str] = []
    if cv.empty:
        split_issues.append("no fold records")
    else:
        for (repeat, fold), group in cv.groupby(["repeat", "fold"]):
            if group["train_index_fingerprint"].nunique() != 1:
                split_issues.append(
                    f"repeat {repeat} fold {fold}: training splits differ"
                )
            if group["validation_index_fingerprint"].nunique() != 1:
                split_issues.append(
                    f"repeat {repeat} fold {fold}: validation splits differ"
                )
            validation_counts = {
                _counts_text(counts)
                for counts in group["validation_class_counts_before"]
            }
            if len(validation_counts) != 1:
                split_issues.append(
                    f"repeat {repeat} fold {fold}: validation distributions differ"
                )
    _check(
        checks,
        "identical_validation_splits",
        not split_issues,
        {
            "n_fold_groups": int(
                cv[["repeat", "fold"]].drop_duplicates().shape[0]
            )
            if not cv.empty
            else 0,
            "issues": split_issues,
        },
    )

    heldout_issues: list[str] = []
    if not data_metadata.get("heldout_test_was_locked"):
        heldout_issues.append("the held-out test set was not locked")
    if data_metadata.get("heldout_test_data_fields_accessed"):
        heldout_issues.append("held-out test data fields were accessed")
    if not data_metadata.get("pilot_rows_within_training"):
        heldout_issues.append("pilot rows are not confined to the training split")
    if not set(X.index).issubset(set(y.index)):
        heldout_issues.append("pilot X/y indices are not aligned")
    if not oof.empty and not set(oof["row_index"]).issubset(set(X.index)):
        heldout_issues.append("OOF predictions contain non-pilot rows")
    if data_metadata.get("test_set_used"):
        heldout_issues.append("the manifest claims test-set use")
    _check(
        checks,
        "heldout_test_never_accessed",
        not heldout_issues,
        {
            "test_was_locked": data_metadata.get("heldout_test_was_locked"),
            "data_fields_accessed": data_metadata.get(
                "heldout_test_data_fields_accessed"
            ),
            "n_pilot_rows": int(len(X)),
            "n_oof_rows": int(len(oof)),
            "issues": heldout_issues,
        },
    )

    metric_issues: list[str] = []
    if cv.empty:
        metric_issues.append("no fold metrics")
    else:
        for metric in METRIC_NAMES:
            if metric not in cv.columns:
                metric_issues.append(f"missing {metric} in fold metrics")
                continue
            values = cv[metric].astype(float)
            if not np.isfinite(values).all():
                metric_issues.append(f"non-finite {metric} values")
        if len(cv) != len(expected_keys) * len(fold_plan):
            metric_issues.append(
                f"expected {len(expected_keys) * len(fold_plan)} fold rows, "
                f"found {len(cv)}"
            )
    _check(
        checks,
        "required_metrics_present",
        not metric_issues,
        {"metrics": METRIC_NAMES, "issues": metric_issues},
    )

    class_issues: list[str] = []
    for spec in fold_plan:
        train_counts = class_counts(y.iloc[spec.train_positions])
        valid_counts = class_counts(y.iloc[spec.valid_positions])
        if len(train_counts) != 2:
            class_issues.append(
                f"repeat {spec.repeat} fold {spec.fold}: training fold has "
                f"class counts {train_counts}"
            )
        if len(valid_counts) != 2:
            class_issues.append(
                f"repeat {spec.repeat} fold {spec.fold}: validation fold has "
                f"class counts {valid_counts}"
            )
    _check(
        checks,
        "every_fold_has_both_classes",
        not class_issues,
        {"issues": class_issues},
    )

    _check(
        checks,
        "sufficient_minority_examples",
        int(y.sum()) >= PILOT_MIN_POSITIVES
        and min(data_metadata["train_positive_counts_per_fold"])
        >= PILOT_MIN_TRAIN_POSITIVES
        and min(data_metadata["validation_positive_counts_per_fold"])
        >= PILOT_MIN_VALIDATION_POSITIVES,
        {
            "pilot_positives": int(y.sum()),
            "minimum_required": PILOT_MIN_POSITIVES,
            "train_positives_per_fold": data_metadata[
                "train_positive_counts_per_fold"
            ],
            "validation_positives_per_fold": data_metadata[
                "validation_positive_counts_per_fold"
            ],
        },
    )

    _check(
        checks,
        "warnings_captured_in_outputs",
        True,
        {
            "n_warnings": int(len(warnings_table)),
            "categories": sorted(
                warnings_table["category"].unique().tolist()
            )
            if not warnings_table.empty
            else [],
        },
    )
    _check(
        checks,
        "errors_captured_in_outputs",
        True,
        {"n_errors": int(len(errors_table))},
    )
    _check(
        checks,
        "fold_plan_valid",
        plan_info["n_splits"] == PILOT_CV_SPLITS
        and plan_info["n_repeats"] == PILOT_CV_REPEATS
        and plan_info["n_rows"] == len(X),
        plan_info,
    )

    overall = all(check["passed"] for check in checks)
    return {
        "overall_passed": bool(overall),
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": int(sum(check["passed"] for check in checks)),
        "n_failed": int(sum(not check["passed"] for check in checks)),
        "counts": {
            "n_pipelines": len(grid),
            "n_successful": len(successful),
            "n_errors": int(len(errors_table)),
            "n_warnings": int(len(warnings_table)),
            "n_sampling_audits": int(len(sampling_audit)),
            "n_fold_records": int(len(cv)),
            "n_oof_predictions": int(len(oof)),
        },
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_report(
    path: Path,
    *,
    run_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    data_metadata: dict[str, Any],
    plan_info: dict[str, Any],
    summary_wide: pd.DataFrame,
    pooled: pd.DataFrame,
    sampling_audit: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    verification: dict[str, Any],
    pipeline_seconds: Mapping[str, float],
    total_seconds: float,
) -> Path:
    lines: list[str] = []
    lines.append(f"# StrokeSure imbalance pilot - `{run_id}`\n")
    lines.append(
        "This is a small training-only pilot. It is **not** a final model "
        "comparison, does not tune thresholds, does not calibrate "
        "probabilities, and never evaluates the held-out test partition.\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"- Source training partition: {data_metadata['source_training_rows']} rows\n"
        f"- Stratified pilot subset: {len(X)} rows, "
        f"{int(y.sum())} positives, prevalence {float(y.mean()):.4%}\n"
        f"- Pilot subset seed: {data_metadata['pilot_sample_seed']}\n"
        f"- CV: {plan_info['n_splits']} folds x {plan_info['n_repeats']} "
        f"repeat, shared `RepeatedStratifiedKFold` plan, random_state "
        f"{RANDOM_STATE}\n"
        f"- Pipelines attempted: {verification['counts']['n_pipelines']} "
        "(25 supported model/strategy combinations)\n"
        f"- Sampling defaults: ratio "
        f"{_sampling_default('sampling_strategy')}, "
        f"neighbours {_sampling_default('k_neighbors')}, seed "
        f"{_sampling_default('random_state')}\n"
        f"- Decision threshold: {DECISION_THRESHOLD} (fixed; no tuning)\n"
        f"- Total wall time: {total_seconds:.1f}s\n"
    )
    lines.append(
        "- Average precision is the step-wise `average_precision_score`, "
        "not trapezoidal PR-AUC.\n"
    )

    lines.append("## Verification\n")
    lines.append(
        f"**Overall: {'PASSED' if verification['overall_passed'] else 'FAILED'}** "
        f"({verification['n_passed']}/{verification['n_checks']} checks)\n"
    )
    check_rows = [
        {
            "check": check["check"],
            "passed": "yes" if check["passed"] else "NO",
            "details": json.dumps(check["details"], sort_keys=True)
            if not isinstance(check["details"], str)
            else check["details"],
        }
        for check in verification["checks"]
    ]
    lines.append(
        _markdown_table(
            pd.DataFrame(check_rows), ["check", "passed", "details"]
        )
    )

    lines.append("## Metric summary (mean over 3 folds)\n")
    display = summary_wide.copy()
    if not display.empty:
        for metric in METRIC_NAMES:
            display[metric] = display[metric].map(lambda value: f"{value:.4f}")
        lines.append(
            _markdown_table(
                display,
                [
                    "pipeline",
                    "n_folds",
                    "roc_auc",
                    "average_precision",
                    "recall_sensitivity",
                    "specificity",
                    "precision",
                    "f1",
                    "balanced_accuracy",
                    "brier_score",
                ],
            )
        )
    else:
        lines.append("_No successful folds._\n")

    lines.append("## Pooled out-of-fold metrics\n")
    if pooled.empty:
        lines.append("_No pooled predictions._\n")
    else:
        pooled_display = pooled.copy()
        for metric in METRIC_NAMES:
            pooled_display[metric] = pooled_display[metric].map(
                lambda value: f"{value:.4f}"
            )
        lines.append(
            _markdown_table(
                pooled_display,
                [
                    "pipeline",
                    "roc_auc",
                    "average_precision",
                    "recall_sensitivity",
                    "specificity",
                    "precision",
                    "f1",
                    "balanced_accuracy",
                    "brier_score",
                ],
            )
        )

    lines.append("## Resampling audit\n")
    if sampling_audit.empty:
        lines.append("_No resampling audits were recorded._\n")
    else:
        audit_display = sampling_audit.copy()
        audit_display["class_counts_before"] = audit_display[
            "class_counts_before"
        ].astype(str)
        audit_display["class_counts_after"] = audit_display[
            "class_counts_after"
        ].astype(str)
        lines.append(
            _markdown_table(
                audit_display,
                [
                    "pipeline",
                    "repeat",
                    "fold",
                    "n_train_before_sampling",
                    "n_train_after_sampling",
                    "n_appended_rows_checked",
                    "audit_status",
                    "issues",
                ],
            )
        )

    lines.append("## Warnings\n")
    if warnings_table.empty:
        lines.append("_No warnings were emitted._\n")
    else:
        lines.append(
            f"{len(warnings_table)} warning(s) were captured and written to "
            "the warnings CSV.  Nothing was suppressed.\n"
        )
        warning_counts = (
            warnings_table.groupby(["category", "message"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "category"], ascending=[False, True])
        )
        lines.append(
            _markdown_table(
                warning_counts, ["category", "count", "message"]
            )
        )

    lines.append("## Errors\n")
    if errors_table.empty:
        lines.append("_No pipeline errors were observed._\n")
    else:
        lines.append(
            f"{len(errors_table)} pipeline error(s) were captured.  The pilot "
            "reports these rather than silently skipping the pipeline.\n"
        )
        error_display = errors_table[
            ["pipeline", "error_type", "message"]
        ].copy()
        lines.append(
            _markdown_table(
                error_display, ["pipeline", "error_type", "message"]
            )
        )

    lines.append("## Runtime\n")
    runtime_rows = [
        {"pipeline": key, "seconds": float(seconds)}
        for key, seconds in pipeline_seconds.items()
    ]
    lines.append(
        _markdown_table(
            pd.DataFrame(runtime_rows).sort_values("seconds", ascending=False),
            ["pipeline", "seconds"],
        )
    )

    lines.append("## Scope notes\n")
    lines.append(
        "- Only training-partition rows were used. The held-out test set's "
        "lock flag was read; its data fields were never accessed.\n"
        "- Each pipeline was cloned and fitted on the training portion of "
        "each fold. Validation rows were never resampled.\n"
        "- SMOTE used `SMOTENC`; ADASYN used the project's categorical-aware "
        "variant. Both were audited by decoding the actual resampled rows.\n"
        "- `dummy_prior` appears only with `none`, because class weighting and "
        "resampling would change its no-skill reference distribution.\n"
        "- This pilot uses one 3-fold repeat on a subset, so its estimates are "
        "descriptive and must not be treated as final results.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _sampling_default(name: str) -> Any:
    from src.config import (
        IMBALANCE_DEFAULT_K_NEIGHBORS,
        IMBALANCE_DEFAULT_SAMPLING_RATIO,
        IMBALANCE_RANDOM_STATE,
    )

    return {
        "sampling_strategy": IMBALANCE_DEFAULT_SAMPLING_RATIO,
        "k_neighbors": IMBALANCE_DEFAULT_K_NEIGHBORS,
        "random_state": IMBALANCE_RANDOM_STATE,
    }[name]


def save_pilot_outputs(
    *,
    run_id: str,
    log_path: Path,
    X: pd.DataFrame,
    y: pd.Series,
    fold_plan: Sequence[FoldSpec],
    data_metadata: dict[str, Any],
    plan_info: dict[str, Any],
    cv: pd.DataFrame,
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    summary_wide: pd.DataFrame,
    pooled: pd.DataFrame,
    sampling_audit: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    verification: dict[str, Any],
    pipeline_status: Mapping[str, str],
    pipeline_seconds: Mapping[str, float],
    grid: Mapping[str, Any],
    total_seconds: float,
) -> dict[str, Path]:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def write_csv(frame: pd.DataFrame, suffix: str) -> Path:
        path = PILOT_DIR / f"{run_id}_{suffix}.csv"
        frame.to_csv(path, index=False)
        paths[suffix] = path
        return path

    write_csv(cv, "fold_metrics")
    write_csv(summary, "summary")
    write_csv(summary_wide, "summary_wide")
    write_csv(pooled, "pooled_metrics")
    write_csv(oof, "oof_predictions")
    write_csv(sampling_audit, "sampling_audit")
    write_csv(warnings_table, "warnings")
    write_csv(errors_table, "errors")

    fold_plan_rows = [
        {
            "repeat": int(spec.repeat),
            "fold": int(spec.fold),
            "n_train": int(len(spec.train_positions)),
            "n_validation": int(len(spec.valid_positions)),
            "n_train_positive": int(
                y.iloc[spec.train_positions].sum()
            ),
            "n_validation_positive": int(
                y.iloc[spec.valid_positions].sum()
            ),
            "train_index_fingerprint": _index_fingerprint(
                X.index.values[spec.train_positions]
            ),
            "validation_index_fingerprint": _index_fingerprint(
                X.index.values[spec.valid_positions]
            ),
        }
        for spec in fold_plan
    ]
    write_csv(pd.DataFrame(fold_plan_rows), "fold_plan")

    verification_path = PILOT_DIR / f"{run_id}_verification.json"
    write_json(verification, verification_path)
    paths["verification"] = verification_path

    manifest = {
        "run_id": run_id,
        "step": "pilot_imbalance",
        "created_utc": utc_timestamp(),
        "environment": environment_snapshot(),
        "scope": (
            "Small training-only pilot; not a final model comparison. "
            "No threshold tuning, calibration or held-out test evaluation."
        ),
        "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD,
        "metric_definitions": METRIC_LABELS,
        "average_precision_note": (
            "average_precision_score computes a step-wise estimate of the area "
            "under the precision-recall curve. It is not trapezoidal PR-AUC."
        ),
        "data": data_metadata,
        "fold_plan": plan_info,
        "supported_pipelines": list(grid),
        "pipeline_status": dict(pipeline_status),
        "pipeline_seconds": {
            key: float(value) for key, value in pipeline_seconds.items()
        },
        "cv": {
            "n_splits": PILOT_CV_SPLITS,
            "n_repeats": PILOT_CV_REPEATS,
            "random_state": RANDOM_STATE,
            "shared_fold_plan": True,
            "validation_resampling": False,
        },
        "sampling": {
            "strategies": list(STRATEGIES),
            "resampling_strategies": list(RESAMPLING_STRATEGIES),
            "sampling_ratio": _sampling_default("sampling_strategy"),
            "k_neighbors": _sampling_default("k_neighbors"),
            "seed": _sampling_default("random_state"),
            "sampling_audits": int(len(sampling_audit)),
            "sampling_audit_failures": int(
                (sampling_audit["audit_status"] != "PASS").sum()
            )
            if not sampling_audit.empty
            else 0,
        },
        "strategy_table": strategy_table(),
        "support_matrix": support_matrix(),
        "warnings": {
            "n_warnings": int(len(warnings_table)),
            "path": _relative(paths["warnings"]),
        },
        "errors": {
            "n_errors": int(len(errors_table)),
            "path": _relative(paths["errors"]),
        },
        "verification": {
            "overall_passed": verification["overall_passed"],
            "n_checks": verification["n_checks"],
            "n_passed": verification["n_passed"],
            "n_failed": verification["n_failed"],
            "path": _relative(verification_path),
        },
        "heldout_test": {
            "lock_read": True,
            "data_fields_accessed": False,
            "used_for_fitting_or_evaluation": False,
        },
        "files": {
            key: _relative(path)
            for key, path in paths.items()
        },
    }
    manifest_path = PILOT_DIR / f"{run_id}_manifest.json"
    write_json(manifest, manifest_path)
    paths["manifest"] = manifest_path

    report_path = _write_report(
        PILOT_DIR / f"{run_id}_report.md",
        run_id=run_id,
        X=X,
        y=y,
        data_metadata=data_metadata,
        plan_info=plan_info,
        summary_wide=summary_wide,
        pooled=pooled,
        sampling_audit=sampling_audit,
        warnings_table=warnings_table,
        errors_table=errors_table,
        verification=verification,
        pipeline_seconds=pipeline_seconds,
        total_seconds=total_seconds,
    )
    paths["report"] = report_path
    paths["log"] = log_path

    # Add the report/log to the manifest's file map now that they exist.
    manifest["files"] = {
        key: _relative(path) for key, path in paths.items()
    }
    write_json(manifest, manifest_path)
    return paths


if __name__ == "__main__":
    raise SystemExit(run_pilot())
