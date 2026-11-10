"""Step 32 / G9 — full-size timing preflight and scientific pipeline verification.

Every artifact produced by this module is a PREFLIGHT diagnostic, clearly
marked "preflight" in its name and stored inside
``results/preflight_full_size/preflight_<run_id>/``.  Preflight results are
*never* publication results: they must not be used to select models,
strategies, thresholds or sampling ratios.

What this module does
---------------------
1. Loads the training partition only (the held-out test container must be
   locked; its data is never read).
2. Builds the original 5-fold stratified CV plan (``random_state=42``, one
   repeat) and verifies, by repeat fingerprint, that the executed repeat is
   exactly the first repeat of the documented 5x5 plan.
3. Runs exactly ONE representative full-size training/validation fold
   (repeat 1, fold 1 — the same fold for every combination) for each of the
   supported model/strategy pipelines, recording per-pipeline fit time,
   prediction time, wall time, sampled peak process memory (RSS probe),
   class counts, actual sampling ratios, warning records, errors,
   out-of-fold probabilities and the full metric suite at the fixed decision
   threshold of 0.5 (no tuning anywhere).
4. Performs a scientific verification of the sampling geometry actually used
   for nearest-neighbour computation by SMOTENC and by the project-specific
   CategoricalADASYN: exact representation, scaling of continuous relative
   to categorical features, categorical integrity, categorical distance
   semantics, synthetic category choice and per-seed allocation.  Standard
   ADASYN is numeric-only, so equivalence is NOT claimed; findings are
   measured, and a methodological concern is reported with a proposed
   correction rather than changed silently.
5. Validates the Step 31 G1/G2/G3 machinery on the preflight scope: source
   hashes (start/end), output isolation, manifest completeness, warning and
   error handling and the expected 25-attempt count.
6. Estimates the computational cost of the proposed full 625-fold run from
   the observed full-size timings (a range, not a promise).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import traceback as traceback_module
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from src.config import (
    DECISION_THRESHOLD,
    IMBALANCE_DEFAULT_K_NEIGHBORS,
    IMBALANCE_DEFAULT_SAMPLING_RATIO,
    IMBALANCE_RANDOM_STATE,
    RESULTS_DIR,
)
from src.evaluation import build_run_id, utc_timestamp, write_json
from src.imbalance import BASELINE_MODEL_SPECS, STRATEGIES, supports_strategy
from src.imbalance_experiment import run_imbalance_cv, validate_shared_fold_plan
from src.imbalance_outputs import (
    EXPECTED_WARNING_CATEGORIES,
    EXPECTED_WARNING_MESSAGE_PREFIX,
    HEX64_PATTERN,
    RESAMPLING_STRATEGIES,
    CapturingRecordingSampler,
    OutputIsolationError,
    RunConfigurationError,
    RunDirectory,
    _markdown_table,
    _repeatability_pin,
    _write_frame,
    audit_resampling_fold,
    build_audited_grid,
    classify_warning,
    configuration_file_hashes,
    core_environment,
    describe_pipelines,
    diff_source_hashes,
    file_sha256,
    load_training_partition,
    protected_paths,
    source_file_hashes,
)
from src.preflight import build_fold_plan

__all__ = [
    "PREFLIGHT_RESULTS_DIR",
    "PREFLIGHT_STEP",
    "analyze_sampling_geometry",
    "create_preflight_directory",
    "estimate_full_run_cost",
    "expected_supported_grid_keys",
    "run_full_size_preflight",
    "run_preflight_attempts",
    "supported_grid_keys",
    "verify_preflight_run",
]

PREFLIGHT_RESULTS_DIR = RESULTS_DIR / "preflight_full_size"
PREFLIGHT_STEP = "step32_preflight_full_size"
PROBE_INTERVAL_S = 0.02


def _dense(value: Any) -> np.ndarray:
    """Dense float view of a numpy array or scipy sparse matrix."""
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Memory probe (best-effort peak RSS measured while a stage runs)
# ---------------------------------------------------------------------------
class RssProbe(threading.Thread):
    """Background thread sampling process RSS at a fixed interval.

    Windows exposes no per-scope resettable working-set peak, so the probe
    records the peak working set observed *while the stage runs*.  Spikes
    shorter than the sampling interval can be missed; the measurement is
    recorded as best-effort in the report.
    """

    def __init__(self, interval_s: float = PROBE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_event = threading.Event()
        self._process = psutil.Process()
        self.baseline_mb = self._process.memory_info().rss / (2**20)
        self.peak_mb = self.baseline_mb

    def run(self) -> None:
        while not self._stop_event.is_set():
            rss_mb = self._process.memory_info().rss / (2**20)
            if rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            self._stop_event.wait(self._interval)

    def stop(self) -> dict[str, float]:
        self._stop_event.set()
        self.join(timeout=2.0)
        return {
            "rss_baseline_mb": round(self.baseline_mb, 2),
            "rss_peak_mb": round(self.peak_mb, 2),
            "rss_peak_delta_mb": round(
                max(0.0, self.peak_mb - self.baseline_mb), 2
            ),
        }


def _configure_logger(log_path: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"preflight_full_size.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Isolated preflight directory (mirrors the G1 run contract)
# ---------------------------------------------------------------------------
def create_preflight_directory(run_id: str | None = None) -> RunDirectory:
    """Fresh ``<preflight parent>/preflight_<UTC id>/`` directory.

    Same contract as :func:`src.imbalance_outputs.create_run_directory`: the
    directory must not exist, and it may not overlap any protected project
    path.  The preflight parent itself is this module's own scope and is the
    only permitted parent.
    """
    if run_id is None:
        run_id = build_run_id("preflight")
    if not re.match(r"^[A-Za-z0-9_]+$", run_id):
        raise RunConfigurationError(
            f"run_id {run_id!r} contains disallowed characters."
        )
    target = (PREFLIGHT_RESULTS_DIR / run_id).resolve()
    parent = PREFLIGHT_RESULTS_DIR.resolve()
    issues: list[str] = []
    if target.parent != parent:
        issues.append(
            f"preflight run directory must live directly under {parent}"
        )
    if target.name.startswith("baseline_"):
        issues.append("preflight runs must not use baseline_ names")
    if target.exists():
        issues.append(
            f"run directory already exists; refusing to overwrite: {target}"
        )
    for protected in protected_paths():
        protected_resolved = protected.resolve()
        if protected_resolved == parent:
            continue  # the preflight parent is this module's own scope
        if target.is_relative_to(protected_resolved):
            issues.append(
                f"{target} is inside protected path {protected_resolved}"
            )
        elif protected_resolved.is_relative_to(parent):
            issues.append(
                f"preflight scope would contain protected path {protected_resolved}"
            )
    if issues:
        raise OutputIsolationError(
            "Refusing to create the preflight directory. Isolation issues: "
            + "; ".join(issues)
        )
    target.mkdir(parents=True, exist_ok=False)
    return RunDirectory(
        run_id=run_id,
        path=target,
        results_dir=parent,
        created_utc=utc_timestamp(),
        isolation_issues=(),
        canonical_results_dir=False,
    )


# ---------------------------------------------------------------------------
# Supported grid keys (independent expectation for the strict check)
# ---------------------------------------------------------------------------
def expected_supported_grid_keys() -> list[str]:
    """The supported ``model__strategy`` keys, derived independently of the
    grid builder (uses the support helper of the strategy layer only)."""
    keys: list[str] = []
    for model_name in BASELINE_MODEL_SPECS:
        for strategy_name in STRATEGIES:
            if supports_strategy(model_name, strategy_name):
                keys.append(f"{model_name}__{strategy_name}")
    return sorted(set(keys))


def supported_grid_keys() -> dict[str, Any]:
    """The supported leakage-safe grid (25 entries when unfiltered)."""
    from src.imbalance_pipelines import build_leakage_safe_grid

    return build_leakage_safe_grid(record_sampling=True)


# ---------------------------------------------------------------------------
# One representative fold per pipeline
# ---------------------------------------------------------------------------
def run_preflight_attempts(
    X: pd.DataFrame,
    y: pd.Series,
    audited_grid: Mapping[str, Any],
    spec: Any,
    *,
    threshold: float = DECISION_THRESHOLD,
    probe_interval_s: float = PROBE_INTERVAL_S,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run ONE fold attempt per pipeline (the same ``spec`` for every one).

    Returns a dict with ``cv`` (fold records incl. metrics, counts and fit /
    predict seconds), ``oof`` (one row per prediction), ``timings`` (timing +
    memory rows), ``sampling_audit``, ``warnings``, ``errors`` and
    ``statuses``.  Failures become explicit rows; nothing is dropped.
    """
    timings_rows: list[dict[str, Any]] = []
    warning_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    cv_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    statuses: dict[str, str] = {}

    for pipeline_key, pipeline in audited_grid.items():
        model_name, _, strategy_name = pipeline_key.partition("__")
        if logger is not None:
            logger.info("preflight attempt %s: starting", pipeline_key)
        fold_audits: list[dict[str, Any]] = []

        def observer(**kwargs: Any) -> None:
            if strategy_name in RESAMPLING_STRATEGIES:
                fold_audits.append(
                    audit_resampling_fold(
                        pipeline_key=kwargs["pipeline_key"],
                        model_name=kwargs["model_name"],
                        strategy_name=kwargs["strategy_name"],
                        spec=kwargs["spec"],
                        pipeline=kwargs["pipeline"],
                        sampler_step=kwargs["sampler_step"],
                        y_train=kwargs["y_train"],
                        y_validation=kwargs["y_validation"],
                    )
                )

        probe = RssProbe(probe_interval_s)
        wall_started = time.perf_counter()
        try:
            probe.start()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fold_cv, fold_oof, _meta = run_imbalance_cv(
                    X,
                    y,
                    {pipeline_key: pipeline},
                    fold_plan=[spec],
                    threshold=threshold,
                    fold_observer=observer,
                    validate_plan=False,
                )
        except Exception as exc:  # noqa: BLE001 -- captured, never silent
            memory = probe.stop()
            wall_seconds = time.perf_counter() - wall_started
            statuses[pipeline_key] = "failed"
            error_records.append(
                {
                    "pipeline": pipeline_key,
                    "model": model_name,
                    "strategy": strategy_name or "none",
                    "repeat": int(spec.repeat),
                    "fold": int(spec.fold),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback_module.format_exc(),
                    "fit_seconds": None,
                    "predict_seconds": None,
                    "wall_seconds": round(wall_seconds, 4),
                    **memory,
                }
            )
            timings_rows.append(
                {
                    "pipeline": pipeline_key,
                    "model": model_name,
                    "strategy": strategy_name or "none",
                    "status": "failed",
                    "repeat": int(spec.repeat),
                    "fold": int(spec.fold),
                    "fit_seconds": None,
                    "predict_seconds": None,
                    "wall_seconds": round(wall_seconds, 4),
                    **memory,
                    "n_features": None,
                    "n_train_before_sampling": None,
                    "n_train_after_sampling": None,
                    "n_validation": None,
                    "n_warnings": None,
                    "n_audits": len(fold_audits),
                    "n_oof_rows": 0,
                }
            )
            if logger is not None:
                logger.error(
                    "preflight attempt %s FAILED: %s: %s",
                    pipeline_key,
                    type(exc).__name__,
                    str(exc)[:200],
                )
            continue

        memory = probe.stop()
        wall_seconds = time.perf_counter() - wall_started
        row = fold_cv.iloc[0]
        statuses[pipeline_key] = "ok"
        timings_rows.append(
            {
                "pipeline": pipeline_key,
                "model": model_name,
                "strategy": strategy_name or "none",
                "status": "ok",
                "repeat": int(spec.repeat),
                "fold": int(spec.fold),
                "fit_seconds": float(row["fit_seconds"]),
                "predict_seconds": float(row["predict_seconds"]),
                "wall_seconds": round(wall_seconds, 4),
                **memory,
                "n_features": int(row["n_features"]),
                "n_train_before_sampling": int(row["n_train_before_sampling"]),
                "n_train_after_sampling": int(row["n_train_after_sampling"]),
                "n_validation": int(row["n_validation"]),
                "n_warnings": len(caught),
                "n_audits": len(fold_audits),
                "n_oof_rows": int(len(fold_oof)),
            }
        )
        audit_records.extend(fold_audits)
        for warning in caught:
            warning_records.append(
                {
                    "pipeline": pipeline_key,
                    "phase": "fit/predict",
                    "repeat": int(spec.repeat),
                    "fold": int(spec.fold),
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": int(warning.lineno) if warning.lineno else -1,
                }
            )
        cv_parts.append(fold_cv)
        oof_parts.append(fold_oof)
        if logger is not None:
            logger.info(
                "preflight attempt %s: ok (fit %.3fs, predict %.3fs, "
                "peak RSS delta %.1f MB)",
                pipeline_key,
                float(row["fit_seconds"]),
                float(row["predict_seconds"]),
                timings_rows[-1]["rss_peak_delta_mb"],
            )

    return {
        "cv": pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame(),
        "oof": pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(),
        "timings": pd.DataFrame(timings_rows),
        "sampling_audit": pd.DataFrame(audit_records),
        "warnings": pd.DataFrame(warning_records),
        "errors": pd.DataFrame(error_records),
        "statuses": statuses,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# Scientific verification of the sampling geometry
# ---------------------------------------------------------------------------
def _pairwise_sq_distance_decomposition(
    matrix: np.ndarray, n_continuous: int
) -> dict[str, Any]:
    """Mean pairwise squared distance split into the first ``n_continuous``
    columns (continuous block) versus the remaining columns (discrete block)."""
    sample = matrix
    if sample.shape[0] > 600:
        rng = np.random.RandomState(0)
        sample = sample[rng.choice(sample.shape[0], size=600, replace=False)]
    diff = sample[:, None, :] - sample[None, :, :]
    d2 = (diff**2).sum(axis=2)
    upper = d2[np.triu_indices_from(d2, k=1)]
    continuous_sq = (diff[..., :n_continuous] ** 2).sum(axis=2)
    cont_mean = float(
        continuous_sq[np.triu_indices_from(continuous_sq, k=1)].mean()
    )
    total_mean = float(upper.mean())
    discrete_mean = total_mean - cont_mean
    return {
        "n_sample_rows": int(sample.shape[0]),
        "n_pairs": int(len(upper)),
        "mean_sq_distance_total": total_mean,
        "mean_sq_distance_continuous": cont_mean,
        "mean_sq_distance_discrete": discrete_mean,
        "continuous_share": (cont_mean / total_mean) if total_mean > 0 else None,
        "discrete_share": (discrete_mean / total_mean) if total_mean > 0 else None,
    }


def analyze_sampling_geometry(
    X: pd.DataFrame,
    y: pd.Series,
    spec: Any,
    *,
    random_state: int,
) -> dict[str, Any]:
    """Measure and verify the mixed-feature sampling geometry on full-size data.

    The two distance spaces are inspected exactly as the algorithms see them
    (behaviour of ``imbalanced-learn 0.14.2`` verified in its sources):

    SMOTENC
        One-hot encodes the ordinal codes it receives, replaces every
        non-zero one-hot entry with ``median_std / sqrt(2)`` (the SMOTE-NC
        paper's substitution), fits its NN graph on the minority class only,
        interpolates continuous columns in raw space and copies each
        categorical feature from the most frequent level among the
        interpolation neighbours (random tie-break).

    CategoricalADASYN
        Fits a continuous-feature StandardScaler on the training fold, then
        one-hot encodes discrete columns and runs ``imblearn`` ADASYN in the
        ``[standardized continuous | one-hot]`` space. Synthetic continuous
        values are inverse-transformed and discrete codes are recovered by
        the per-block argmax. This is a project-specific categorical-aware
        ADASYN variant, not standard categorical ADASYN or SMOTE-NC.
    """
    import src.imbalance_pipelines as ip
    from sklearn.neighbors import NearestNeighbors

    X_train = X.iloc[spec.train_positions].reset_index(drop=True)
    y_train = y.iloc[spec.train_positions].reset_index(drop=True)
    y_arr = np.asarray(y_train)
    minority_label = 1
    minority_idx = np.flatnonzero(y_arr == minority_label)

    report: dict[str, Any] = {"checks": [], "findings": []}

    def check(condition: bool, name: str, **values: Any) -> None:
        report["checks"].append(
            {"check": name, "passed": bool(condition), **values}
        )

    # -- the exact representation --------------------------------------------
    fitted: dict[str, tuple[Any, Any]] = {}
    for strategy in ("smote", "adasyn"):
        base = ip.build_leakage_safe_pipeline(
            "logistic_regression",
            strategy,
            random_state=random_state,
        )
        inner = base.named_steps["sampler"].sampler
        steps = list(base.steps)
        idx = next(i for i, (name, _) in enumerate(steps) if name == "sampler")
        steps[idx] = ("sampler", CapturingRecordingSampler(inner))
        wrapped = ip.ImblearnPipeline(steps=steps)
        wrapped.fit(X_train, y_train)
        fitted[strategy] = (wrapped, wrapped.named_steps["sampler"])

    smote_pipe, smote_rec = fitted["smote"]
    adasyn_pipe, adasyn_rec = fitted["adasyn"]
    encoder = smote_pipe.named_steps["sampling_encoder"]
    feature_names = list(encoder.output_feature_names())
    n_continuous = len(encoder.numeric_features_)
    cat_indices = np.asarray(encoder.categorical_output_indices(), dtype=int)
    Z = np.asarray(encoder.transform(X_train), dtype=float)

    report["representation"] = {
        "encoder_class": type(encoder).__name__,
        "n_rows": int(Z.shape[0]),
        "n_columns": int(Z.shape[1]),
        "feature_block": {
            "continuous": feature_names[:n_continuous],
            "discrete": feature_names[n_continuous:],
            "discrete_representation": (
                "ordinal integer codes (binary as 0/1) from a fold-fit "
                "vocabulary; the model never sees fractional categories"
            ),
        },
        "smotenc_internal_space": {
            "class": type(smote_rec.sampler).__name__,
            "internal_one_hot_encoder": type(
                smote_rec.sampler.categorical_encoder_
            ).__name__,
            "median_std_by_class": {
                str(k): float(v)
                for k, v in smote_rec.sampler.median_std_.items()
            },
            "non_zero_onehot_entries_scaled_to": "median_std / sqrt(2)",
        },
        "categoricaladasyn_internal_space": {
            "class": type(adasyn_rec.sampler).__name__,
            "internal_one_hot_encoder": type(
                adasyn_rec.sampler.categorical_encoder_
            ).__name__,
            "delegates_to": type(adasyn_rec.sampler.adasyn_).__name__,
        },
    }

    # -- SMOTENC distance space ----------------------------------------------
    smotenc = smote_rec.sampler
    median_std = float(smotenc.median_std_[minority_label])
    continuous_stds = {
        name: float(np.std(Z[minority_idx, i]))
        for i, name in enumerate(feature_names[:n_continuous])
    }
    one_hot = _dense(
        smotenc.categorical_encoder_.transform(Z[:, cat_indices])
    )
    scaled_one_hot = one_hot.copy()
    scaled_one_hot[np.isclose(scaled_one_hot, 1.0)] = median_std / math.sqrt(2.0)
    smotenc_space_minority = np.hstack(
        [Z[minority_idx, :n_continuous], scaled_one_hot[minority_idx]]
    )
    smote_decomp = _pairwise_sq_distance_decomposition(
        smotenc_space_minority, n_continuous
    )
    report["smotenc_space"] = {
        "scope": (
            "NN graph fitted on minority rows only; non-zero one-hot entries"
            " scaled to median_std/sqrt(2); continuous columns raw"
        ),
        "median_std_reference": median_std,
        "minority_continuous_std_by_feature": continuous_stds,
        "pairwise_sq_distance_decomposition": smote_decomp,
        "categorical_mismatch_semantics": {
            "rule": (
                "same category -> 0 squared-distance contribution; different"
                " category -> median_std^2 (two substituted one-hot entries"
                " differing by median_std/sqrt(2)"
            ),
            "paper_faithful": True,
        },
    }
    check(
        smote_decomp["continuous_share"] is not None
        and math.isfinite(smote_decomp["continuous_share"]),
        "smotenc_distance_decomposition_is_finite",
        continuous_share=smote_decomp["continuous_share"],
        discrete_share=smote_decomp["discrete_share"],
    )

    # -- SMOTENC synthetic-sample generation ----------------------------------
    X_smote_res = np.asarray(smote_rec.resampled_X_, dtype=float)
    y_smote_res = np.asarray(smote_rec.resampled_y_)
    n_before = len(y_train)
    n_new_smote = len(y_smote_res) - n_before
    synth_smote = X_smote_res[n_before:]
    train_rows = X_smote_res[:n_before]
    synth_codes = synth_smote[:, cat_indices]
    train_codes = train_rows[:, cat_indices]
    synth_blocks = set(map(tuple, np.round(synth_codes, 6)))
    train_blocks = set(map(tuple, np.round(train_codes, 6)))
    novel_blocks = synth_blocks - train_blocks
    integer_ok = bool(np.all(synth_codes == np.round(synth_codes)))
    valid_levels_ok = True
    discrete_names = feature_names[n_continuous:]
    for j, _name in enumerate(discrete_names):
        # codes are ordinal positions into the fold-fit vocabulary
        n_levels = len(encoder.discrete_encoder_.categories_[j])
        allowed = set(float(code) for code in range(n_levels))
        column_values = set(np.round(synth_codes[:, j], 6).tolist())
        if not column_values.issubset(allowed):
            valid_levels_ok = False
    n_minority_before = int((y_arr == minority_label).sum())
    n_synthetic_requested = int(
        smotenc.sampling_strategy_.get(minority_label, -1)
    )
    intended_target = n_minority_before + n_synthetic_requested
    final_minority = int((y_smote_res == minority_label).sum())
    report["smotenc_generation"] = {
        "n_synthetic": int(n_new_smote),
        "intended_minority_target": intended_target,
        "final_minority_count": final_minority,
        "count_matches_target": bool(final_minority == intended_target),
        "synthetic_category_blocks_all_observed": bool(not novel_blocks),
        "n_novel_category_combinations": int(len(novel_blocks)),
        "all_synthetic_codes_integer": integer_ok,
        "all_synthetic_codes_valid_levels": bool(valid_levels_ok),
        "category_choice_rule": (
            "per discrete feature: the most frequent level among the"
            " interpolation neighbours; exact ties broken randomly;"
            " categorical values are never numerically interpolated"
        ),
    }
    check(
        integer_ok and bool(valid_levels_ok),
        "smotenc_synthetic_codes_integer_and_valid",
        integer_ok=integer_ok,
        valid_levels_ok=bool(valid_levels_ok),
        n_synthetic=int(n_new_smote),
    )
    check(
        final_minority == intended_target,
        "smotenc_final_class_count_matches_strategy_target",
        final=final_minority,
        target=intended_target,
    )

    # -- CategoricalADASYN distance space + exact replication -----------------
    wrapper = adasyn_rec.sampler  # CategoricalADASYN (fitted)
    adasyn = wrapper.adasyn_
    # Use this pipeline's own fitted encoder, never a sibling pipeline's
    # representation even when their preprocessing currently agrees.
    Z = np.asarray(
        adasyn_pipe.named_steps["sampling_encoder"].transform(X_train), dtype=float
    )
    cat_idx_adasyn = np.asarray(wrapper.categorical_features, dtype=int)
    cont_idx_adasyn = np.setdiff1d(np.arange(Z.shape[1]), cat_idx_adasyn)
    one_hot_adasyn = _dense(
        wrapper.categorical_encoder_.transform(Z[:, cat_idx_adasyn])
    )
    scaled_continuous = wrapper.continuous_scaler_.transform(
        Z[:, cont_idx_adasyn]
    )
    encoded_adasyn = np.hstack([
        scaled_continuous,
        one_hot_adasyn,
    ])
    fitted_generation_space = _dense(adasyn.nn_._fit_X)
    expected_generation_space = encoded_adasyn[y_arr == minority_label]
    check(
        fitted_generation_space.shape == expected_generation_space.shape
        and np.array_equal(fitted_generation_space, expected_generation_space),
        "adasyn_representation_matches_fitted_neighbours",
        scope="exact minority generation NN training matrix",
    )
    adasyn_decomp = _pairwise_sq_distance_decomposition(
        encoded_adasyn, len(cont_idx_adasyn)
    )
    report["adasyn_space"] = {
        "scope": (
            "hardness NN is fitted on ALL training-fold rows of"
            " [standardized continuous | one-hot]; generation NN on the"
            " minority class only; continuous scaling is training-fold-local"
        ),
        "pairwise_sq_distance_decomposition": adasyn_decomp,
        "categorical_mismatch_semantics": {
            "rule": (
                "same category -> 0; different category -> 2 squared-units"
                " (unscaled one-hot), independent of the continuous scales"
            ),
        },
    }
    check(
        adasyn_decomp["continuous_share"] is not None,
        "adasyn_distance_decomposition_is_finite",
        continuous_share=adasyn_decomp["continuous_share"],
        discrete_share=adasyn_decomp["discrete_share"],
    )

    # hardness-weight sensitivity: full space vs continuous-only
    target_class_idx = np.flatnonzero(y_arr == minority_label)
    X_class_mixed = encoded_adasyn[target_class_idx]
    nn_specs = {
        key: value
        for key, value in adasyn.nn_.get_params(deep=False).items()
        if key != "n_jobs"
    }
    nn_full = NearestNeighbors(**nn_specs).fit(encoded_adasyn)
    nn_cont = NearestNeighbors(**nn_specs).fit(
        encoded_adasyn[:, : len(cont_idx_adasyn)]
    )
    nns_full = nn_full.kneighbors(X_class_mixed, return_distance=False)[:, 1:]
    nns_cont = nn_cont.kneighbors(
        X_class_mixed[:, : len(cont_idx_adasyn)], return_distance=False
    )[:, 1:]
    n_neighbours_hard = int(nn_full.n_neighbors) - 1
    ratio_full = (y_arr[nns_full] != minority_label).sum(axis=1) / n_neighbours_hard
    ratio_cont = (y_arr[nns_cont] != minority_label).sum(axis=1) / n_neighbours_hard
    denom = float(
        math.sqrt(float((ratio_full**2).sum()))
        * math.sqrt(float((ratio_cont**2).sum()))
    )
    hardness_agreement = (
        float(np.dot(ratio_full, ratio_cont) / denom) if denom > 0 else None
    )
    report["categoricaladasyn_hardness"] = {
        "cosine_similarity_full_vs_continuous_only": hardness_agreement,
        "interpretation": (
            "1.0 would mean the one-hot columns have no measurable effect on"
            " which minority rows are considered difficult; values close to 1"
            " indicate adaptive weights are mostly driven by the continuous"
            " coordinates in the measured standardized feature space"
        ),
    }

    # exact replication of ADASYN's allocation (proves the attribution)
    if not float(np.sum(ratio_full)) > 0:
        check(
            False,
            "adasyn_allocation_replication_is_exact",
            detail="no majority neighbour visible; ADASYN unsuited for this fold",
        )
        report["categoricaladasyn_generation"] = {
            "error": "ADASYN's guard triggered: no majority neighbours within k"
        }
    else:
        n_gen_target = int(adasyn.sampling_strategy_[minority_label])
        n_samples_generate = np.rint(
            ratio_full / float(np.sum(ratio_full)) * n_gen_target
        ).astype(int)
        n_gen_actual = int(n_samples_generate.sum())
        from numpy.random import RandomState

        rng = RandomState(random_state)
        rows_local = np.repeat(
            np.arange(len(target_class_idx)), n_samples_generate
        )
        cols_local = rng.choice(n_neighbours_hard, size=n_gen_actual)
        generation_nns = (
            adasyn.nn_.kneighbors(X_class_mixed, return_distance=False)[:, 1:]
        )
        seeded = X_class_mixed[rows_local]
        diffs = X_class_mixed[generation_nns[rows_local, cols_local]] - seeded
        steps = rng.uniform(size=(n_gen_actual, 1))
        replicated_mixed = seeded + steps * diffs

        actual_resampled = np.asarray(adasyn_rec.resampled_X_, dtype=float)
        actual_new = actual_resampled[n_before:]
        shape_ok = actual_new.shape == (
            n_gen_actual,
            actual_resampled.shape[1],
        )

        # The geometry checker must never compare misaligned synthetic rows.
        # Report a failed verification instead of broadcasting/truncating rows.
        max_deviation_continuous = float("inf")
        max_deviation_codes = float("inf")
        fraction_seed = None
        fraction_neighbour = None
        fraction_other = None
        replication_ok = False
        if shape_ok and n_gen_actual > 0:
            replicated_continuous = wrapper.continuous_scaler_.inverse_transform(
                replicated_mixed[:, : len(cont_idx_adasyn)]
            )
            max_deviation_continuous = float(
                np.max(
                    np.abs(
                        actual_new[:, cont_idx_adasyn]
                        - replicated_continuous
                    )
                )
            )

            decoded_blocks = []
            offset = len(cont_idx_adasyn)
            for levels in wrapper.categorical_encoder_.categories_:
                width = len(levels)
                block = replicated_mixed[:, offset : offset + width]
                indices = np.argmax(block, axis=1)
                codes = np.asarray(levels, dtype=float)[indices]
                empty = np.isclose(block.sum(axis=1), 0.0, atol=1e-12)
                codes[empty] = -1.0
                decoded_blocks.append(codes)
                offset += width
            replicated_codes = np.column_stack(decoded_blocks)
            max_deviation_codes = float(
                np.max(np.abs(actual_new[:, cat_idx_adasyn] - replicated_codes))
            )
            replication_ok = (
                max_deviation_continuous <= 1e-9
                and max_deviation_codes <= 1e-9
            )

            synth_decoded = actual_new[:, cat_idx_adasyn]
            seed_blocks = Z[target_class_idx[rows_local]][:, cat_idx_adasyn]
            neighbour_blocks = Z[
                target_class_idx[generation_nns[rows_local, cols_local]]
            ][:, cat_idx_adasyn]

            # Category inheritance is per feature. A synthetic observation may
            # legitimately combine the seed's value for one discrete feature
            # with its neighbour's value for another feature.
            matches_seed = np.isclose(
                synth_decoded, seed_blocks, rtol=0.0, atol=1e-9
            )
            matches_neighbour = np.isclose(
                synth_decoded, neighbour_blocks, rtol=0.0, atol=1e-9
            )
            fraction_seed = float(matches_seed.mean())
            fraction_neighbour = float(matches_neighbour.mean())
            # Align every synthetic row with its own seed/neighbour pair;
            # membership in a global set of categories cannot establish this.
            fraction_other = float(
                (~np.all(matches_seed | matches_neighbour, axis=1)).mean()
            )
        n_zero_rows = int(
            (
                np.isclose(one_hot_adasyn.sum(axis=1), 0.0, atol=1e-12)
            ).sum()
        )
        unique_local, counts_local = np.unique(rows_local, return_counts=True)
        report["categoricaladasyn_generation"] = {
            "n_target_for_class": n_gen_target,
            "n_generated_actual": n_gen_actual,
            "rounding_note": (
                "rint of the hardness-normalised share; the summed per-seed"
                " counts may differ from the target by at most the number of"
                " minority seeds (this is standard ADASYN's documented"
                " rounding behaviour)"
            ),
            "n_seeds_used": int(len(unique_local)),
            "n_minority_seeds_available": int(len(target_class_idx)),
            "max_count_single_seed": int(counts_local.max())
            if len(counts_local)
            else 0,
            "n_zero_onehot_rows_in_training_fold": n_zero_rows,
            "category_source_fractions": {
                "same_as_seed": fraction_seed,
                "same_as_chosen_neighbour": fraction_neighbour,
                "other": fraction_other,
                "denominators": "seed/neighbour: feature cells; other: rows with any invalid inheritance",
            },
            "category_choice_rule": (
                "per feature-block argmax of the interpolated one-hot row:"
                " when the seed and neighbour levels differ, the synthetic"
                " code inherits the seed's level when step < 0.5, and the"
                " neighbour's level when step > 0.5; an exact 0.5 tie uses"
                " the first level in encoder order (never a third"
                " level; unlike SMOTENC there is no neighbourhood majority"
                " vote)"
            ),
            "replication": {
                "shape_ok": bool(shape_ok),
                "max_abs_deviation_continuous": max_deviation_continuous,
                "max_abs_deviation_categ_codes": max_deviation_codes,
                "is_exact": replication_ok,
            },
        }
        check(
            replication_ok,
            "adasyn_allocation_replication_is_exact",
            n_generated=n_gen_actual,
            max_count_single_seed=(
                int(counts_local.max()) if len(counts_local) else 0
            ),
        )
        check(
            fraction_other is not None and fraction_other == 0.0,
            "adasyn_synthetic_categories_come_only_from_seed_or_neighbour",
            fraction_seed=fraction_seed,
            fraction_neighbour=fraction_neighbour,
            fraction_other=fraction_other,
        )
        check(
            n_zero_rows == 0,
            "adasyn_no_empty_onehot_rows",
            n_zero_rows=n_zero_rows,
        )

    # -- findings (measured, never silent) -------------------------------------
    ad_share = report.get("adasyn_space", {}).get(
        "pairwise_sq_distance_decomposition", {}
    ).get("continuous_share")
    sm_share = report["smotenc_space"]["pairwise_sq_distance_decomposition"][
        "continuous_share"
    ]
    if ad_share is not None and sm_share is not None and ad_share > 0.9:
        report["findings"].append(
            {
                "id": "F1_continuous_dominance_in_adasyn_space",
                "severity": "methodological",
                "measured": {
                    "adasyn_space_continuous_share": ad_share,
                    "smotenc_space_continuous_share": sm_share,
                    "smotenc_median_std": median_std,
                },
                "detail": (
                    "In the space CategoricalADASYN computes its neighbour"
                    f" distances in, {ad_share:.1%} of the mean pairwise"
                    " squared distance comes from standardized continuous"
                    " columns; the one-hot blocks contribute"
                    f" {1.0 - ad_share:.1%}. Neighbour selection for the"
                    " adaptive weights remain continuous-dominated despite"
                    " fold-local scaling, while SMOTENC scales categorical"
                    f" mismatch to its median-std reference ({median_std:.2f})."
                    " The asymmetry describes the measured distance geometry"
                    " of this custom sampler, not a proof of a coding bug."
                ),
                "proposed_correction": (
                    "If categorical influence remains too small, consider"
                    " a separately justified and pre-specified categorical"
                    " weighting or distance metric; this requires a new"
                    " sampler version and fresh preflight validation."
                ),
            }
        )
    report["standards_compliance_note"] = (
        "Standard ADASYN operates on a purely numeric matrix."
        " CategoricalADASYN delegates to standard ADASYN on a specific mixed"
        " one-hot representation and decodes via argmax. This preflight does"
        " NOT claim equivalence of the categorical behaviour with a raw-space"
        " ADASYN; it measures and documents the actual differences"
        " (fold-local numeric standardization, one-hot distances, argmax"
        " decoding, rint allocation)."
        " The adaptive weighting itself is verifiably the standard ADASYN"
        " weighting because the full allocation is replicated exactly from"
        " its own internals."
    )
    # silence unused-import linters for objects referenced in docs only
    del smote_pipe, adasyn_pipe
    return report


# ---------------------------------------------------------------------------
# Full-run cost estimate (a range, not a promise)
# ---------------------------------------------------------------------------
def estimate_full_run_cost(
    timings: pd.DataFrame,
    *,
    folds_per_pipeline: int,
    n_pipelines_plan: int,
    oof_rows_this_preflight: int,
    oof_rows_per_pipeline_per_full_run: int,
    max_peak_delta_mb: float,
) -> dict[str, Any]:
    ok = timings[timings["status"] == "ok"]
    fit = ok["fit_seconds"].astype(float)
    predict = ok["predict_seconds"].astype(float)
    wall = ok["wall_seconds"].astype(float)
    audit_overhead_mean = float((wall - (fit + predict)).clip(lower=0).mean())
    fits = float(fit.sum()) * folds_per_pipeline
    predicts = float(predict.sum()) * folds_per_pipeline
    audits = audit_overhead_mean * len(ok) * folds_per_pipeline
    projected = fits + predicts + audits
    per_pipeline_compute = (
        ok["fit_seconds"].astype(float) + ok["predict_seconds"].astype(float)
    ) * folds_per_pipeline
    total_oof = int(oof_rows_per_pipeline_per_full_run) * int(n_pipelines_plan)
    growth = (
        float(total_oof / oof_rows_this_preflight)
        if oof_rows_this_preflight
        else None
    )
    return {
        "assumptions": [
            "executor treated as sequential: one fold attempt at a time",
            "observed folds are full size (~3,270 training rows, ~818 validation)",
            "machine state comparable to the preflight environment; no grid parallelism",
        ],
        "estimated_fold_attempts": int(n_pipelines_plan * folds_per_pipeline),
        "projected_fit_seconds": round(fits, 1),
        "projected_predict_seconds": round(predicts, 1),
        "projected_audit_overhead_seconds": round(audits, 1),
        "projected_compute_total_seconds": round(projected, 1),
        "projected_minutes_lower": round(projected * 0.85 / 60.0, 1),
        "projected_minutes_upper": round(projected * 1.35 / 60.0, 1),
        "fastest_pipeline_full_estimate_minutes": round(
            float(per_pipeline_compute.min()) / 60.0, 1
        ),
        "slowest_pipeline_full_estimate_minutes": round(
            float(per_pipeline_compute.max()) / 60.0, 1
        ),
        "oof_rows_full_run_estimate": total_oof,
        "oof_growth_factor_vs_this_preflight": growth,
        "max_observed_peak_rss_delta_mb": round(float(max_peak_delta_mb), 1),
        "memory_constraint_note": (
            "fold attempts run sequentially, so the process peak stays close"
            " to the largest single-attempt peak plus the once-materialised"
            " OOF frame; per-attempt RSS peaks are a 20 ms-interval sampled"
            " measurement"
        ),
        "wall_time_is_estimate": True,
    }


# ---------------------------------------------------------------------------
# Preflight verification (strict, mirrors the G1/G2/G3 style)
# ---------------------------------------------------------------------------
def verify_preflight_run(
    *,
    run_directory: RunDirectory,
    cv: pd.DataFrame,
    oof: pd.DataFrame,
    timings: pd.DataFrame,
    sampling_audit: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    statuses: Mapping[str, str],
    manifest: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    spec: Any,
    grid_keys: Sequence[str],
    expect_full_grid: bool,
    geometry_report: Mapping[str, Any],
    data_metadata: Mapping[str, Any],
    pin: Mapping[str, Any],
    n_splits_plan: int,
    n_repeats_plan: int,
) -> dict[str, Any]:
    """Run the strict preflight checks and return a machine-readable verdict."""
    checks: list[dict[str, Any]] = []

    def _check(name: str, condition: bool, **details: Any) -> None:
        checks.append({"check": name, "passed": bool(condition), **details})

    # -- supported grid (25 supported combinations) ---------------------------
    expected_keys = expected_supported_grid_keys()
    grid_ok = sorted(set(grid_keys)) == expected_keys
    _check(
        "grid_matches_supported_set",
        bool(grid_ok or not expect_full_grid),
        n_pipelines=len(grid_keys),
        n_expected=len(expected_keys),
        missing=sorted(set(expected_keys) - set(grid_keys))[:6],
        unexpected=sorted(set(grid_keys) - set(expected_keys))[:6],
        expect_full_grid=expect_full_grid,
    )

    # -- attempt completeness -------------------------------------------------
    failed = {key for key, status in statuses.items() if status != "ok"}
    ok_keys = sorted(set(grid_keys) - failed)
    missing_statuses = sorted(set(grid_keys) - set(statuses))
    _check(
        "all_combinations_attempted",
        not missing_statuses,
        missing=missing_statuses,
        n_failed=len(failed),
    )
    _check(
        "expected_attempt_count_recorded",
        len(cv) == len(grid_keys) and len(timings) == len(grid_keys),
        n_attempt_rows=int(len(cv)),
        n_timing_rows=int(len(timings)),
        n_grid=len(grid_keys),
        scope="one fold attempt per model/strategy combination (25 when full)",
    )
    _check(
        "single_shared_fold_used",
        bool(
            not cv.empty
            and cv["repeat"].eq(int(spec.repeat)).all()
            and cv["fold"].eq(int(spec.fold)).all()
            and cv["train_index_fingerprint"].nunique() == 1
            and cv["validation_index_fingerprint"].nunique() == 1
        ),
        executed_repeat=int(spec.repeat),
        executed_fold=int(spec.fold),
        distinct_train_fingerprints=(
            int(cv["train_index_fingerprint"].nunique()) if not cv.empty else 0
        ),
        distinct_validation_fingerprints=(
            int(cv["validation_index_fingerprint"].nunique())
            if not cv.empty
            else 0
        ),
    )
    _check(
        "fold_split_matches_plan_shape",
        bool(
            int(cv["n_validation"].max()) == int(cv["n_validation"].min())
            and int(plan_info["n_splits"]) == int(n_splits_plan)
            and int(plan_info["n_repeats"]) == int(n_repeats_plan)
        ),
        n_validation=int(cv["n_validation"].iloc[0]) if not cv.empty else 0,
        plan_splits=int(plan_info["n_splits"]),
        plan_repeats=int(plan_info["n_repeats"]),
    )

    # -- timings & memory -----------------------------------------------------
    ok_rows = timings[timings["status"] == "ok"]
    _check(
        "fit_and_predict_times_recorded",
        bool(
            len(ok_rows) == len(ok_keys)
            and (ok_rows["fit_seconds"].astype(float) > 0).all()
            and (ok_rows["predict_seconds"].astype(float) > 0).all()
        ),
        n_rows=int(len(ok_rows)),
    )
    _check(
        "memory_probe_recorded",
        bool(
            ok_rows["rss_peak_mb"].notna().all()
            and (
                ok_rows["rss_peak_mb"].astype(float)
                >= ok_rows["rss_baseline_mb"].astype(float)
            ).all()
        ),
        probe_interval_s=PROBE_INTERVAL_S,
        max_peak_delta_mb=(
            round(
                float(ok_rows["rss_peak_delta_mb"].astype(float).max()), 1
            )
            if len(ok_rows)
            else 0.0
        ),
        limitation="20 ms interval sampling; sub-interval spikes can be missed",
    )

    # -- resampling audits ----------------------------------------------------
    audit_issues: list[str] = []
    for record in sampling_audit.to_dict(orient="records"):
        if record.get("audit_status") != "PASS":
            audit_issues.append(
                f"{record.get('pipeline')}: audit status {record.get('audit_status')}"
            )
    expected_audited = {
        key for key in grid_keys if key.partition("__")[2] in RESAMPLING_STRATEGIES
    }
    audited_pipelines = (
        set(sampling_audit["pipeline"]) if not sampling_audit.empty else set()
    )
    missing_audits = expected_audited - audited_pipelines
    if missing_audits:
        audit_issues.append(f"no audit rows for: {sorted(missing_audits)}")
    _check(
        "resampling_audited_legally",
        not audit_issues,
        issues=audit_issues[:12],
        n_audits=int(len(sampling_audit)),
        expected_audits=len(expected_audited),
    )

    # -- warnings and errors (G3 machinery, reused classifier) -----------------
    svc_keys = {key for key in grid_keys if key.startswith("svc_rbf")}
    svc_folds = 0 if cv.empty else int(cv["pipeline"].isin(svc_keys).sum())
    svc_warning_count = (
        0
        if warnings_table.empty
        else int(
            (
                (warnings_table["category"] == "FutureWarning")
                & warnings_table["message"]
                .str.startswith(EXPECTED_WARNING_MESSAGE_PREFIX, na=False)
                & warnings_table["pipeline"].isin(svc_keys)
            ).sum()
        )
    )
    warning_issues: list[str] = []
    for record in warnings_table.to_dict(orient="records"):
        status, reason = classify_warning(record)
        if status != "EXPECTED":
            warning_issues.append(
                f"{record.get('pipeline', '?')} [{reason}]: unexpected warning"
            )
    _check(
        "warnings_expected_and_counted",
        not warning_issues and svc_warning_count == svc_folds,
        issues=warning_issues[:12],
        svc_fold_fits=svc_folds,
        captured_svc_probability_warnings=svc_warning_count,
        allow_list=sorted(EXPECTED_WARNING_CATEGORIES),
        message_prefix=EXPECTED_WARNING_MESSAGE_PREFIX,
    )
    error_pipelines = (
        set(errors_table["pipeline"]) if not errors_table.empty else set()
    )
    field_complete = all(
        str(getattr(row, "error_type", "") or "").strip()
        and str(getattr(row, "message", "") or "").strip()
        and str(getattr(row, "traceback", "") or "").strip()
        for row in errors_table.itertuples()
    )
    _check(
        "errors_reported_consistently",
        error_pipelines == failed and field_complete,
        error_pipelines=sorted(error_pipelines),
        failed_pipelines=sorted(failed),
    )

    # -- probabilities --------------------------------------------------------
    prob_issues: list[str] = []
    n_valid = int(cv["n_validation"].iloc[0]) if not cv.empty else 0
    per_pipeline_counts = (
        oof.groupby("pipeline").size().to_dict() if not oof.empty else {}
    )
    incomplete = sorted(
        key
        for key in ok_keys
        if int(per_pipeline_counts.get(key, 0)) != n_valid
    )
    if incomplete:
        prob_issues.append(f"incomplete OOF for: {incomplete[:6]}")
    if not oof.empty:
        values = oof["y_proba"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            prob_issues.append("non-finite probabilities present")
        if bool(((values < 0.0) | (values > 1.0)).any()):
            prob_issues.append("probabilities outside [0, 1]")
    _check(
        "probabilities_valid_and_complete",
        not prob_issues,
        issues=prob_issues[:6],
        n_oof_rows=int(len(oof)),
        n_complete_pipelines=sum(
            1 for count in per_pipeline_counts.values() if int(count) == n_valid
        ),
    )

    # -- manifest / provenance (G2 machinery, reused helpers) ------------------
    manifest_issues: list[str] = []
    if not HEX64_PATTERN.match(
        str(manifest.get("data", {}).get("data_sha256", ""))
    ):
        manifest_issues.append("data.data_sha256 missing or malformed")
    for key in ("source_files", "source_files_start", "source_files_end"):
        block = manifest.get(key, {})
        if not isinstance(block, dict) or any(
            not HEX64_PATTERN.match(str(value)) for value in block.values()
        ):
            manifest_issues.append(f"{key} incomplete or malformed")
    config_hashes = manifest.get("configuration_hashes", {})
    if not config_hashes or not all(
        HEX64_PATTERN.match(str(value or ""))
        for value in config_hashes.values()
    ):
        manifest_issues.append("configuration_hashes incomplete")
    fingerprints = manifest.get("fold_plan", {}).get(
        "repeat_assignment_fingerprints", {}
    )
    if not fingerprints:
        manifest_issues.append("fold_plan.repeat_assignment_fingerprints missing")
    if not HEX64_PATTERN.match(str(pin.get("sha256", ""))):
        manifest_issues.append("repeatability pin missing")
    if "pipeline_parameters" not in manifest:
        manifest_issues.append("pipeline_parameters missing")
    _check(
        "manifest_preflight_complete",
        not manifest_issues,
        issues=manifest_issues,
        scope="PREFLIGHT-ONLY manifest; never a publication manifest",
    )
    changed = diff_source_hashes(
        manifest.get("source_files_start", {}),
        manifest.get("source_files_end", {}),
    )
    _check("source_freeze_stable", len(changed) == 0, changed_files=changed)

    # -- geometry -------------------------------------------------------------
    needed_sections = (
        "representation",
        "smotenc_space",
        "smotenc_generation",
        "adasyn_space",
        "categoricaladasyn_generation",
        "categoricaladasyn_hardness",
        "checks",
        "findings",
        "standards_compliance_note",
    )
    geometry_sections_ok = all(
        section in geometry_report for section in needed_sections
    )
    geometry_check_names = {
        entry["check"]
        for entry in geometry_report.get("checks", [])
        if entry["passed"]
    }
    needed_checks = {
        "smotenc_distance_decomposition_is_finite",
        "smotenc_final_class_count_matches_strategy_target",
        "adasyn_distance_decomposition_is_finite",
        "adasyn_allocation_replication_is_exact",
        "adasyn_synthetic_categories_come_only_from_seed_or_neighbour",
        "adasyn_representation_matches_fitted_neighbours",
        "smotenc_synthetic_codes_integer_and_valid",
        "adasyn_no_empty_onehot_rows",
    }
    failed_geometry_checks = [
        entry["check"] for entry in geometry_report.get("checks", [])
        if not entry["passed"]
    ]
    _check(
        "sampling_geometry_verified",
        geometry_sections_ok and needed_checks.issubset(geometry_check_names)
        and not failed_geometry_checks,
        failed_geometry_checks=failed_geometry_checks,
        missing_sections=sorted(set(needed_sections) - set(geometry_report)),
        missing_geometry_checks=sorted(needed_checks - geometry_check_names),
        findings_reported=geometry_report.get("findings", []),
        non_equivalence_statement=bool(
            geometry_report.get("standards_compliance_note")
        ),
    )

    # -- lock / scope ---------------------------------------------------------
    _check("all_pipelines_fit_successfully", not failed, failed_pipelines=sorted(failed))
    _check(
        "heldout_test_locked",
        bool(data_metadata.get("heldout_test_was_locked"))
        and not bool(data_metadata.get("heldout_test_data_fields_accessed")),
        scope="training partition only",
    )

    n_passed = sum(1 for check in checks if check["passed"])
    n_failed = len(checks) - n_passed
    return {
        "run_id": run_directory.run_id,
        "created_utc": utc_timestamp(),
        "overall_passed": n_failed == 0,
        "n_checks": len(checks),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "checks": checks,
        "scope_note": (
            "PREFLIGHT verification: structural and scientific gates only;"
            " no model selection verdicts and no tuning anywhere."
        ),
    }


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------
def _draft_preflight_manifest(
    *,
    run_directory: RunDirectory,
    data_metadata: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    pipeline_parameters: Sequence[Mapping[str, Any]],
    pipeline_status: Mapping[str, str],
    pipeline_fold_counts: Mapping[str, Mapping[str, int]],
    timings_summary: Mapping[str, Any],
    pin: Mapping[str, Any],
    grid_keys: Sequence[str],
    source_hashes_start: Mapping[str, str],
    geometry_summary: Mapping[str, Any],
    expected_full_run_attempts: int,
) -> dict[str, Any]:
    return {
        "run_id": run_directory.run_id,
        "step": PREFLIGHT_STEP,
        "category": "preflight_not_for_publication",
        "created_utc": run_directory.created_utc,
        "scope": (
            "PREFLIGHT: one full-size fold per pipeline on the training"
            " partition. Diagnostic timing/memory/geometry verification"
            " only; results must not be used to select models, strategies,"
            " thresholds or sampling ratios."
        ),
        "environment": core_environment(),
        "source_files": source_file_hashes(),
        "source_files_start": dict(source_hashes_start),
        "source_files_end": source_file_hashes(),
        "configuration_hashes": configuration_file_hashes(),
        "data": dict(data_metadata),
        "heldout_test": {
            "locked": bool(data_metadata.get("heldout_test_was_locked")),
            "data_fields_accessed": bool(
                data_metadata.get("heldout_test_data_fields_accessed")
            ),
            "note": "the prohibited held-out partition is never read here",
        },
        "fold_plan": dict(plan_info),
        "preflight_scope": {
            "n_splits_plan": int(plan_info["n_splits"]),
            "n_repeats_plan": int(plan_info["n_repeats"]),
            "executed_repeat": 1,
            "executed_fold": 1,
            "attempts_expected": len(grid_keys),
            "full_run_attempts_reference": int(expected_full_run_attempts),
        },
        "cv": {
            "cv_random_state": int(IMBALANCE_RANDOM_STATE),
            "decision_threshold": float(DECISION_THRESHOLD),
        },
        "sampling": {
            "sampling_strategy_ratio": float(IMBALANCE_DEFAULT_SAMPLING_RATIO),
            "k_neighbors": int(IMBALANCE_DEFAULT_K_NEIGHBORS),
            "sampler_seed": int(IMBALANCE_RANDOM_STATE),
            "resampling_strategies": list(RESAMPLING_STRATEGIES),
        },
        "pipeline_parameters": list(pipeline_parameters),
        "pipeline_status": dict(pipeline_status),
        "pipeline_fold_counts": {
            key: dict(counts) for key, counts in pipeline_fold_counts.items()
        },
        "timings_summary": dict(timings_summary),
        "geometry_summary": dict(geometry_summary),
        "repeatability_pin": dict(pin),
        "supported_grid_keys": list(grid_keys),
        "outcomes": {},
        "total_seconds": None,
        "verification": {},
        "files": {},
    }


def _write_preflight_report(
    path: Path,
    *,
    run_directory: RunDirectory,
    data_metadata: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    timings: pd.DataFrame,
    cv: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    geometry_report: Mapping[str, Any],
    cost_estimate: Mapping[str, Any],
    verification: Mapping[str, Any],
    total_seconds: float,
) -> None:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "recall_sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "brier_score",
    ]
    ok_rows = timings[timings["status"] == "ok"]
    sorted_ok = ok_rows.sort_values("fit_seconds", ascending=False)

    counts_table = cv[
        [
            "pipeline",
            "n_train_before_sampling",
            "n_train_after_sampling",
            "n_validation",
        ]
    ].copy()
    ratios: list[Any] = []
    for record in cv.to_dict(orient="records"):
        after = record.get("sampling_class_counts_after")
        if isinstance(after, dict) and after.get("0"):
            values = float(after["0"])
            ratios.append(
                round(float(after.get("1", 0.0)) / values, 4)
                if values > 0
                else ""
            )
        else:
            ratios.append("")
    counts_table["actual_minority_majority_ratio_after"] = ratios

    adasyn_generation = geometry_report.get("categoricaladasyn_generation", {})
    adasyn_replication = adasyn_generation.get("replication", {})
    warn_categories = (
        warnings_table["category"].value_counts().to_dict()
        if not warnings_table.empty
        else {}
    )
    status_text = "PASSED" if verification["overall_passed"] else "FAILED"
    ad_decomp = geometry_report["adasyn_space"][
        "pairwise_sq_distance_decomposition"
    ]
    sm_decomp = geometry_report["smotenc_space"][
        "pairwise_sq_distance_decomposition"
    ]
    lines = [
        "# Step 32 / G9 — Full-size timing preflight (PREFLIGHT)",
        "",
        "> **PREFLIGHT ONLY.** These diagnostics measure runtime, memory and",
        "> the sampling geometry. They are **not publication results** and",
        "> must not be used to select models, strategies, thresholds or",
        "> sampling ratios.",
        "",
        f"- run_id: `{run_directory.run_id}`",
        f"- created_utc: {run_directory.created_utc}",
        f"- verification: **{status_text}** ({verification['n_passed']}/"
        f"{verification['n_checks']} checks)",
        f"- total wall seconds: {total_seconds:.2f}",
        "",
        "## Data",
        "",
        f"- training rows: {data_metadata.get('n_train')}",
        f"- raw data sha256: `{data_metadata.get('data_sha256')}`",
        f"- held-out test locked: {data_metadata.get('heldout_test_was_locked')}"
        " (never read)",
        "",
        "## Fold plan (original 5-fold stratified CV, seed 42)",
        "",
        f"- plan: {plan_info['n_splits']} splits x {plan_info['n_repeats']} repeat"
        f" ({plan_info['n_folds_total']} folds total)",
        "- executed: exactly one fold (repeat 1, fold 1), the SAME split for"
        " all 25 combinations",
        "- first-repeat fingerprint equals the original 5x5 plan: "
        f"`{plan_info.get('first_repeat_equals_original_plan')}`",
        f"- validation rows in the executed fold:"
        f" {int(cv['n_validation'].iloc[0]) if not cv.empty else 0}",
        "",
        "## Timing and memory per combination (ok attempts)",
        "",
        "### Slowest first (fit seconds)",
        "",
        _markdown_table(
            sorted_ok.assign(
                fit_seconds=sorted_ok["fit_seconds"].round(3),
                predict_seconds=sorted_ok["predict_seconds"].round(4),
            ),
            ["pipeline", "fit_seconds", "predict_seconds", "rss_peak_delta_mb"],
        ),
        "### Memory (peak RSS delta from baseline, MB)",
        "",
        _markdown_table(
            sorted_ok.sort_values("rss_peak_delta_mb", ascending=False),
            ["pipeline", "rss_baseline_mb", "rss_peak_mb", "rss_peak_delta_mb"],
        ),
        "## Class counts and actual sampling ratios",
        "",
        "Recorded by the RecordingSampler inside each fitted pipeline; the"
        " ratio is the achieved minority/majority count after resampling.",
        "",
        _markdown_table(
            counts_table,
            [
                "pipeline",
                "n_train_before_sampling",
                "n_train_after_sampling",
                "actual_minority_majority_ratio_after",
            ],
        ),
        "## Diagnostic metrics per combination (NOT for selection)",
        "",
        _markdown_table(cv, ["pipeline", *metric_columns]),
        "## Warnings and errors",
        "",
        f"- warning categories: {warn_categories or 'none'}",
        f"- failed attempts: {len(errors_table)}",
        "",
        "## Sampling geometry verification (SMOTENC / CategoricalADASYN)",
        "",
        "### Representation",
        "",
        f"- encoder: `{geometry_report['representation']['encoder_class']}`"
        f" ({geometry_report['representation']['n_rows']} x"
        f" {geometry_report['representation']['n_columns']})",
        "- continuous block:"
        f" {geometry_report['representation']['feature_block']['continuous']}",
        "- discrete block:"
        f" {geometry_report['representation']['feature_block']['discrete']}"
        " (" + geometry_report["representation"]["feature_block"][
            "discrete_representation"
        ] + ")",
        "- SMOTENC internal: one-hot + median-std substitution (median std:"
        f" {geometry_report['smotenc_space']['median_std_reference']:.4f})",
        f"- CategoricalADASYN internal: fold-standardized continuous + one-hot + ADASYN"
        f" ({geometry_report['representation']['categoricaladasyn_internal_space']['delegates_to']})",
        "",
        "### Distance decomposition (mean pairwise squared-distance shares)",
        "",
        f"- SMOTENC space (minority rows, scaled one-hot): continuous"
        f" {sm_decomp['continuous_share']:.1%} vs discrete"
        f" {sm_decomp['discrete_share']:.1%}",
        f"- CategoricalADASYN space (all rows, raw one-hot): continuous"
        f" {ad_decomp['continuous_share']:.1%} vs discrete"
        f" {ad_decomp['discrete_share']:.1%}",
        "",
        "### Synthetic-sample semantics (measured)",
        "",
        f"- SMOTENC: {geometry_report['smotenc_generation']['n_synthetic']}"
        " synthetic rows; category blocks all observed:"
        f" `{geometry_report['smotenc_generation']['synthetic_category_blocks_all_observed']}`;"
        " categorical values copied, never interpolated",
        f"- CategoricalADASYN:"
        f" {adasyn_generation.get('n_generated_actual')} synthetic rows;"
        " exact allocation replication:"
        f" `{adasyn_replication.get('is_exact')}`",
        "- category source fractions:"
        f" {adasyn_generation.get('category_source_fractions')}",
        "- hardness-weight sensitivity (cosine similarity with continuous-only weights):"
        f" {geometry_report['categoricaladasyn_hardness']['cosine_similarity_full_vs_continuous_only']}",
        "",
        "### Geometry checks",
        "",
        _markdown_table(
            pd.DataFrame(geometry_report.get("checks", [])),
            ["check", "passed"],
        ),
        "### Findings",
        "",
    ]
    for finding in geometry_report.get("findings", []):
        lines += [
            f"- **{finding['id']}** ({finding['severity']}):",
            f"  - {finding['detail']}",
            f"  - proposed correction: {finding.get('proposed_correction', 'n/a')}",
        ]
    lines += [
        "",
        f"> Equivalence note: {geometry_report.get('standards_compliance_note', '')}",
        "",
        "## Step 31 machinery validated in this preflight",
        "",
        _markdown_table(
            pd.DataFrame(verification["checks"])[["check", "passed"]],
            ["check", "passed"],
        ),
        "*The expected 25-attempt count is one fold per supported combination"
        " (25), NOT the full 625-fold run.*",
        "",
        "## Full-run cost estimate (range, not a promise)",
        "",
        "- projected compute total:"
        f" {cost_estimate['projected_compute_total_seconds']} s"
        f" ({cost_estimate['projected_minutes_lower']}-{cost_estimate['projected_minutes_upper']} min range)",
        "- OOF rows in the full run: ~"
        f"{cost_estimate['oof_rows_full_run_estimate']}"
        f" (x{cost_estimate['oof_growth_factor_vs_this_preflight']} this preflight)",
        "- largest observed per-attempt peak RSS delta:"
        f" {cost_estimate['max_observed_peak_rss_delta_mb']} MB",
        "",
        "## Limitations",
        "",
        "- a single fold per combination: timings carry fold-level variance and",
        "  the metrics carry no selection information (diagnostic only, fixed",
        "  0.5 threshold)",
        "- the memory probe samples at 20 ms; sub-interval spikes can be missed",
        "- all preflight artifacts live in this isolated directory, clearly",
        "  marked; publication-grade imbalance outputs remain reserved for the",
        "  run-scoped CV layer (results/imbalance_cv/<run_id>/)",
        "- Step 31 source hashes, output isolation, manifest completeness,",
        "  warning/error handling and the 25-attempt gate are validated by the",
        "  checks reported above",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_full_size_preflight(
    *,
    results_dir: Path | None = None,
    run_id: str | None = None,
    expect_full_grid: bool = True,
    random_state: int = IMBALANCE_RANDOM_STATE,
    threshold: float = DECISION_THRESHOLD,
    n_splits: int = 5,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Run preflight and release run-local resources even on failure."""
    global PREFLIGHT_RESULTS_DIR
    original_results_dir = PREFLIGHT_RESULTS_DIR
    actual_run_id = build_run_id("preflight") if run_id is None else run_id
    try:
        return _run_full_size_preflight(
            results_dir=results_dir, run_id=actual_run_id,
            expect_full_grid=expect_full_grid, random_state=random_state,
            threshold=threshold, n_splits=n_splits, n_repeats=n_repeats,
        )
    finally:
        PREFLIGHT_RESULTS_DIR = original_results_dir
        logger = logging.getLogger(f"preflight_full_size.{actual_run_id}")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _run_full_size_preflight(
    *,
    results_dir: Path | None = None,
    run_id: str | None = None,
    expect_full_grid: bool = True,
    random_state: int = IMBALANCE_RANDOM_STATE,
    threshold: float = DECISION_THRESHOLD,
    n_splits: int = 5,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Execute the G9 preflight end to end and persist every artifact."""
    global PREFLIGHT_RESULTS_DIR
    if results_dir is not None:
        PREFLIGHT_RESULTS_DIR = Path(results_dir)

    started = time.perf_counter()
    X, y, data_metadata = load_training_partition()
    run_directory = create_preflight_directory(run_id)
    logger = _configure_logger(run_directory.log_path(), run_directory.run_id)
    logger.info("=" * 76)
    logger.info("StrokeSure G9 full-size preflight (one fold per pipeline)")
    logger.info("run_id=%s training_rows=%d", run_directory.run_id, len(X))
    logger.info("=" * 76)
    source_hashes_start = source_file_hashes()

    # fold plan: the original 5-fold plan; fingerprint-match the executed
    # repeat against the first repeat of the documented original 5x5 plan.
    plan = build_fold_plan(
        X, y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    plan_info = dict(
        validate_shared_fold_plan(
            plan, n_rows=len(X), n_splits=n_splits, n_repeats=n_repeats
        )
    )
    full_plan = build_fold_plan(
        X, y, n_splits=n_splits, n_repeats=5, random_state=random_state
    )
    full_plan_info = validate_shared_fold_plan(
        full_plan, n_rows=len(X), n_splits=n_splits, n_repeats=5
    )
    plan_info["cv_random_state"] = int(random_state)
    plan_info["n_splits_documented"] = int(n_splits)
    plan_info["n_repeats_documented"] = int(n_repeats)
    plan_info["first_repeat_equals_original_plan"] = bool(
        plan_info.get("repeat_assignment_fingerprints", {}).get(n_repeats)
        == full_plan_info.get("repeat_assignment_fingerprints", {}).get(1)
    )
    plan_info["equivalence_note"] = (
        "The executed repeat is fingerprint-matched to the first repeat of"
        " RepeatedStratifiedKFold(5 x 5, random_state=42): it re-uses the"
        " original plan's fold structure verbatim, not a new split."
    )

    grid = build_audited_grid(supported_grid_keys())
    grid_keys = list(grid)

    spec = plan[0]
    attempt = run_preflight_attempts(
        X, y, grid, spec, threshold=threshold, logger=logger
    )
    cv, timings = attempt["cv"], attempt["timings"]

    geometry_report = analyze_sampling_geometry(
        X, y, spec, random_state=random_state
    )

    statuses = attempt["statuses"]
    pipeline_fold_counts = {
        key: {
            "successful": 1 if statuses.get(key) == "ok" else 0,
            "failed": 0 if statuses.get(key) == "ok" else 1,
        }
        for key in grid_keys
    }
    pin = _repeatability_pin(
        step=PREFLIGHT_STEP,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        threshold=threshold,
        sampling_ratio=IMBALANCE_DEFAULT_SAMPLING_RATIO,
        k_neighbors=IMBALANCE_DEFAULT_K_NEIGHBORS,
        sampler_seed=IMBALANCE_RANDOM_STATE,
        grid_keys=grid_keys,
        n_rows=len(X),
    )
    ok_rows = timings[timings["status"] == "ok"]
    timings_summary = {
        "total_fit_seconds": round(float(ok_rows["fit_seconds"].sum()), 3),
        "total_predict_seconds": round(
            float(ok_rows["predict_seconds"].sum()), 3
        ),
        "max_peak_rss_delta_mb": round(
            float(ok_rows["rss_peak_delta_mb"].max()), 2
        ),
    }
    cost_estimate = estimate_full_run_cost(
        timings,
        folds_per_pipeline=int(full_plan_info["n_folds_total"]),
        n_pipelines_plan=len(grid_keys),
        oof_rows_this_preflight=int(len(attempt["oof"])),
        oof_rows_per_pipeline_per_full_run=int(len(X)) * int(
            full_plan_info["n_repeats"]
        ),
        max_peak_delta_mb=float(ok_rows["rss_peak_delta_mb"].max()),
    )
    manifest = _draft_preflight_manifest(
        run_directory=run_directory,
        data_metadata=data_metadata,
        plan_info=plan_info,
        pipeline_parameters=describe_pipelines(grid),
        pipeline_status=statuses,
        pipeline_fold_counts=pipeline_fold_counts,
        timings_summary=timings_summary,
        pin=pin,
        grid_keys=grid_keys,
        source_hashes_start=source_hashes_start,
        geometry_summary={
            "n_geometry_checks": len(geometry_report["checks"]),
            "n_findings": len(geometry_report["findings"]),
            "finding_ids": [
                finding["id"] for finding in geometry_report["findings"]
            ],
        },
        expected_full_run_attempts=int(full_plan_info["n_folds_total"])
        * len(grid_keys),
    )

    paths = {
        "fold_metrics": run_directory.csv_path("fold_metrics"),
        "oof_predictions": run_directory.csv_path("oof_predictions"),
        "timings": run_directory.csv_path("timings"),
        "sampling_audit": run_directory.csv_path("sampling_audit"),
        "warnings": run_directory.csv_path("warnings"),
        "errors": run_directory.csv_path("errors"),
        "sampling_geometry": run_directory.json_path("sampling_geometry"),
        "fold_plan": run_directory.json_path("fold_plan"),
        "verification": run_directory.json_path("verification"),
        "manifest": run_directory.json_path("manifest"),
        "report": run_directory.report_path(),
    }
    for key, frame in (
        ("fold_metrics", cv),
        ("oof_predictions", attempt["oof"]),
        ("timings", timings),
        ("sampling_audit", attempt["sampling_audit"]),
        ("warnings", attempt["warnings"]),
        ("errors", attempt["errors"]),
    ):
        _write_frame(frame, paths[key])
    write_json(
        geometry_report,
        paths["sampling_geometry"],
    )
    write_json(
        {
            "fold_plan": plan_info,
            "executed_spec": {
                "repeat": int(spec.repeat),
                "fold": int(spec.fold),
                "n_train": int(len(spec.train_positions)),
                "n_validation": int(len(spec.valid_positions)),
            },
            "plan_equivalence": plan_info["first_repeat_equals_original_plan"],
        },
        paths["fold_plan"],
    )

    verification = verify_preflight_run(
        run_directory=run_directory,
        cv=cv,
        oof=attempt["oof"],
        timings=timings,
        sampling_audit=attempt["sampling_audit"],
        warnings_table=attempt["warnings"],
        errors_table=attempt["errors"],
        statuses=statuses,
        manifest=manifest,
        plan_info=plan_info,
        spec=spec,
        grid_keys=grid_keys,
        expect_full_grid=expect_full_grid,
        geometry_report=geometry_report,
        data_metadata=data_metadata,
        pin=pin,
        n_splits_plan=n_splits,
        n_repeats_plan=n_repeats,
    )
    write_json(verification, paths["verification"])

    total_seconds = time.perf_counter() - started
    manifest["verification"] = {
        "overall_passed": verification["overall_passed"],
        "n_checks": verification["n_checks"],
        "n_passed": verification["n_passed"],
        "n_failed": verification["n_failed"],
        "path": str(paths["verification"]),
    }
    manifest["outcomes"] = {
        "n_attempts": int(len(cv)),
        "n_oof_rows": int(len(attempt["oof"])),
        "n_audits": int(len(attempt["sampling_audit"])),
        "n_warnings": int(len(attempt["warnings"])),
        "n_errors": int(len(attempt["errors"])),
        "n_failed_attempts": int(
            sum(counts["failed"] for counts in pipeline_fold_counts.values())
        ),
    }
    manifest["total_seconds"] = round(float(total_seconds), 3)
    # Write the preflight report FIRST, before hashing artifacts.
    _write_preflight_report(
        paths["report"],
        run_directory=run_directory,
        data_metadata=data_metadata,
        plan_info=plan_info,
        timings=timings,
        cv=cv,
        warnings_table=attempt["warnings"],
        errors_table=attempt["errors"],
        geometry_report=geometry_report,
        cost_estimate=cost_estimate,
        verification=verification,
        total_seconds=total_seconds,
    )

    # Create the artifact index after all files have been saved.
    files: dict[str, Any] = {
        "manifest": {
            "path": str(paths["manifest"]),
            "sha256": None,
        },
        "manifest_note": (
            "manifest.json cannot hash itself; every other "
            "artifact's SHA-256 is listed."
        ),
    }

    for key, path_obj in paths.items():
        if key == "manifest":
            continue

        files[key] = {
            "path": str(path_obj),
            "sha256": file_sha256(path_obj),
        }

    manifest["files"] = files

    # Save the manifest LAST.
    write_json(manifest, paths["manifest"])
    logger.info(
        "run %s preflight verification: %s (%d/%d checks)",
        run_directory.run_id,
        "PASSED" if verification["overall_passed"] else "FAILED",
        verification["n_passed"],
        verification["n_checks"],
    )
    logger.info(
        "total wall time %.2fs (including artifact persistence)", total_seconds
    )
    return {
        "run_id": run_directory.run_id,
        "run_directory": run_directory,
        "paths": paths,
        "cv": cv,
        "oof": attempt["oof"],
        "timings": timings,
        "sampling_audit": attempt["sampling_audit"],
        "warnings_table": attempt["warnings"],
        "errors_table": attempt["errors"],
        "statuses": statuses,
        "geometry_report": geometry_report,
        "cost_estimate": cost_estimate,
        "plan_info": plan_info,
        "spec": spec,
        "verification": verification,
        "manifest": manifest,
        "overall_passed": verification["overall_passed"],
    }
