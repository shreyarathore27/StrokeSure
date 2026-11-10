"""Run-scoped, isolation-checked persistence layer for imbalance experiments.

This module implements the pending gap definitions from the Step 26-29
research-readiness report (``results/step26_29_research_readiness_report.md``):

G1 - run-scoped persistence
    Every artifact is written to a *fresh* ``results/imbalance_cv/<run_id>/``
    directory. The directory must not exist beforehand (runs never overwrite
    one another) and it must not overlap any protected project path: baseline
    tables/figures/models, the pilot artifacts, Notebooks 01-03, the raw
    dataset, ``logs/``, source code or tests. Baseline artifact file names
    (``baseline_*``) are rejected outright.

G2 - source and configuration provenance
    The experiment manifest records SHA-256 hashes of every runner module
    (captured at run start *and* run end, with changes flagged by the
    ``source_freeze_stable`` verification check), the configuration inputs
    (``requirements.txt``), the raw-data hash, full package versions
    (including a ``pip freeze`` capture), the git state (commit, branch,
    dirty flag, status), the CV fold-plan fingerprints, CV configuration,
    sampling parameters and per-pipeline model parameters.
    Note: pilot artifacts (``results/pilot_imbalance/``) are preserved as-is;
    their manifests were produced before source hashing existed and are *not*
    retroactively attributed to the current source hashes.

G3 - strict warning and error capture
    Warnings and exceptions are captured per model-strategy-fold with a full
    record (pipeline, repeat, fold, category, message, filename, line /
    error_type, message, traceback). Failed folds are reported individually
    and marked the containing pipeline as ``failed``; the pipeline is then
    excluded from the OOF pool. Unexpected warning categories (anything that
    is not the *exactly matched* documented ``SVC(probability=...)``
    FutureWarning), numerical warnings, failed folds or invalid probabilities
    fail the run's verification. Nothing counts as successful silently;
    ``pipeline_status`` and ``errors.csv`` always agree.

The module is agnostic about *which* supported grid the caller passes: the
full 25-pipeline 5x5 experiment (25 x 25 = 625 fold evaluations, 511,000 OOF
prediction rows) and small synthetic-data unit tests use the same code path.
It never reads the held-out test partition.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback as traceback_module
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import (
    CV_N_REPEATS,
    CV_N_SPLITS,
    DATA_DIR,
    DECISION_THRESHOLD,
    FIGURES_DIR,
    IMBALANCE_DEFAULT_K_NEIGHBORS,
    IMBALANCE_DEFAULT_SAMPLING_RATIO,
    IMBALANCE_RANDOM_STATE,
    LOGS_DIR,
    METRIC_LABELS,
    METRIC_NAMES,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RESULTS_DIR,
    SMOKE_DIR,
    TABLES_DIR,
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
from src.imbalance import STRATEGIES, supports_strategy
from src.imbalance_experiment import (
    _index_fingerprint,
    run_imbalance_cv,
    validate_shared_fold_plan,
)
from src.imbalance_pipelines import (
    RecordingSampler,
    class_counts,
)
from src.models import BASELINE_MODEL_SPECS
from src.preflight import FoldSpec, build_fold_plan
from src.preprocessing import BINARY_FEATURES, CATEGORY_VALUES, NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUN_PREFIX = "imbalance_cv"
RUN_DIRECTORY_PARENT = RESULTS_DIR / "imbalance_cv"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")

RESAMPLING_STRATEGIES = (
    "random_over_sampler",
    "random_under_sampler",
    "smote",
    "adasyn",
)

# G2: configuration inputs whose content is frozen alongside the sources.
CONFIGURATION_FILES: tuple[str, ...] = (
    "requirements.txt",
)

# G3: the only warning category a healthy run may capture, and only when the
# message exactly matches the documented SVC deprecation. Anything else
# (numerical, convergence, a different deprecation) must fail verification.
EXPECTED_WARNING_CATEGORIES = frozenset({"FutureWarning"})
EXPECTED_WARNING_MESSAGE_PREFIX = (
    "The `probability` parameter was deprecated in 1.9"
)

# G2: every module whose source participates in producing results.
RUNNER_SOURCE_FILES: tuple[str, ...] = (
    "src/config.py",
    "src/data_loading.py",
    "src/evaluation.py",
    "src/models.py",
    "src/preprocessing.py",
    "src/preflight.py",
    "src/imbalance.py",
    "src/imbalance_pipelines.py",
    "src/imbalance_experiment.py",
    "src/imbalance_outputs.py",
    "src/preflight_full_size.py",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class OutputIsolationError(RuntimeError):
    """A write attempt escapes the run directory or overlaps protected paths."""


class RunConfigurationError(RuntimeError):
    """The experiment configuration cannot be run as specified."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _relative(path: Path) -> str:
    """Project-relative POSIX path; falls back to absolute outside the project."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return resolved.as_posix()
    return str(relative).replace("\\", "/")


def _counts_text(counts: Mapping[Any, Any]) -> str:
    return json.dumps(
        {str(key): int(value) for key, value in sorted(counts.items())},
        sort_keys=True,
    )


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe representation of an arbitrary parameter value."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def file_sha256(path: Path) -> str:
    """Streamed SHA-256 of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pip_freeze() -> list[str]:
    """Full installed-package snapshot (``name==version`` strings), sorted."""
    from importlib.metadata import distributions

    frozen = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata.get("Name") is not None
    )
    return frozen


def _git_status_short() -> str:
    """Short git status output for the manifest provenance block."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )  # noqa: S603, S607 (fixed, trusted args)
    except OSError:
        return "<unavailable>"
    if completed.returncode != 0:
        return "<unavailable>"
    return completed.stdout.rstrip()


def core_environment() -> dict[str, Any]:
    """Package versions, platform and git state (extends evaluation's snapshot)."""
    from importlib.metadata import PackageNotFoundError, version

    snapshot = environment_snapshot()
    snapshot["git_status_short"] = _git_status_short()
    for package, key in (
        ("imbalanced-learn", "imbalanced_learn"),
        ("sklearn-compat", "sklearn_compat"),
    ):
        try:
            snapshot[key] = version(package)
        except PackageNotFoundError:
            snapshot[key] = None
    snapshot["pip_freeze"] = pip_freeze()
    return snapshot


def source_file_hashes() -> dict[str, str]:
    """SHA-256 of every runner module (G2 provenance freeze)."""
    hashes: dict[str, str] = {}
    for relative in RUNNER_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RunConfigurationError(
                f"runner source file {relative!r} is missing; the manifest "
                "cannot be frozen."
            )
        hashes[relative] = file_sha256(path)
    return hashes


def configuration_file_hashes() -> dict[str, str]:
    """SHA-256 of configuration inputs (requirements, config module)."""
    hashes: dict[str, str] = {}
    for relative in CONFIGURATION_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RunConfigurationError(
                f"configuration file {relative!r} is missing; the manifest "
                "cannot be frozen."
            )
        hashes[relative] = file_sha256(path)
    return hashes


def diff_source_hashes(
    start: Mapping[str, str], end: Mapping[str, str]
) -> list[str]:
    """Return the set of source files whose hash changed during a run (G2)."""
    changed: list[str] = []
    for relative in sorted(set(start) | set(end)):
        if start.get(relative) != end.get(relative):
            changed.append(relative)
    return changed


def classify_warning(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(status, reason)`` for one captured warning record.

    ``EXPECTED`` is reserved for the exact documented ``SVC(probability=True)``
    deprecation FutureWarning (category *and* message prefix must match).
    Everything else - any other category, any numerical/convergence warning,
    or an unrecognised FutureWarning text - is ``UNEXPECTED`` and fails the run.
    """
    category = str(record.get("category", ""))
    message = str(record.get("message", ""))
    if category not in EXPECTED_WARNING_CATEGORIES:
        return "UNEXPECTED", f"category {category!r} is not in the allow-list"
    if not message.startswith(EXPECTED_WARNING_MESSAGE_PREFIX):
        return (
            "UNEXPECTED",
            "FutureWarning does not exactly match the documented "
            "'probability' deprecation",
        )
    return "EXPECTED", "documented SVC(probability=True) deprecation"


# ---------------------------------------------------------------------------
# G1: protected paths and the run-directory contract
# ---------------------------------------------------------------------------
def protected_paths() -> list[Path]:
    """Every project path the imbalance writer must never create or touch."""
    candidate_roots = [
        DATA_DIR,
        LOGS_DIR,
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        TABLES_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        SMOKE_DIR,
        RESULTS_DIR / "pilot_imbalance",
        RESULTS_DIR / "reports",
        RESULTS_DIR / "step21_25_report.md",
        RESULTS_DIR / "step26_29_research_readiness_report.md",
        RESULTS_DIR / "step26_29_audit.json",
        RESULTS_DIR / "step26_29_dependencies.txt",
    ]
    protected = {path.resolve() for path in candidate_roots}
    protected -= {RUN_DIRECTORY_PARENT.resolve()}
    return sorted(protected)


def _protected_overlap_issues(directory: Path) -> list[str]:
    """Overlap issues between a directory and the protected project paths."""
    issues: list[str] = []
    resolved = Path(directory).resolve()
    if resolved.name.startswith("baseline_"):
        issues.append(f"directory name {resolved.name!r} is not allowed")
    for protected in protected_paths():
        if resolved.is_relative_to(protected):
            issues.append(f"{resolved} is inside protected path {protected}")
        elif Path(protected).is_relative_to(resolved):
            issues.append(f"{resolved} would contain protected path {protected}")
    return issues


@dataclass(frozen=True)
class RunDirectory:
    """A validated, never-overwritten output directory for one experiment."""

    run_id: str
    path: Path
    results_dir: Path
    created_utc: str
    isolation_issues: tuple[str, ...] = ()
    canonical_results_dir: bool = True

    def check_path(self, path: Path) -> Path:
        """Validate ``path`` as a direct child of this run directory."""
        resolved = Path(path).resolve()
        base = self.path.resolve()
        if resolved.parent != base:
            raise OutputIsolationError(
                f"{_relative(resolved)} is not a direct child of the run "
                f"directory {_relative(base)}; the run must not write "
                "outside its own scope."
            )
        if resolved.name.startswith("baseline_"):
            raise OutputIsolationError(
                f"artifact name {resolved.name!r} collides with baseline "
                "artifacts; refusing to write."
            )
        if not ARTIFACT_NAME_PATTERN.match(resolved.name):
            raise OutputIsolationError(
                f"artifact name {resolved.name!r} uses disallowed characters."
            )
        return resolved

    def csv_path(self, stem: str) -> Path:
        return self.check_path(self.path / f"{self.run_id}_{stem}.csv")

    def json_path(self, stem: str) -> Path:
        return self.check_path(self.path / f"{self.run_id}_{stem}.json")

    def report_path(self) -> Path:
        return self.check_path(self.path / f"{self.run_id}_report.md")

    def log_path(self) -> Path:
        return self.check_path(self.path / f"{self.run_id}.log")

    def artifact_map(self, paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
        """Project-relative location plus SHA-256 for finished artifacts."""
        index: dict[str, dict[str, Any]] = {}
        for key, path in paths.items():
            resolved = self.check_path(path)
            index[key] = {
                "path": _relative(resolved),
                "sha256": file_sha256(resolved) if Path(resolved).exists() else None,
            }
        return index


def create_run_directory(
    run_id: str | None = None,
    results_dir: Path | None = None,
) -> RunDirectory:
    """Create a fresh, isolation-checked run directory (G1).

    Parameters
    ----------
    run_id
        Optional explicit identifier; defaults to ``imbalance_cv_<UTC>``.
    results_dir
        Parent directory. Production runs must leave this as ``None`` (the
        canonical ``results/imbalance``); unit tests may point it at a
        temporary directory to keep the repository clean.
    """
    if run_id is None:
        run_id = build_run_id(RUN_PREFIX)
    if not RUN_ID_PATTERN.match(run_id):
        raise RunConfigurationError(
            f"run_id {run_id!r} contains disallowed characters."
        )
    parent = Path(results_dir) if results_dir is not None else RUN_DIRECTORY_PARENT
    target = (parent / run_id).resolve()

    issues: list[str] = []
    if results_dir is None and target.parent != RUN_DIRECTORY_PARENT.resolve():
        issues.append(
            f"run directory must live directly under "
            f"{_relative(RUN_DIRECTORY_PARENT)}; got {target}"
        )
    if target.exists():
        issues.append(f"run directory already exists; refusing to overwrite: {target}")
    issues.extend(_protected_overlap_issues(target))

    if issues:
        raise OutputIsolationError(
            "Refusing to create the run directory. Isolation issues: "
            + "; ".join(issues)
        )
    target.mkdir(parents=True, exist_ok=False)
    return RunDirectory(
        run_id=run_id,
        path=target,
        results_dir=parent.resolve(),
        created_utc=utc_timestamp(),
        isolation_issues=tuple(issues),
        canonical_results_dir=results_dir is None,
    )


# ---------------------------------------------------------------------------
# Sampling audit machinery (categorical integrity of resampled rows)
# ---------------------------------------------------------------------------
class CapturingRecordingSampler(RecordingSampler):
    """Recording sampler that also retains the exact rows it produced.

    The retained matrix is only used in-process to decode and audit the
    categorical/binary values the sampler actually generated. It is never
    serialised.
    """

    def fit_resample(self, X: Any, y: Any):  # noqa: ANN202
        X_resampled, y_resampled = super().fit_resample(X, y)
        self.resampled_X_ = np.asarray(X_resampled, dtype=float)
        self.resampled_y_ = np.asarray(y_resampled).astype(int)
        return X_resampled, y_resampled


def build_audited_grid(grid: Mapping[str, Any]) -> dict[str, Any]:
    """Clone ``grid`` and make every sampler observable for the audit hook.

    Pipelines built by :func:`src.imbalance.build_imbalance_pipeline` use a
    plain sampler step by default (no recording); the recorder wrapper makes
    the wrapper kind uniform for the runner and the audit hook.
    """
    audited: dict[str, Any] = {}
    for key, pipeline in grid.items():
        named = pipeline.named_steps
        if "sampler" not in named:
            audited[key] = pipeline
            continue
        wrapper = named["sampler"]
        cloned = clone(pipeline)
        if isinstance(wrapper, RecordingSampler):
            cloned.set_params(
                sampler=CapturingRecordingSampler(clone(wrapper.sampler))
            )
        else:
            cloned.set_params(sampler=CapturingRecordingSampler(clone(wrapper)))
        audited[key] = cloned
    return audited


def _discrete_issues(frame: pd.DataFrame) -> list[str]:
    """Binary/category/numeric validity checks for a decoded resampled frame."""
    from src.imbalance_pipelines import UNSEEN_CATEGORY

    issues: list[str] = []
    for column in BINARY_FEATURES:
        observed = set(frame[column].unique().tolist())
        if not observed.issubset({0, 1}):
            issues.append(
                f"{column}: invalid binary values {sorted(map(str, observed))}"
            )
    for column, allowed in CATEGORY_VALUES.items():
        observed = {str(value) for value in frame[column].unique().tolist()}
        level_text = {str(level) for level in allowed}
        if not observed.issubset(level_text):
            issues.append(f"{column}: invalid categories {sorted(observed)}")
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
    """Audit the model preprocessor's one-hot block for fractional values."""
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
    fractional = (categorical_block > 1e-9) & (categorical_block < 1.0 - 1e-9)
    outside = (categorical_block < -1e-12) | (categorical_block > 1.0 + 1e-12)
    n_fractional = int(fractional.sum())
    if int(outside.sum()):
        issues.append(f"{int(outside.sum())} one-hot values outside [0, 1]")

    family = BASELINE_MODEL_SPECS[model_name].family
    for feature in CATEGORY_VALUES:
        columns = [
            index
            for index, name in enumerate(names)
            if name.startswith(f"cat__{feature}_")
        ]
        if not columns:
            issues.append(f"{feature}: no one-hot columns found")
            continue
        block = matrix[:, columns]
        row_sums = block.sum(axis=1)
        if family == "tree":
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
    spec: Any,
    pipeline: Any,
    sampler_step: Any,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> dict[str, Any]:
    """Decode and validate the exact rows this fold's sampler produced."""
    import traceback as traceback_module

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
        "sampler_random_state": getattr(sampler_step.sampler, "random_state", None),
        "n_fit_resample_calls": int(sampler_step.n_fit_resample_calls_),
        "n_train_before_sampling": int(sampler_step.n_samples_before_),
        "n_train_after_sampling": int(sampler_step.n_samples_after_),
        "class_counts_before": _counts_text(sampler_step.sampling_class_counts_before_),
        "class_counts_after": _counts_text(sampler_step.sampling_class_counts_after_),
        "n_appended_rows": int(
            max(0, sampler_step.n_samples_after_ - sampler_step.n_samples_before_)
        ),
        "n_decoded_rows": 0,
        "n_appended_rows_checked": 0,
        "n_fractional_one_hot_values": 0,
        "n_nonfinite_model_values": 0,
        "appended_rows_checked": False,
        "audit_status": "PASS",
        "issues": "",
        "traceback": "",
    }

    try:
        if not hasattr(sampler_step, "resampled_X_"):
            raise RuntimeError("sampler did not retain its resampled rows")

        decoder: CategoricalCodeDecoder = pipeline.named_steps["sampling_decoder"]
        resampled_X = np.asarray(sampler_step.resampled_X_, dtype=float)
        resampled_y = np.asarray(sampler_step.resampled_y_)
        if len(resampled_X) != len(resampled_y):
            raise RuntimeError("resampled X/y lengths differ")
        if len(resampled_y) != int(sampler_step.n_samples_after_):
            raise RuntimeError("resampled row count differs from recorded count")
        if not np.isfinite(resampled_X).all():
            raise RuntimeError("resampled encoded matrix has non-finite values")

        decoded = decoder.transform(resampled_X)
        if len(decoded) != len(resampled_y):
            raise RuntimeError("decoded row count differs from resampled labels")

        issues = _discrete_issues(decoded)
        one_hot_problems, n_fractional, n_nonfinite = _one_hot_issues(
            pipeline, decoded, model_name
        )
        issues.extend(one_hot_problems)

        n_before = int(sampler_step.n_samples_before_)
        appended = decoded.iloc[n_before:] if n_before < len(decoded) else decoded.iloc[0:0]
        appended_issues = _discrete_issues(appended) if len(appended) else []
        if appended_issues:
            issues.extend(f"appended rows: {issue}" for issue in appended_issues)

        record.update(
            {
                "n_decoded_rows": int(len(decoded)),
                "n_appended_rows_checked": int(len(appended)),
                "n_fractional_one_hot_values": int(n_fractional),
                "n_nonfinite_model_values": int(n_nonfinite),
                "appended_rows_checked": bool(len(appended) == int(
                    max(0, sampler_step.n_samples_after_ - sampler_step.n_samples_before_)
                )),
                "audit_status": "PASS" if not issues else "FAIL",
                "issues": " | ".join(issues),
            }
        )
    except Exception as exc:  # audit failures are reported, never hidden
        record.update(
            {
                "audit_status": "ERROR",
                "issues": f"{type(exc).__name__}: {exc}",
                "traceback": traceback_module.format_exc(),
            }
        )
    return record


# ---------------------------------------------------------------------------
# Pipeline description (model + sampling parameters for the manifest)
# ---------------------------------------------------------------------------
def _step_description(element: Any) -> dict[str, Any]:
    return {
        "class": type(element).__name__,
        "params": {
            str(name): _jsonable(value)
            for name, value in element.get_params(deep=False).items()
        },
    }


def describe_pipelines(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe every pipeline's classifier, sampler and steps for the manifest."""
    records: list[dict[str, Any]] = []
    for key, pipeline in grid.items():
        model_name, _, strategy_name = key.partition("__")
        record: dict[str, Any] = {
            "pipeline": key,
            "model": model_name,
            "strategy": strategy_name or "none",
            "preprocessor_family": BASELINE_MODEL_SPECS[model_name].family,
            "steps": list(pipeline.named_steps.keys()),
            "classifier": _step_description(pipeline.named_steps["classifier"]),
            "sampler": (
                _step_description(pipeline.named_steps["sampler"].sampler)
                if "sampler" in pipeline.named_steps
                and hasattr(pipeline.named_steps["sampler"], "sampler")
                else None
            ),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def build_fold_summaries(cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long- and wide-format fold-metric summaries (mean/SD/min/max)."""
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
            "sampler": (
                group["sampler"].dropna().iloc[0]
                if group["sampler"].notna().any()
                else "none"
            ),
            "mean_n_train_before": float(group["n_train_before_sampling"].mean()),
            "mean_n_train_after": float(group["n_train_after_sampling"].mean()),
            "n_features": int(group["n_features"].iloc[0]),
        }
        for metric in METRIC_NAMES:
            values = group[metric].astype(float)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            long_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy_name,
                    "pipeline": f"{model_name}__{strategy_name}",
                    "scope": "fold_level",
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "mean": float(values.mean()),
                    "sd": std,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_folds": int(len(values)),
                }
            )
            wide_row[metric] = float(values.mean())
            wide_row[f"{metric}_sd"] = std
        wide_rows.append(wide_row)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def build_pooled_within_repeat(oof: pd.DataFrame) -> pd.DataFrame:
    """Pool the OOF predictions *within* each repeat (never across repeats).

    For the 5x5 protocol this yields 25 pipelines x 5 repeats rows; the OOF
    counts of the five repeats overlap on the same rows, so cross-repeat
    pooling is statistically invalid and is not produced.
    """
    if oof.empty:
        return pd.DataFrame(
            columns=[
                "pipeline",
                "model",
                "strategy",
                "repeat",
                "scope",
                "average_precision_definition",
                "n_samples",
                "n_positive",
                *METRIC_NAMES,
                "threshold",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (pipeline, repeat), group in oof.groupby(["pipeline", "repeat"], sort=True):
        model_name, _, strategy_name = pipeline.partition("__")
        y_true = group["y_true"].to_numpy()
        y_proba = group["y_proba"].to_numpy(dtype=float)
        positive = int((y_true == 1).sum())
        rows.append(
            {
                "pipeline": pipeline,
                "model": model_name,
                "strategy": strategy_name or "none",
                "repeat": int(repeat),
                "scope": "pooled_within_repeat",
                "average_precision_definition": (
                    "average_precision_score (step-wise PR area, not "
                    "trapezoidal PR-AUC)"
                ),
                "n_samples": int(len(group)),
                "n_positive": positive,
                **compute_metrics(y_true, y_proba, threshold=DECISION_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verification: strict, non-tautological checks (G3)
# ---------------------------------------------------------------------------
def _check(
    checks: list[dict[str, Any]], name: str, passed: Any, details: Any
) -> None:
    checks.append({"check": name, "passed": bool(passed), "details": details})


def _unique_values(series: pd.Series) -> list[Any]:
    return sorted(set(series.tolist()), key=str)


def verify_imbalance_run(
    *,
    run_directory: RunDirectory,
    grid: Mapping[str, Any],
    fold_plan: Sequence[FoldSpec],
    plan_info: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: pd.DataFrame,
    oof: pd.DataFrame,
    sampling_audit: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    pipeline_status: Mapping[str, str],
    pipeline_parameters: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    written_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the strict verification battery for one experiment run.

    Every check reports concrete issues in its ``details`` block; nothing is
    tautological (contrast G3 of the readiness report, where two pilot
    checks always passed).
    """
    checks: list[dict[str, Any]] = []
    grid_keys = list(grid)
    resampling_keys = [
        key for key in grid_keys if key.partition("__")[2] in RESAMPLING_STRATEGIES
    ]
    svc_keys = [key for key in grid_keys if key.partition("__")[0] == "svc_rbf"]
    successful = {key for key, status in pipeline_status.items() if status == "ok"}
    expected_full_grid_keys = [
        f"{model}__{strategy}"
        for model in BASELINE_MODEL_SPECS
        for strategy in STRATEGIES
        if supports_strategy(model, strategy)
    ]

    # -- G1: output isolation ------------------------------------------------
    isolation_issues = list(run_directory.isolation_issues)
    isolation_issues.extend(_protected_overlap_issues(run_directory.path))
    if run_directory.canonical_results_dir and (
        run_directory.path.parent != RUN_DIRECTORY_PARENT.resolve()
    ):
        isolation_issues.append(
            f"run directory {run_directory.path} is not directly under "
            f"{RUN_DIRECTORY_PARENT}"
        )
    for artifact_key, path in paths.items():
        try:
            run_directory.check_path(path)
        except OutputIsolationError as exc:
            isolation_issues.append(f"{artifact_key}: {exc}")
    _check(
        checks,
        "run_directory_isolated_and_new",
        not isolation_issues,
        {
            "run_directory": _relative(run_directory.path),
            "issues": isolation_issues,
        },
    )

    # -- grid validity -------------------------------------------------------
    grid_issues: list[str] = []
    for key in grid_keys:
        model_name, _, strategy_name = key.partition("__")
        if not _:
            grid_issues.append(f"{key}: pipeline keys must be 'model__strategy'")
            continue
        if model_name not in BASELINE_MODEL_SPECS:
            grid_issues.append(f"{key}: unknown model {model_name!r}")
        if strategy_name not in STRATEGIES:
            grid_issues.append(f"{key}: unknown strategy {strategy_name!r}")
        elif not supports_strategy(model_name, strategy_name):
            grid_issues.append(
                f"{key}: strategy {strategy_name!r} is not supported for "
                f"{model_name!r}"
            )
    if bool(config.get("expect_full_grid")) and grid_keys != expected_full_grid_keys:
        grid_issues.append(
            f"full-grid mode expected {expected_full_grid_keys}, found {grid_keys}"
        )
    _check(
        checks,
        "supported_pipelines_valid",
        not grid_issues,
        {"issues": grid_issues, "n_pipelines": len(grid_keys)},
    )

    # -- attempt completeness ------------------------------------------------
    _check(
        checks,
        "all_pipelines_attempted",
        set(pipeline_status) == set(grid_keys),
        {
            "attempted": sorted(pipeline_status),
            "missing": sorted(set(grid_keys) - set(pipeline_status)),
            "unexpected": sorted(set(pipeline_status) - set(grid_keys)),
        },
    )

    # -- fold-plan sharing and validity --------------------------------------
    fold_issues: list[str] = []
    if plan_info["n_rows"] != len(X):
        fold_issues.append(
            f"fold plan n_rows {plan_info['n_rows']} != len(X) {len(X)}"
        )
    if plan_info["n_splits"] != int(config["n_splits"]):
        fold_issues.append(
            f"fold plan n_splits {plan_info['n_splits']} != configured "
            f"{config['n_splits']}"
        )
    if plan_info["n_repeats"] != int(config["n_repeats"]):
        fold_issues.append(
            f"fold plan n_repeats {plan_info['n_repeats']} != configured "
            f"{config['n_repeats']}"
        )
    expected_pairs = {
        (int(spec.repeat), int(spec.fold)) for spec in fold_plan
    }
    if not cv.empty:
        for (repeat, fold), group in cv.groupby(["repeat", "fold"]):
            if len(group["train_index_fingerprint"].unique()) != 1:
                fold_issues.append(
                    f"repeat {repeat} fold {fold}: train fingerprints differ "
                    "across pipelines"
                )
            if len(group["validation_index_fingerprint"].unique()) != 1:
                fold_issues.append(
                    f"repeat {repeat} fold {fold}: validation fingerprints "
                    "differ across pipelines"
                )
        for pipeline_key in cv["pipeline"].unique():
            pairs = list(
                zip(
                    cv.loc[cv["pipeline"] == pipeline_key, "repeat"],
                    cv.loc[cv["pipeline"] == pipeline_key, "fold"],
                )
            )
            if len(pairs) != len(set(pairs)):
                fold_issues.append(f"{pipeline_key}: duplicated (repeat, fold) rows")
            if set(pairs) != expected_pairs:
                fold_issues.append(
                    f"{pipeline_key}: fold coverage differs from the shared plan"
                )
        # FingerprintMatch: cv fingerprints must equal plan-computed ones.
        folds_by_pair = {
            (int(spec.repeat), int(spec.fold)): spec for spec in fold_plan
        }
        mismatches = 0
        for row in cv.itertuples():
            spec = folds_by_pair.get((int(row.repeat), int(row.fold)))
            if spec is None:
                mismatches += 1
                continue
            train_fp = _index_fingerprint(X.index.values[spec.train_positions])
            valid_fp = _index_fingerprint(X.index.values[spec.valid_positions])
            if (
                row.train_index_fingerprint != train_fp
                or row.validation_index_fingerprint != valid_fp
            ):
                mismatches += 1
        if mismatches:
            fold_issues.append(
                f"{mismatches} fold rows have fingerprints differing from the plan"
            )
    _check(
        checks,
        "fold_plan_shared_and_valid",
        not fold_issues,
        {
            "issues": fold_issues,
            "n_folds_total": plan_info["n_folds_total"],
            "n_splits": plan_info["n_splits"],
            "n_repeats": plan_info["n_repeats"],
        },
    )

    # -- sampling counts against the fold plan -------------------------------
    plan_by_pair = {(int(spec.repeat), int(spec.fold)): spec for spec in fold_plan}
    count_issues: list[str] = []
    for row in cv.itertuples():
        spec = plan_by_pair.get((int(row.repeat), int(row.fold)))
        if spec is None:
            count_issues.append(
                f"{row.pipeline} repeat {row.repeat} fold {row.fold}: "
                "fold is absent from the plan"
            )
            continue
        label = f"{row.pipeline} r{row.repeat}f{row.fold}"
        if int(row.n_train_before_sampling) != len(spec.train_positions):
            count_issues.append(f"{label}: n_train_before differs from the plan")
        if int(row.n_validation) != len(spec.valid_positions):
            count_issues.append(f"{label}: n_validation differs from the plan")
        if _counts_text(row.sampling_class_counts_before) != _counts_text(
            class_counts(y.iloc[spec.train_positions])
        ):
            count_issues.append(f"{label}: sampled 'before' counts are wrong")
        if _counts_text(row.validation_class_counts_before) != _counts_text(
            class_counts(y.iloc[spec.valid_positions])
        ):
            count_issues.append(f"{label}: validation counts are wrong")
        if row.validation_class_counts_before != row.validation_class_counts_after:
            count_issues.append(f"{label}: validation labels were altered")
        if bool(row.sampling_applied):
            if int(row.n_fit_resample_calls) != 1:
                count_issues.append(
                    f"{label}: {row.n_fit_resample_calls} fit_resample calls"
                )
            direction_issue = {
                "random_over_sampler": row.n_train_after_sampling
                <= row.n_train_before_sampling,
                "smote": row.n_train_after_sampling <= row.n_train_before_sampling,
                "adasyn": row.n_train_after_sampling <= row.n_train_before_sampling,
                "random_under_sampler": row.n_train_after_sampling
                >= row.n_train_before_sampling,
            }.get(row.strategy)
            if direction_issue:
                count_issues.append(
                    f"{label}: resampling direction is wrong for {row.strategy}"
                )
        elif int(row.n_fit_resample_calls) != 0:
            count_issues.append(f"{label}: sampler invoked without resampling")
        if int(row.n_train_after_sampling) != int(
            sum(row.sampling_class_counts_after.values())
        ):
            count_issues.append(f"{label}: after-counts disagree with row counts")
    _check(
        checks,
        "sampling_counts_match_fold_plan",
        not count_issues,
        {"issues": count_issues, "n_fold_rows": int(len(cv))},
    )

    # -- fit success ---------------------------------------------------------
    success_issues: list[str] = []
    if not errors_table.empty:
        success_issues.append("error rows exist; the run did not fit cleanly")
    failed_pipelines = {
        key for key, status in pipeline_status.items() if status != "ok"
    }
    if failed_pipelines:
        success_issues.append(
            f"failed pipelines: {sorted(failed_pipelines)}"
        )
    fold_total = len(fold_plan)
    if not cv.empty:
        per_pipeline = cv["pipeline"].value_counts()
        for key in grid_keys:
            if key not in per_pipeline.index:
                continue  # covered by "failed pipelines"
            if int(per_pipeline[key]) != fold_total:
                success_issues.append(
                    f"{key}: {per_pipeline[key]} fold rows, expected {fold_total}"
                )
    if len(cv) != len(grid_keys) * fold_total:
        success_issues.append(
            f"fold record count {len(cv)} != {len(grid_keys)} x {fold_total}"
        )
    _check(
        checks,
        "all_pipelines_fit_successfully",
        not success_issues,
        {
            "issues": success_issues,
            "n_successful": len(successful),
            "n_failed": len(failed_pipelines),
            "n_fold_records": int(len(cv)),
            "expected_fold_records": len(grid_keys) * fold_total,
        },
    )

    # -- every pipeline/fold combination was attempted (G3) -------------------
    attempt_issues: list[str] = []
    plan_pairs = {(int(spec.repeat), int(spec.fold)) for spec in fold_plan}
    succeeded_pairs: dict[str, set[tuple[int, int]]] = {}
    if not cv.empty:
        succeeded_pairs = {
            pipeline_key: {
                (int(repeat), int(fold))
                for repeat, fold in zip(group["repeat"], group["fold"])
            }
            for pipeline_key, group in cv.groupby("pipeline")
        }
    error_pairs: dict[str, set[tuple[int, int]]] = {}
    if not errors_table.empty:
        for row in errors_table.itertuples():
            try:
                pair = (int(row.repeat), int(row.fold))
            except (TypeError, ValueError):
                attempt_issues.append(
                    f"{row.pipeline}: error row lacks repeat/fold attribution"
                )
                continue
            if pair not in plan_pairs:
                attempt_issues.append(
                    f"{row.pipeline}: error pair {pair} is outside the plan"
                )
            error_pairs.setdefault(row.pipeline, set()).add(pair)
    for pipeline_key in grid_keys:
        n_ok = len(succeeded_pairs.get(pipeline_key, set()))
        n_err = len(error_pairs.get(pipeline_key, set()))
        if n_ok + n_err != fold_total:
            attempt_issues.append(
                f"{pipeline_key}: {n_ok} successful + {n_err} reported "
                f"failed folds != {fold_total} planned folds"
            )
        overlap = succeeded_pairs.get(pipeline_key, set()) & error_pairs.get(
            pipeline_key, set()
        )
        if overlap:
            attempt_issues.append(
                f"{pipeline_key}: folds {sorted(overlap)} both succeeded "
                "and errored"
            )
    _check(
        checks,
        "all_folds_attempted",
        not attempt_issues,
        {
            "issues": attempt_issues,
            "n_folds_total": fold_total,
            "n_pipelines": len(grid_keys),
            "expected_fold_attempts": len(grid_keys) * fold_total,
            "n_successful_fold_rows": int(len(cv)),
            "n_error_rows": int(len(errors_table)),
        },
    )

    # -- error reporting consistency (G3) ------------------------------------
    error_issues: list[str] = []
    error_pipelines = (
        set(errors_table["pipeline"].tolist()) if not errors_table.empty else set()
    )
    if error_pipelines != failed_pipelines:
        error_issues.append(
            f"error rows cover {sorted(error_pipelines)} but failed pipelines "
            f"are {sorted(failed_pipelines)}"
        )
    if not errors_table.empty:
        for row in errors_table.itertuples():
            for field_name in ("error_type", "message", "traceback"):
                if not str(getattr(row, field_name, "") or "").strip():
                    error_issues.append(
                        f"{row.pipeline} r{row.repeat}f{row.fold}: empty "
                        f"{field_name} in errors.csv"
                    )
            success_issue = row.pipeline in successful
            if success_issue:
                error_issues.append(
                    f"{row.pipeline}: error row exists but pipeline is 'ok'"
                )
    _check(
        checks,
        "errors_reported_consistently",
        not error_issues,
        {
            "issues": error_issues,
            "n_error_rows": int(len(errors_table)),
            "failed_pipelines": sorted(failed_pipelines),
        },
    )

    # -- strict warning classification (G3) ----------------------------------
    warning_issues: list[str] = []
    if warnings_table.empty:
        svc_fold_rows = (
            0 if cv.empty else int(cv["pipeline"].isin(svc_keys).sum())
        )
        if svc_fold_rows:
            warning_issues.append(
                f"{svc_fold_rows} SVC fold fits produced zero captured warnings"
                " (suppression or capture failure)"
            )
    else:
        for record in warnings_table.to_dict("records"):
            for field_name in ("category", "message", "filename"):
                if not str(record.get(field_name, "") or "").strip():
                    warning_issues.append(
                        f"{record.get('pipeline', '?')}: empty {field_name}"
                    )
            status, reason = classify_warning(record)
            if status == "UNEXPECTED":
                warning_issues.append(
                    f"{record.get('pipeline', '?')} "
                    f"[{record.get('category', '?')}]: {reason}"
                )
    _check(
        checks,
        "warnings_expected_categories_only",
        not warning_issues,
        {
            "issues": warning_issues[:12],
            "n_warning_rows": int(len(warnings_table)),
            "categories": (
                _unique_values(warnings_table["category"])
                if not warnings_table.empty
                else []
            ),
            "allow_list": sorted(EXPECTED_WARNING_CATEGORIES),
        },
    )

    # -- expected SVC FutureWarning count ------------------------------------
    svc_fold_fits = 0 if cv.empty else int(cv["pipeline"].isin(svc_keys).sum())
    captured_svc_warnings = (
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
    run_is_clean = (
        errors_table.empty
        and all(status == "ok" for status in pipeline_status.values())
        and len(cv) == len(grid_keys) * fold_total
    )
    if run_is_clean:
        warning_count_ok = captured_svc_warnings == svc_fold_fits
        warning_count_note = (
            "clean run: captured SVC FutureWarnings must equal SVC fold fits"
        )
    else:
        warning_count_ok = False
        warning_count_note = (
            "run is not clean (failed folds/errors); the expected warning "
            "count cannot be pinned, so the check fails with the rest"
        )
    _check(
        checks,
        "warning_counts_match_fit_events",
        warning_count_ok,
        {
            "note": warning_count_note,
            "captured_svc_probability_warnings": captured_svc_warnings,
            "svc_fold_fits": svc_fold_fits,
        },
    )

    # -- probability validity ------------------------------------------------
    probability_issues: list[str] = []

    if oof.empty:
        if successful:
            probability_issues.append(
                "no out-of-fold predictions despite successful pipelines"
            )
    else:
        unknown_pipelines = set(oof["pipeline"]) - set(grid_keys)
        if unknown_pipelines:
            probability_issues.append(
                f"unknown pipelines in OOF predictions: {sorted(unknown_pipelines)}"
            )

        leaked = set(oof["pipeline"]) - successful
        if leaked:
            probability_issues.append(
                f"failed pipelines produced OOF predictions: {sorted(leaked)}"
            )

        for pipeline_key, group in oof.groupby("pipeline"):
            if pipeline_key not in successful:
                continue

            values = group["y_proba"].to_numpy(dtype=float)
            label = pipeline_key

            # Each observation receives one OOF prediction per CV repeat.
            expected_rows = len(X) * int(config["n_repeats"])

            if len(group) != expected_rows:
                probability_issues.append(
                    f"{label}: {len(group)} OOF rows, expected {expected_rows}"
                )

            # Check probabilities are finite and within [0, 1].
            if not np.isfinite(values).all():
                probability_issues.append(
                    f"{label}: non-finite probabilities"
                )

            if bool(((values < 0.0) | (values > 1.0)).any()):
                probability_issues.append(
                    f"{label}: probabilities outside [0, 1]"
                )

            # Every repeat must cover the complete training partition.
            expected_repeats = set(
                range(1, int(config["n_repeats"]) + 1)
            )
            actual_repeats = set(group["repeat"].astype(int))

            if actual_repeats != expected_repeats:
                probability_issues.append(
                    f"{label}: incorrect CV repeat coverage "
                    f"(expected {sorted(expected_repeats)}, "
                    f"found {sorted(actual_repeats)})"
                )

            # Duplicate indices are valid across repeats,
            # but never within the same repeat.
            if group.duplicated(
                subset=["repeat", "row_index"]
            ).any():
                probability_issues.append(
                    f"{label}: duplicate OOF predictions within a repeat"
                )

            for repeat, repeat_group in group.groupby("repeat"):
                if len(repeat_group) != len(X):
                    probability_issues.append(
                        f"{label}: repeat {repeat} has "
                        f"{len(repeat_group)} OOF rows, expected {len(X)}"
                    )

                if set(repeat_group["row_index"]) != set(X.index):
                    probability_issues.append(
                        f"{label}: repeat {repeat} does not cover "
                        "all training rows exactly once"
                    )

            # Verify that every prediction has the correct true label.
            expected_true = y.loc[group["row_index"]].to_numpy()

            if not bool(
                (group["y_true"].to_numpy() == expected_true).all()
            ):
                probability_issues.append(
                    f"{label}: y_true does not match y"
                )

    _check(
        checks,
        "probabilities_valid",
        not probability_issues,
        {
            "issues": probability_issues[:12],
            "n_oof_rows": int(len(oof)),
            "expected_oof_rows": (
                len(X)
                * int(config["n_repeats"])
                * len(successful)
            ),
        },
    )
    # -- resampling audit legality -------------------------------------------
    audit_issues: list[str] = []
    resampling_rows = (
        0 if cv.empty else int(cv["pipeline"].isin(resampling_keys).sum())
    )
    if resampling_keys:
        if resampling_rows == 0:
            audit_issues.append("resampling strategies present but no cv rows")
        elif sampling_audit.empty:
            audit_issues.append("no sampling audits were recorded")
        else:
            if set(sampling_audit["audit_status"].unique()) - {"PASS"}:
                audit_issues.append(
                    "some audits are not PASS: "
                    + str(_unique_values(sampling_audit["audit_status"]))
                )
            if bool((sampling_audit["n_fractional_one_hot_values"] > 0).any()):
                audit_issues.append("fractional one-hot values in sampled rows")
            if bool((~sampling_audit["appended_rows_checked"].astype(bool)).any()):
                audit_issues.append("some appended rows were not checked")
            audit_pairs = list(
                zip(
                    sampling_audit["pipeline"],
                    sampling_audit["repeat"],
                    sampling_audit["fold"],
                )
            )
            if len(audit_pairs) != len(set(audit_pairs)):
                audit_issues.append("duplicate (pipeline, repeat, fold) audits")
            cv_pairs = {
                (row.pipeline, int(row.repeat), int(row.fold))
                for row in cv.itertuples()
                if row.pipeline in resampling_keys
            }
            if set(audit_pairs) != cv_pairs:
                audit_issues.append(
                    "audit coverage differs from resampling fold fits "
                    f"(audits: {len(audit_pairs)}, fits: {len(cv_pairs)})"
                )
            cv_by_key = {
                (row.pipeline, int(row.repeat), int(row.fold)): row
                for row in cv.itertuples()
            }
            for audit_row in sampling_audit.itertuples():
                cv_row = cv_by_key.get(
                    (audit_row.pipeline, int(audit_row.repeat), int(audit_row.fold))
                )
                if cv_row is None:
                    continue
                if int(audit_row.n_train_after_sampling) != int(
                    cv_row.n_train_after_sampling
                ):
                    audit_issues.append(
                        f"{audit_row.pipeline} r{audit_row.repeat}"
                        f"f{audit_row.fold}: audit and cv row counts disagree"
                    )
            for (strategy, repeat, fold), group in sampling_audit.groupby(
                ["strategy", "repeat", "fold"]
            ):
                if group["n_train_after_sampling"].nunique() != 1:
                    audit_issues.append(
                        f"{strategy} r{repeat}f{fold}: models received "
                        "different resampled row counts"
                    )
                if group["class_counts_after"].nunique() != 1:
                    audit_issues.append(
                        f"{strategy} r{repeat}f{fold}: models received "
                        "different resampled class counts"
                    )
    _check(
        checks,
        "resampling_audited_legally",
        not audit_issues,
        {
            "issues": audit_issues[:12],
            "n_resampling_fold_rows": resampling_rows,
            "n_audits": int(len(sampling_audit)),
        },
    )

    # -- manifest completeness (G2) ------------------------------------------
    manifest_issues: list[str] = []
    environment = manifest.get("environment", {})
    for package_key in (
        "python",
        "platform",
        "numpy",
        "scipy",
        "pandas",
        "scikit_learn",
        "joblib",
        "threadpoolctl",
        "imbalanced_learn",
    ):
        value = environment.get(package_key)
        if not isinstance(value, str) or not value:
            manifest_issues.append(f"environment.{package_key} missing")
    if not environment.get("git_commit"):
        manifest_issues.append("environment.git_commit missing")
    if not environment.get("git_branch"):
        manifest_issues.append("environment.git_branch missing")
    if not isinstance(environment.get("git_dirty"), bool):
        manifest_issues.append("environment.git_dirty missing")
    if not environment.get("pip_freeze"):
        manifest_issues.append("environment.pip_freeze missing")
    source_hashes = manifest.get("source_files", {})
    if len(source_hashes) < len(RUNNER_SOURCE_FILES):
        manifest_issues.append(
            f"source_files has {len(source_hashes)} entries, expected "
            f">= {len(RUNNER_SOURCE_FILES)}"
        )
    for relative, digest in source_hashes.items():
        if not HEX64_PATTERN.match(str(digest)):
            manifest_issues.append(f"source_files[{relative!r}] is not a sha256")
    start_hashes = manifest.get("source_files_start", {})
    if set(start_hashes) != set(source_hashes):
        manifest_issues.append(
            "source_files_start covers different files than source_files"
        )
    for relative, digest in start_hashes.items():
        if not HEX64_PATTERN.match(str(digest)):
            manifest_issues.append(
                f"source_files_start[{relative!r}] is not a sha256"
            )
    configuration_hashes = manifest.get("configuration_hashes", {})
    for relative in CONFIGURATION_FILES:
        digest = configuration_hashes.get(relative)
        if not HEX64_PATTERN.match(str(digest or "")):
            manifest_issues.append(
                f"configuration_hashes[{relative!r}] missing or not a sha256"
            )
    fingerprints = manifest.get("fold_plan", {}).get(
        "repeat_assignment_fingerprints", {}
    )
    if not fingerprints:
        manifest_issues.append("fold_plan.repeat_assignment_fingerprints missing")
    for repeat_key, fingerprint in fingerprints.items():
        if not re.match(r"^[0-9a-f]{16}$", str(fingerprint)):
            manifest_issues.append(
                f"repeat {repeat_key}: fingerprint is not a 16-hex value"
            )
    if "pipeline_fold_counts" not in manifest:
        manifest_issues.append("pipeline_fold_counts missing")
    data_block = manifest.get("data", {})
    if not HEX64_PATTERN.match(str(data_block.get("data_sha256", ""))):
        manifest_issues.append("data.data_sha256 missing or not a sha256")
    for count_key in ("n_rows", "n_train"):
        if int(data_block.get(count_key, 0) or 0) <= 0:
            manifest_issues.append(f"data.{count_key} missing")
    sampling_block = manifest.get("sampling", {})
    for sampling_key in ("sampling_strategy_ratio", "k_neighbors", "sampler_seed"):
        if sampling_block.get(sampling_key) is None:
            manifest_issues.append(f"sampling.{sampling_key} missing")
    cv_block = manifest.get("cv", {})
    for cv_key in ("n_splits", "n_repeats", "cv_random_state", "decision_threshold"):
        if cv_block.get(cv_key) is None:
            manifest_issues.append(f"cv.{cv_key} missing")
    described = {record["pipeline"]: record for record in pipeline_parameters}
    for key in grid_keys:
        record = described.get(key)
        if record is None:
            manifest_issues.append(f"model parameters missing for {key}")
            continue
        if not record.get("classifier", {}).get("params"):
            manifest_issues.append(f"{key}: no classifier parameters recorded")
        if record.get("preprocessor_family") is None:
            manifest_issues.append(f"{key}: no preprocessor family recorded")
    pin = manifest.get("repeatability_pin", {})
    if not HEX64_PATTERN.match(str(pin.get("sha256", ""))):
        manifest_issues.append("repeatability_pin.sha256 missing")
    _check(
        checks,
        "manifest_completeness",
        not manifest_issues,
        {"issues": manifest_issues},
    )

    # -- source freeze across the run (G2) ------------------------------------
    freeze_issues: list[str] = []
    start_hashes = manifest.get("source_files_start", {})
    end_hashes = manifest.get("source_files", {})
    changed = diff_source_hashes(start_hashes, end_hashes)
    if changed:
        freeze_issues.append(
            f"source files changed during the run: {sorted(changed)}"
        )
    _check(
        checks,
        "source_freeze_stable",
        not freeze_issues,
        {
            "issues": freeze_issues,
            "changed_files": changed,
            "note": (
                "Hashes captured immediately after run-directory creation and "
                "again before manifest serialization."
            ),
        },
    )

    # -- repeatability pin vs runtime ----------------------------------------
    pin_issues: list[str] = []
    pin = manifest.get("repeatability_pin", {})
    comparisons = {
        "n_splits": config["n_splits"],
        "n_repeats": config["n_repeats"],
        "cv_random_state": config["random_state"],
        "decision_threshold": float(config["threshold"]),
        "sampling_strategy_ratio": float(config["sampling_ratio"]),
        "k_neighbors": int(config["k_neighbors"]),
        "sampler_seed": int(config["sampler_seed"]),
    }
    for pin_key, runtime_value in comparisons.items():
        pin_value = pin.get(pin_key)
        if isinstance(runtime_value, (int, float)):
            pin_value = (
                type(runtime_value)(pin_value)
                if pin_value is not None
                else None
            )
        if pin_value != runtime_value:
            pin_issues.append(
                f"pin.{pin_key}={pin_value!r} != runtime {runtime_value!r}"
            )
    if list(pin.get("grid_keys", [])) != grid_keys:
        pin_issues.append("pin.grid_keys differs from the executed grid")
    observed_cv_seeds = (
        _unique_values(cv["cv_random_state"]) if not cv.empty else []
    )
    if observed_cv_seeds != [int(config["random_state"])]:
        pin_issues.append(
            f"cv rows used cv seeds {observed_cv_seeds}, expected "
            f"[{config['random_state']}]"
        )
    observed_sampler_seeds = (
        [
            value
            for value in _unique_values(cv["sampler_random_state"].dropna())
            if value is not None
        ]
        if not cv.empty
        else []
    )
    if bool(cv["sampler_random_state"].notna().any() if not cv.empty else False):
        if [int(v) for v in observed_sampler_seeds] != [
            int(config["sampler_seed"])
        ]:
            pin_issues.append(
                f"cv rows used sampler seeds {observed_sampler_seeds}, expected "
                f"[{config['sampler_seed']}]"
            )
    for record in pipeline_parameters:
        sampler = record.get("sampler")
        if not sampler:
            continue
        params = sampler.get("params", {})
        if params.get("random_state") != int(config["sampler_seed"]):
            pin_issues.append(
                f"{record['pipeline']}: sampler random_state {params.get('random_state')!r} "
                f"!= {int(config['sampler_seed'])}"
            )
        if "sampling_strategy" in params and float(
            params["sampling_strategy"]
        ) != float(config["sampling_ratio"]):
            pin_issues.append(
                f"{record['pipeline']}: sampling_strategy {params.get('sampling_strategy')!r} "
                f"!= {float(config['sampling_ratio'])}"
            )
        neighbor_value = params.get("k_neighbors", params.get("n_neighbors"))
        if record["strategy"] in ("smote", "adasyn") and neighbor_value is not None:
            if int(neighbor_value) != int(config["k_neighbors"]):
                pin_issues.append(
                    f"{record['pipeline']}: neighbour count {neighbor_value!r} "
                    f"!= {int(config['k_neighbors'])}"
                )
    _check(
        checks,
        "repeatability_pin_matches_runtime",
        not pin_issues,
        {"issues": pin_issues},
    )

    # -- persisted artifacts -------------------------------------------------
    artifact_issues: list[str] = []
    for artifact_key, path in written_paths.items():
        if not Path(path).is_file():
            artifact_issues.append(f"{artifact_key}: file was not written")
        elif Path(path).stat().st_size == 0 and artifact_key not in (
            "warnings",
            "errors",
            "sampling_audit",
        ):
            artifact_issues.append(f"{artifact_key}: file is empty")
    _check(
        checks,
        "artifacts_persisted",
        not artifact_issues,
        {"issues": artifact_issues, "n_files": len(written_paths)},
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
            "Training-partition cross-validation only. No threshold tuning, "
            "no calibration and no held-out test evaluation."
        ),
    }


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------
def _repeatability_pin(
    *,
    step: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    threshold: float,
    sampling_ratio: float,
    k_neighbors: int,
    sampler_seed: int,
    grid_keys: Sequence[str],
    n_rows: int,
) -> dict[str, Any]:
    core = {
        "step": step,
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "cv_random_state": int(random_state),
        "decision_threshold": float(threshold),
        "sampling_strategy_ratio": float(sampling_ratio),
        "k_neighbors": int(k_neighbors),
        "sampler_seed": int(sampler_seed),
        "grid_keys": list(grid_keys),
        "n_rows": int(n_rows),
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    pin = dict(core)
    pin["sha256"] = digest
    pin["note"] = (
        "Frozen pre-run configuration; sha256 over the sorted core mapping."
    )
    return pin


def _draft_manifest(
    *,
    run_directory: RunDirectory,
    step: str,
    data_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    plan_info: Mapping[str, Any],
    pipeline_parameters: Sequence[Mapping[str, Any]],
    pipeline_status: Mapping[str, str],
    pipeline_seconds: Mapping[str, float],
    pipeline_fold_counts: Mapping[str, Mapping[str, int]],
    source_hashes_start: Mapping[str, str],
    pin: Mapping[str, Any],
    grid_keys: Sequence[str],
) -> dict[str, Any]:
    return {
        "run_id": run_directory.run_id,
        "step": step,
        "created_utc": run_directory.created_utc,
        "scope": (
            "Run-scoped imbalance cross-validation on the training "
            "partition only. No threshold tuning, no calibration and no "
            "held-out test evaluation."
        ),
        "random_state": int(config["random_state"]),
        "decision_threshold": float(config["threshold"]),
        "metric_definitions": METRIC_LABELS,
        "average_precision_note": (
            "average_precision_score computes a step-wise estimate of the "
            "area under the precision-recall curve. It is not trapezoidal "
            "PR-AUC."
        ),
        "environment": core_environment(),
        "source_files": source_file_hashes(),
        "source_files_start": dict(source_hashes_start),
        "source_files_end": source_file_hashes(),
        "configuration_hashes": configuration_file_hashes(),
        "source_freeze_note": (
            "source_files_start was captured immediately after run-directory "
            "creation; source_files_end immediately before manifest "
            "serialization. Differences are flagged by the "
            "source_freeze_stable verification check."
        ),
        "data": dict(data_metadata),
        "heldout_test": {
            "locked": bool(data_metadata.get("heldout_test_was_locked")),
            "data_fields_accessed": bool(
                data_metadata.get("heldout_test_data_fields_accessed")
            ),
            "note": "Imbalance runs never fit, predict, tune or evaluate on the test partition.",
        },
        "fold_plan": dict(plan_info),
        "cv": {
            "n_splits": int(config["n_splits"]),
            "n_repeats": int(config["n_repeats"]),
            "cv_random_state": int(config["random_state"]),
            "decision_threshold": float(config["threshold"]),
            "n_folds_total": int(plan_info["n_folds_total"]),
            "validation_resampling": False,
            "test_set_used": False,
        },
        "sampling": {
            "sampling_strategy_ratio": float(config["sampling_ratio"]),
            "k_neighbors": int(config["k_neighbors"]),
            "sampler_seed": int(config["sampler_seed"]),
            "resampling_strategies": list(RESAMPLING_STRATEGIES),
            "note": (
                "SMOTE is implemented with SMOTENC; ADASYN is the "
                "categorical-aware project variant (CategoricalADASYN)."
            ),
        },
        "pipelines": list(pipeline_parameters),
        "pipeline_status": dict(pipeline_status),
        "pipeline_fold_counts": {
            key: dict(counts) for key, counts in pipeline_fold_counts.items()
        },
        "pipeline_seconds": {
            key: round(float(value), 4)
            for key, value in pipeline_seconds.items()
        },
        "repeatability_pin": dict(pin),
        "supported_grid_keys": list(grid_keys),
        "outcomes": {},
        "total_seconds": None,
        "verification": {},
        "files": {},
    }


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render a small GitHub-flavoured Markdown table without tabulate."""
    existing = [column for column in columns if column in frame.columns]
    if frame.empty or not existing:
        return "_none_\n"
    view = frame[existing]
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    separator = "|" + "|".join("---" for _ in view.columns) + "|"
    rows = [
        "| "
        + " | ".join(
            _markdown_cell(view.iloc[row_index][column])
            for column in view.columns
        )
        + " |"
        for row_index in range(len(view))
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as CSV; empty frames result in a 0-byte CSV file."""
    if frame.empty:
        Path(path).write_text("", encoding="utf-8")
    else:
        frame.to_csv(path, index=False)


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}" if np.isfinite(value) else str(value)
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _write_report(
    path: Path,
    *,
    run_directory: RunDirectory,
    data_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    verification: Mapping[str, Any],
    summary_wide: pd.DataFrame,
    pooled_by_repeat: pd.DataFrame,
    sampling_audit: pd.DataFrame,
    warnings_table: pd.DataFrame,
    errors_table: pd.DataFrame,
    pipeline_seconds: Mapping[str, float],
    artifact_paths: Mapping[str, str],
) -> None:
    """Persist the human-readable run report."""
    lines: list[str] = []
    lines.append("# Imbalance experiment report\n")
    lines.append(f"* **run_id:** `{run_directory.run_id}`\n")
    lines.append(f"* **created (UTC):** {run_directory.created_utc}\n")
    lines.append(
        "* **scope:** training-partition cross-validation only; no threshold "
        "tuning, no calibration, no held-out test evaluation\n"
    )
    lines.append("\n## Setup\n")
    lines.append(
        f"* data: {data_metadata.get('n_train', '?')} training rows, "
        f"{data_metadata.get('train_class_counts', {})} — prevalence "
        f"{data_metadata.get('train_prevalence', float('nan')):.4f}\n"
    )
    lines.append(
        f"* data SHA-256: `{data_metadata.get('data_sha256', '?')}`\n"
    )
    lines.append(
        f"* CV: {config['n_splits']} folds x {config['n_repeats']} repeats, "
        f"shared fold plan, cv seed {config['random_state']}\n"
    )
    lines.append(
        f"* threshold {config['threshold']}; sampling ratio "
        f"{config['sampling_ratio']}; k_neighbors {config['k_neighbors']}; "
        f"sampler seed {config['sampler_seed']}\n"
    )
    lines.append(
        f"* pipelines: {len(pipeline_seconds)}; wall time per pipeline is in "
        "the manifest\n"
    )
    lines.append(f"\n## Verification: "
                 f"{'PASSED' if verification['overall_passed'] else 'FAILED'} "
                 f"({verification['n_passed']}/{verification['n_checks']} checks)\n")
    check_rows = pd.DataFrame(
        [
            {
                "check": check["check"],
                "result": "PASS" if check["passed"] else "FAIL",
                "details": json.dumps(check["details"], default=str)[:220],
            }
            for check in verification["checks"]
        ]
    )
    lines.append(_markdown_table(check_rows, ["check", "result", "details"]))
    lines.append("\n## Fold-level mean metrics\n")
    lines.append(
        _markdown_table(
            summary_wide,
            [
                "pipeline",
                "n_folds",
                "sampler",
                "mean_n_train_before",
                "mean_n_train_after",
                *METRIC_NAMES,
            ],
        )
    )
    if not pooled_by_repeat.empty:
        per_pipeline = (
            pooled_by_repeat.groupby(["model", "strategy"], as_index=False)
            .agg(
                average_precision=("average_precision", "mean"),
                roc_auc=("roc_auc", "mean"),
            )
            .sort_values("average_precision", ascending=False)
        )
        lines.append(
            "\n## Mean pooled-within-repeat metrics (across repeats)\n"
        )
        lines.append(
            _markdown_table(
                per_pipeline,
                ["model", "strategy", "average_precision", "roc_auc"],
            )
        )
        lines.append(
            "\n_note: pooled metrics aggregate the five validation folds of "
            "**one repeat**; repeats overlap on rows, so cross-repeat pooling "
            "is not performed._\n"
        )
    lines.append("\n## Warnings\n")
    if warnings_table.empty:
        lines.append("_none captured_\n")
    else:
        by_category = (
            warnings_table.groupby("category", as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        lines.append(_markdown_table(by_category, ["category", "n"]))
        first_messages = warnings_table.drop_duplicates(subset=["message"]).head(3)
        for message in first_messages["message"].tolist():
            lines.append(f"> {message}\n")
    lines.append("\n## Errors\n")
    if errors_table.empty:
        lines.append("_none_\n")
    else:
        lines.append(
            _markdown_table(
                errors_table,
                ["pipeline", "error_type", "message"],
            )
        )
    lines.append("\n## Sampling audit\n")
    if sampling_audit.empty:
        lines.append("_no resampling folds ran_\n")
    else:
        audit_summary = (
            sampling_audit.groupby(["strategy", "audit_status"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        lines.append(_markdown_table(audit_summary, ["strategy", "audit_status", "n"]))
    lines.append("\n## Artifacts\n")
    for key in sorted(artifact_paths):
        lines.append(f"* {key}: `{artifact_paths[key]}`\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class ImbalanceRunResult:
    """Everything one imbalance experiment run produced."""

    run_directory: RunDirectory
    grid: dict[str, Any]
    fold_plan: list[FoldSpec]
    plan_info: dict[str, Any]
    cv: pd.DataFrame
    oof: pd.DataFrame
    summary: pd.DataFrame
    summary_wide: pd.DataFrame
    pooled_by_repeat: pd.DataFrame
    sampling_audit: pd.DataFrame
    warnings_table: pd.DataFrame
    errors_table: pd.DataFrame
    pipeline_status: dict[str, str]
    pipeline_seconds: dict[str, float]
    verification: dict[str, Any]
    manifest: dict[str, Any]
    paths: dict[str, Path]
    total_seconds: float

    @property
    def overall_passed(self) -> bool:
        return bool(self.verification["overall_passed"])


def run_imbalance_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    grid: Mapping[str, Any],
    *,
    data_metadata: Mapping[str, Any],
    step: str = "imbalance_cv",
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
    random_state: int = RANDOM_STATE,
    threshold: float = DECISION_THRESHOLD,
    sampling_ratio: float = IMBALANCE_DEFAULT_SAMPLING_RATIO,
    k_neighbors: int = IMBALANCE_DEFAULT_K_NEIGHBORS,
    sampler_seed: int = IMBALANCE_RANDOM_STATE,
    fold_plan: Sequence[FoldSpec] | None = None,
    expect_full_grid: bool = False,
    run_id: str | None = None,
    results_dir: Path | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> ImbalanceRunResult:
    """Run the imbalance grid over a shared fold plan with strict capture.

    Parameters
    ----------
    X, y
        The training partition only. The held-out test set must never be
        passed here.
    grid
        Mapping ``"model__strategy"`` -> unfitted pipeline (the builders in
        :mod:`src.imbalance` / :mod:`src.imbalance_pipelines`).
    data_metadata
        Must contain ``data_sha256`` (64-hex) and ``heldout_test_was_locked``
        (True) — assertions below refuse anything else.
    n_splits, n_repeats, random_state, threshold
        Protocol configuration recorded in the repeatability pin.
    sampling_ratio, k_neighbors, sampler_seed
        Default resampling parameters; per-pipeline overrides inside ``grid``
        are allowed but then the pin check compares against the *recorded*
        sampler parameters (mismatches fail verification if the override
        contradicts the pin intending defaults).
    fold_plan
        Optional pre-materialised shared :class:`FoldSpec` plan.
    expect_full_grid
        ``True`` (the future full 5x5 run) additionally asserts that the grid
        equals the complete supported 25-pipeline set.
    run_id, results_dir
        Run-directory overrides (production runs must keep ``results_dir``).
    extra_manifest
        Optional additional manifest fields from the caller.

    Returns
    -------
    ImbalanceRunResult
        Frames, statuses, verification and manifest; ``overall_passed`` is
        False when any strict check failed — failures are never silently
        converted into successful experiments.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
    if len(X) != len(y):
        raise RunConfigurationError(
            f"X/y length mismatch: {len(X)} vs {len(y)}."
        )
    if not HEX64_PATTERN.match(str(data_metadata.get("data_sha256", ""))):
        raise RunConfigurationError(
            "data_metadata must include the raw-data 'data_sha256' (64 hex)."
        )
    if not bool(data_metadata.get("heldout_test_was_locked")):
        raise RunConfigurationError(
            "The held-out test container must be locked to run imbalance "
            "experiments; imbalance runs must not read the test partition."
        )
    if not grid:
        raise RunConfigurationError("The pipeline grid is empty.")

    run_directory = create_run_directory(run_id, results_dir)
    logger = configure_run_logger(
        run_directory.run_id, run_directory.log_path()
    )
    started = time.perf_counter()
    # G2: freeze the source tree at run start (exit snapshot is taken before
    # manifest serialization, and differences are flagged by verification).
    source_hashes_start = source_file_hashes()
    logger.info("=" * 76)
    logger.info("StrokeSure imbalance experiment (run-scoped persistence)")
    logger.info("run_id=%s step=%s", run_directory.run_id, step)
    logger.info("=" * 76)

    # -- audited grid + build-phase warnings ---------------------------------
    build_warning_records: list[dict[str, Any]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audited_grid = build_audited_grid(grid)
        pipeline_parameters = describe_pipelines(audited_grid)
    for warning in caught:
        record = {
            "pipeline": "<grid-build>",
            "phase": "build",
            "category": warning.category.__name__,
            "message": str(warning.message),
            "filename": warning.filename,
            "lineno": int(warning.lineno) if warning.lineno is not None else -1,
        }
        build_warning_records.append(record)
        logger.info(
            "build warning: %s: %s", record["category"], record["message"][:120]
        )

    # -- shared fold plan ----------------------------------------------------
    if fold_plan is None:
        fold_plan = build_fold_plan(
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    plan_info = validate_shared_fold_plan(
        fold_plan, n_rows=len(X), n_splits=n_splits, n_repeats=n_repeats
    )
    logger.info(
        "fold plan: %d folds total (%d x %d), random_state=%d",
        plan_info["n_folds_total"],
        n_splits,
        n_repeats,
        random_state,
    )

    # -- planned artifact paths ----------------------------------------------
    csv_artifacts = (
        "fold_metrics",
        "oof_predictions",
        "summary",
        "summary_wide",
        "pooled_by_repeat",
        "sampling_audit",
        "warnings",
        "errors",
        "fold_plan",
    )
    paths: dict[str, Path] = {
        stem: run_directory.csv_path(stem) for stem in csv_artifacts
    }
    paths["verification"] = run_directory.json_path("verification")
    paths["report"] = run_directory.report_path()
    paths["manifest"] = run_directory.json_path("manifest")

    # -- per-(pipeline, fold) execution with strict capture -------------------
    # Every fold is an independent, individually-audited attempt: warnings and
    # exceptions are attributed to the exact (pipeline, repeat, fold) pair and
    # a failed fold never blocks or silently hides the remaining folds.
    pipeline_status: dict[str, str] = {}
    pipeline_seconds: dict[str, float] = {}
    pipeline_fold_counts: dict[str, dict[str, int]] = {}
    warning_records: list[dict[str, Any]] = list(build_warning_records)
    error_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    cv_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []

    for pipeline_key, pipeline in audited_grid.items():
        model_name, _, strategy_name = pipeline_key.partition("__")
        logger.info("pipeline %s: starting", pipeline_key)
        pipeline_started = time.perf_counter()
        n_ok = 0
        n_failed = 0
        pipeline_oof_parts: list[pd.DataFrame] = []

        for spec in fold_plan:
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

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fold_cv, fold_oof, _fold_meta = run_imbalance_cv(
                        X,
                        y,
                        {pipeline_key: pipeline},
                        fold_plan=[spec],
                        threshold=threshold,
                        fold_observer=observer,
                        validate_plan=False,
                    )
            except Exception as exc:  # noqa: BLE001 -- captured, never silent
                n_failed += 1
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
                    }
                )
                logger.error(
                    "pipeline %s fold r%d f%d FAILED: %s: %s",
                    pipeline_key,
                    spec.repeat,
                    spec.fold,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                continue

            audit_records.extend(fold_audits)
            for warning in caught:
                record = {
                    "pipeline": pipeline_key,
                    "phase": "fit/predict",
                    "repeat": int(spec.repeat),
                    "fold": int(spec.fold),
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": int(warning.lineno)
                    if warning.lineno is not None
                    else -1,
                }
                warning_records.append(record)
            cv_parts.append(fold_cv)
            # OOF predictions only count for pipelines that complete every
            # fold; a partially-fitted pipeline must not contribute OOF rows.
            pipeline_oof_parts.append(fold_oof)
            n_ok += 1

        pipeline_fold_counts[pipeline_key] = {
            "successful": n_ok,
            "failed": n_failed,
        }
        pipeline_status[pipeline_key] = "ok" if n_failed == 0 else "failed"
        pipeline_seconds[pipeline_key] = time.perf_counter() - pipeline_started
        logger.info(
            "pipeline %s: status=%s in %.2fs (%d ok / %d failed folds)",
            pipeline_key,
            pipeline_status[pipeline_key],
            pipeline_seconds[pipeline_key],
            n_ok,
            n_failed,
        )
        if n_failed == 0:
            oof_parts.extend(pipeline_oof_parts)

    cv = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    summary, summary_wide = build_fold_summaries(cv)
    pooled_by_repeat = build_pooled_within_repeat(oof)
    sampling_audit = pd.DataFrame(audit_records)
    warnings_table = pd.DataFrame(warning_records)
    errors_table = pd.DataFrame(error_records)
    logger.info(
        "execution totals: fold_records=%d oof_rows=%d audits=%d warnings=%d errors=%d",
        len(cv),
        len(oof),
        len(sampling_audit),
        len(warnings_table),
        len(errors_table),
    )

    # -- persist fold-plan and total artifacts before verification ------------
    fold_plan_rows = [
        {
            "repeat": int(spec.repeat),
            "fold": int(spec.fold),
            "n_train": int(len(spec.train_positions)),
            "n_validation": int(len(spec.valid_positions)),
            "n_train_positive": int(y.iloc[spec.train_positions].sum()),
            "n_validation_positive": int(y.iloc[spec.valid_positions].sum()),
            "train_index_fingerprint": _index_fingerprint(
                X.index.values[spec.train_positions]
            ),
            "validation_index_fingerprint": _index_fingerprint(
                X.index.values[spec.valid_positions]
            ),
        }
        for spec in fold_plan
    ]
    frames: dict[str, pd.DataFrame] = {
        "fold_plan": pd.DataFrame(fold_plan_rows),
        "fold_metrics": cv,
        "oof_predictions": oof,
        "summary": summary,
        "summary_wide": summary_wide,
        "pooled_by_repeat": pooled_by_repeat,
        "sampling_audit": sampling_audit,
        "warnings": warnings_table,
        "errors": errors_table,
    }
    for stem, frame in frames.items():
        _write_frame(frame, paths[stem])
        logger.info(
            "artifact %s: %s", stem, _relative(paths[stem])
        )
    written_csv = {stem: paths[stem] for stem in frames}

    # -- draft manifest, verification, final manifest/report ------------------
    pin = _repeatability_pin(
        step=step,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        threshold=threshold,
        sampling_ratio=sampling_ratio,
        k_neighbors=k_neighbors,
        sampler_seed=sampler_seed,
        grid_keys=list(audited_grid),
        n_rows=len(X),
    )
    config = {
        "step": step,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "threshold": threshold,
        "sampling_ratio": sampling_ratio,
        "k_neighbors": k_neighbors,
        "sampler_seed": sampler_seed,
        "expect_full_grid": expect_full_grid,
    }
    manifest = _draft_manifest(
        run_directory=run_directory,
        step=step,
        data_metadata=data_metadata,
        config=config,
        plan_info=plan_info,
        pipeline_parameters=pipeline_parameters,
        pipeline_status=pipeline_status,
        pipeline_seconds=pipeline_seconds,
        pipeline_fold_counts=pipeline_fold_counts,
        source_hashes_start=source_hashes_start,
        pin=pin,
        grid_keys=list(audited_grid),
    )
    manifest["outcomes"] = {
        "n_pipelines": len(audited_grid),
        "n_fold_records": int(len(cv)),
        "n_oof_predictions": int(len(oof)),
        "n_sampling_audits": int(len(sampling_audit)),
        "n_warnings": int(len(warnings_table)),
        "n_errors": int(len(errors_table)),
        "n_failed_folds": int(sum(c["failed"] for c in pipeline_fold_counts.values())),
    }

    verification = verify_imbalance_run(
        run_directory=run_directory,
        grid=audited_grid,
        fold_plan=fold_plan,
        plan_info=plan_info,
        X=X,
        y=y,
        cv=cv,
        oof=oof,
        sampling_audit=sampling_audit,
        warnings_table=warnings_table,
        errors_table=errors_table,
        pipeline_status=pipeline_status,
        pipeline_parameters=pipeline_parameters,
        manifest=manifest,
        paths=paths,
        written_paths=written_csv,
        config=config,
    )
    write_json(verification, paths["verification"])

    artifact_paths_rel = {
        key: _relative(value) for key, value in paths.items() if key != "manifest"
    }
    _write_report(
        paths["report"],
        run_directory=run_directory,
        data_metadata=data_metadata,
        config=config,
        verification=verification,
        summary_wide=summary_wide,
        pooled_by_repeat=pooled_by_repeat,
        sampling_audit=sampling_audit,
        warnings_table=warnings_table,
        errors_table=errors_table,
        pipeline_seconds=pipeline_seconds,
        artifact_paths=artifact_paths_rel,
    )

    total_seconds = time.perf_counter() - started
    manifest["verification"] = {
        "overall_passed": verification["overall_passed"],
        "n_checks": verification["n_checks"],
        "n_passed": verification["n_passed"],
        "n_failed": verification["n_failed"],
        "path": _relative(paths["verification"]),
    }
    manifest["total_seconds"] = round(float(total_seconds), 3)
    manifest["files"] = {
        **run_directory.artifact_map({k: v for k, v in paths.items() if k != "manifest"}),
        "manifest": {"path": _relative(paths["manifest"]), "sha256": None},
    }
    manifest["manifest_note"] = (
        "manifest.json cannot hash itself; every other artifact's SHA-256 is listed."
    )
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    write_json(manifest, paths["manifest"])

    status_text = (
        "PASSED" if verification["overall_passed"] else "FAILED"
    )
    logger.info(
        "run %s verification: %s (%d/%d checks)",
        run_directory.run_id,
        status_text,
        verification["n_passed"],
        verification["n_checks"],
    )
    logger.info(
        "total wall time %.2fs (including artifact persistence)", total_seconds
    )

    return ImbalanceRunResult(
        run_directory=run_directory,
        grid=audited_grid,
        fold_plan=list(fold_plan),
        plan_info=plan_info,
        cv=cv,
        oof=oof,
        summary=summary,
        summary_wide=summary_wide,
        pooled_by_repeat=pooled_by_repeat,
        sampling_audit=sampling_audit,
        warnings_table=warnings_table,
        errors_table=errors_table,
        pipeline_status=pipeline_status,
        pipeline_seconds=pipeline_seconds,
        verification=verification,
        manifest=manifest,
        paths=paths,
        total_seconds=total_seconds,
    )


# ---------------------------------------------------------------------------
# Training-partition loader (test set stays locked)
# ---------------------------------------------------------------------------
def load_training_partition(
    path: Path | str = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load only the training partition and shape metadata about the rest.

    The split machinery may compute aggregate counts of the test partition
    internally, but this helper copies only training-level fields; the test
    partition's data is never read, modelled or evaluated here.
    """
    split = load_split(path)
    if split.test_unlocked:
        raise RunConfigurationError(
            "The held-out test container must stay locked for imbalance runs."
        )
    y = split.y_train
    metadata = {
        "data_sha256": split.metadata["data_sha256"],
        "n_rows": int(split.metadata["n_rows"]),
        "n_features": int(split.metadata["n_features"]),
        "n_train": int(split.metadata["n_train"]),
        "n_test": int(split.metadata["n_test"]),
        "test_size": float(split.metadata["test_size"]),
        "random_state": int(split.metadata["random_state"]),
        "full_prevalence": float(split.metadata["full_prevalence"]),
        "train_prevalence": float(y.mean()),
        "train_class_counts": {
            "1": int((y == 1).sum()),
            "0": int((y == 0).sum()),
        },
        "heldout_test_was_locked": True,
        "heldout_test_data_fields_accessed": False,
        "heldout_test_note": (
            "Only training fields were loaded; test data was never accessed."
        ),
    }
    return split.X_train, y, metadata


# ---------------------------------------------------------------------------
__all__ = [
    "CONFIGURATION_FILES",
    "EXPECTED_WARNING_CATEGORIES",
    "EXPECTED_WARNING_MESSAGE_PREFIX",
    "ImbalanceRunResult",
    "OutputIsolationError",
    "RESAMPLING_STRATEGIES",
    "RUN_DIRECTORY_PARENT",
    "RunConfigurationError",
    "RunDirectory",
    "RUNNER_SOURCE_FILES",
    "audit_resampling_fold",
    "build_audited_grid",
    "build_fold_summaries",
    "build_pooled_within_repeat",
    "classify_warning",
    "configuration_file_hashes",
    "core_environment",
    "create_run_directory",
    "describe_pipelines",
    "diff_source_hashes",
    "file_sha256",
    "load_training_partition",
    "pip_freeze",
    "protected_paths",
    "run_imbalance_experiment",
    "source_file_hashes",
    "verify_imbalance_run",
]
