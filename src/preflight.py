"""Step 21 preflight checks.

Run with::

    python -m src.preflight

The preflight is deliberately read-only: it inspects the data, the split, the
pipeline definitions and the repeated-CV fold plan, and fails loudly if any
invariant that the experiments rely on is violated.  It does not fit any model
on the full training set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.config import (
    CV_N_REPEATS,
    CV_N_SPLITS,
    EXPECTED_FEATURE_COUNTS,
    RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loading import TestSetAccessError, load_split
from src.models import BASELINE_MODEL_SPECS, build_baseline_pipelines
from src.preprocessing import build_preprocessor


class PreflightError(AssertionError):
    """Raised when an experiment precondition is not satisfied."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


@dataclass
class FoldSpec:
    """One CV fold, identified by repeat and fold number."""

    repeat: int
    fold: int
    train_positions: np.ndarray
    valid_positions: np.ndarray


def build_fold_plan(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> list[FoldSpec]:
    """Materialise the 5x5 fold plan once, so every model sees identical folds.

    ``RepeatedStratifiedKFold`` emits repeat-major splits: the first
    ``n_splits`` members belong to repeat 1, then repeat 2, and so on.  The
    ordering assumption is verified by :func:`validate_fold_plan`.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    plan: list[FoldSpec] = []
    for position, (train_positions, valid_positions) in enumerate(
        splitter.split(X, y)
    ):
        plan.append(
            FoldSpec(
                repeat=position // n_splits + 1,
                fold=position % n_splits + 1,
                train_positions=np.asarray(train_positions, dtype=int),
                valid_positions=np.asarray(valid_positions, dtype=int),
            )
        )
    return plan


def validate_fold_plan(plan: list[FoldSpec], n_rows: int) -> dict[str, Any]:
    """Verify coverage, disjointness, pairing and repeat independence."""
    check(
        len(plan) == CV_N_SPLITS * CV_N_REPEATS,
        f"Expected {CV_N_SPLITS * CV_N_REPEATS} folds, found {len(plan)}.",
    )

    all_positions = np.arange(n_rows)
    repeats: dict[int, list[FoldSpec]] = {}
    for spec in plan:
        repeats.setdefault(spec.repeat, []).append(spec)

    check(
        sorted(repeats) == list(range(1, CV_N_REPEATS + 1)),
        f"Unexpected repeat identifiers: {sorted(repeats)}.",
    )

    fingerprints: dict[int, str] = {}
    for repeat, specs in repeats.items():
        check(
            len(specs) == CV_N_SPLITS,
            f"Repeat {repeat} has {len(specs)} folds.",
        )

        covered = np.concatenate([s.valid_positions for s in specs])
        check(
            len(covered) == n_rows and np.array_equal(np.sort(covered), all_positions),
            f"Repeat {repeat}: validation folds do not partition the data exactly once.",
        )

        for spec in specs:
            check(
                len(np.intersect1d(spec.train_positions, spec.valid_positions)) == 0,
                f"Repeat {repeat} fold {spec.fold}: train/validation overlap.",
            )
            check(
                len(spec.train_positions) + len(spec.valid_positions) == n_rows,
                f"Repeat {repeat} fold {spec.fold}: train + validation != n_rows.",
            )

        assignment = np.empty(n_rows, dtype=int)
        for spec in specs:
            assignment[spec.valid_positions] = spec.fold
        fingerprints[repeat] = hashlib.sha256(
            assignment.astype(np.int8).tobytes()
        ).hexdigest()[:16]

    check(
        len(set(fingerprints.values())) == CV_N_REPEATS,
        "Repeats do not produce distinct fold assignments (same shuffle reused).",
    )

    return {
        "n_folds_total": len(plan),
        "n_splits": CV_N_SPLITS,
        "n_repeats": CV_N_REPEATS,
        "rows": int(n_rows),
        "repeat_assignment_fingerprints": fingerprints,
        "pairing": "All models iterate the same FoldSpec list.",
    }


def run_preflight() -> dict[str, Any]:
    """Execute all preflight checks and print a summary."""
    report: dict[str, Any] = {}

    print("=" * 78)
    print("STEP 21 - EXPERIMENT PREFLIGHT")
    print("=" * 78)

    # --- data and split -----------------------------------------------------
    split = load_split()
    check(split.n_train == 4088, f"Training partition is {split.n_train}, expected 4088.")
    check(split.n_test == 1022, f"Test partition is {split.n_test}, expected 1022.")
    check(
        len(set(split.train_index) & set(split.test_index)) == 0,
        "Train/test indices overlap.",
    )
    check(
        split.metadata["data_sha256"]
        == hashlib.sha256(Path(RAW_DATA_PATH).read_bytes()).hexdigest(),
        "Data hash does not match the raw file.",
    )

    print(f"data file            : {Path(RAW_DATA_PATH).name}")
    print(f"data sha256          : {split.metadata['data_sha256'][:16]}...")
    print(f"training partition   : {split.n_train} rows ({split.metadata['train_class_counts']})")
    print(f"held-out partition   : {split.n_test} rows ({split.metadata['test_class_counts']})")
    print(f"split parameters     : test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratified")

    # --- test-set lock ------------------------------------------------------
    check(not split.test.unlocked, "Held-out test set is unlocked.")
    raised = False
    try:
        split.test.get()
    except TestSetAccessError:
        raised = True
    check(raised, "Locked test set did not refuse access.")
    print("test set             : LOCKED (access raises TestSetAccessError)")

    # --- pipelines and preprocessing ---------------------------------------
    pipelines = build_baseline_pipelines()
    check(len(pipelines) == 5, f"Expected 5 baselines, found {len(pipelines)}.")
    print("models               : " + ", ".join(pipelines))

    unfitted_preprocessors = 0
    class_weight_flags: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        check(isinstance(pipeline, Pipeline), f"{name} is not a sklearn Pipeline.")
        check(
            list(pipeline.named_steps) == ["preprocessor", "classifier"],
            f"{name}: unexpected pipeline steps {list(pipeline.named_steps)}.",
        )
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        check(
            not hasattr(preprocessor, "transformers_"),
            f"{name}: preprocessor is already fitted outside CV.",
        )
        try:
            check_is_fitted(classifier)
            fitted = True
        except NotFittedError:
            fitted = False
        check(not fitted, f"{name}: classifier is already fitted outside CV.")
        unfitted_preprocessors += 1

        family = BASELINE_MODEL_SPECS[name].family
        expected = EXPECTED_FEATURE_COUNTS[family]
        class_weight_flags[name] = {
            "family": family,
            "expected_features": expected,
            "class_weight": classifier.get_params().get("class_weight", None),
        }
        check(
            class_weight_flags[name]["class_weight"] is None,
            f"{name}: class_weight is set, but baseline experiments are unweighted.",
        )

        check(
            isinstance(build_preprocessor(family), type(preprocessor)),
            f"{name}: preprocessing factory mismatch.",
        )

    check(
        unfitted_preprocessors == 5,
        "Not all preprocessors were confirmed unfitted.",
    )
    print("preprocessing        : 0 fitted transformers outside CV (all pipelines unfitted)")
    print("class weighting      : none (class_weight=None for every estimator)")
    for name, info in class_weight_flags.items():
        print(
            f"  {name:22s} family={info['family']:10s} "
            f"features={info['expected_features']:2d}"
        )

    # --- fold plan ----------------------------------------------------------
    plan = build_fold_plan(split.X_train, split.y_train)
    plan_info = validate_fold_plan(plan, n_rows=len(split.X_train))
    report["fold_plan"] = plan_info
    report["fold_plan_specs"] = [
        {
            "repeat": spec.repeat,
            "fold": spec.fold,
            "n_train": int(len(spec.train_positions)),
            "n_validation": int(len(spec.valid_positions)),
            "n_positive_validation": int(
                split.y_train.iloc[spec.valid_positions].sum()
            ),
            "train_index_fingerprint": hashlib.sha256(
                np.sort(split.train_index.values[spec.train_positions])
                .astype(np.int64)
                .tobytes()
            ).hexdigest()[:16],
            "validation_index_fingerprint": hashlib.sha256(
                np.sort(split.train_index.values[spec.valid_positions])
                .astype(np.int64)
                .tobytes()
            ).hexdigest()[:16],
        }
        for spec in plan
    ]

    print(
        f"CV plan              : {CV_N_SPLITS} folds x {CV_N_REPEATS} repeats "
        f"= {plan_info['n_folds_total']} folds, random_state={RANDOM_STATE}"
    )
    print("  repeat shuffles    : " + json.dumps(plan_info["repeat_assignment_fingerprints"]))
    print("  pairing            : all models share the identical FoldSpec list")

    # --- sanity on fold sizes ----------------------------------------------
    fold_sizes = {len(spec.valid_positions) for spec in plan}
    positives = {int(split.y_train.iloc[spec.valid_positions].sum()) for spec in plan}
    print(f"  validation sizes   : {sorted(fold_sizes)} rows")
    print(f"  positives per fold : min={min(positives)}, max={max(positives)}")

    report["data_sha256"] = split.metadata["data_sha256"]
    report["n_train"] = split.n_train
    report["n_test"] = split.n_test
    report["test_locked"] = True
    report["models"] = class_weight_flags

    print()
    print("PREFLIGHT: PASSED")
    return report


if __name__ == "__main__":
    try:
        run_preflight()
    except PreflightError as exc:
        print(f"PREFLIGHT: FAILED - {exc}", file=sys.stderr)
        raise SystemExit(1)
