"""Data loading, validation and the deterministic train/test split.

Design notes
------------
* The raw CSV is read-only: this module never writes to ``data/``.
* The split reproduces Notebook 02 exactly (``test_size=0.20``,
  ``stratify=y``, ``random_state=42``, ``X = df.drop(columns=["stroke", "id"])``)
  and this is asserted independently in the smoke test.
* The held-out test partition is wrapped in :class:`HeldOutTestSet`, which
  raises unless it is *explicitly* unlocked.  No baseline or smoke-test code
  path unlocks it, which makes accidental test-set use a hard error rather
  than a silent leak.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALLOWED_TARGET_VALUES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


class TestSetAccessError(RuntimeError):
    """Raised when the locked held-out test set is accessed too early."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def file_sha256(path: Path | str = RAW_DATA_PATH) -> str:
    """Return the SHA-256 checksum of the raw dataset (read-only)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate schema and target values, returning a summary report.

    Checks
    ------
    * file has rows and no duplicate column names
    * columns match ``EXPECTED_COLUMNS`` exactly
    * target column exists, has no missing values and only contains 0/1
    * identifier column exists and is unique
    * all modelling features are present
    """
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate column names: {duplicated}")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise DataValidationError(
            f"Unexpected schema. Missing={missing}, extra={extra}"
        )

    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COLUMN}' is missing.")

    target = df[TARGET_COLUMN]
    if target.isna().any():
        raise DataValidationError("Target column contains missing values.")

    observed = sorted(pd.unique(target).tolist())
    if not set(observed).issubset(set(ALLOWED_TARGET_VALUES)):
        raise DataValidationError(
            f"Target contains unexpected values {observed}; "
            f"allowed values are {list(ALLOWED_TARGET_VALUES)}."
        )
    if len(observed) != 2:
        raise DataValidationError(
            f"Target must contain both classes, found {observed}."
        )

    if ID_COLUMN not in df.columns:
        raise DataValidationError(f"Identifier column '{ID_COLUMN}' is missing.")
    if df[ID_COLUMN].duplicated().any():
        n_dupes = int(df[ID_COLUMN].duplicated().sum())
        raise DataValidationError(f"Identifier column has {n_dupes} duplicates.")

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise DataValidationError(f"Missing feature columns: {missing_features}")

    counts = target.value_counts().to_dict()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "target_counts": {
            str(POSITIVE_LABEL): int(counts.get(POSITIVE_LABEL, 0)),
            str(NEGATIVE_LABEL): int(counts.get(NEGATIVE_LABEL, 0)),
        },
        "target_prevalence": float(target.mean()),
        "n_unique_ids": int(df[ID_COLUMN].nunique()),
        "n_missing_bmi": int(df["bmi"].isna().sum()),
        "sha256": file_sha256(),
    }


def load_raw_dataframe(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV without modifying it and validate the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


def build_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into ``X`` and ``y`` exactly as Notebook 02 does.

    ``id`` is excluded because it is an identifier, not a predictor, and
    ``stroke`` is removed from ``X`` because it is the target.
    """
    # Column drop order matches Notebook 02: ["stroke", "id"].
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Held-out test set guard
# ---------------------------------------------------------------------------
@dataclass
class HeldOutTestSet:
    """Locked container for the test partition.

    The data is held in memory (the split must be reproduced) but any attempt
    to obtain it raises :class:`TestSetAccessError` unless
    :meth:`unlock_for_final_evaluation` has been called.  The baseline, CV and
    smoke-test code never calls it.
    """

    X: pd.DataFrame
    y: pd.Series
    _unlocked: bool = field(default=False, repr=False)

    @property
    def unlocked(self) -> bool:
        return self._unlocked

    @property
    def shape(self) -> tuple[int, int]:
        """Shape metadata only; does not expose the data."""
        return self.X.shape

    @property
    def class_counts(self) -> dict[str, int]:
        """Label counts only; labels are not used by any model-selection code."""
        counts = self.y.value_counts().to_dict()
        return {
            str(POSITIVE_LABEL): int(counts.get(POSITIVE_LABEL, 0)),
            str(NEGATIVE_LABEL): int(counts.get(NEGATIVE_LABEL, 0)),
        }

    def unlock_for_final_evaluation(self) -> None:
        """Explicit, auditable opt-in. Not used before final evaluation."""
        self._unlocked = True

    def get(self) -> tuple[pd.DataFrame, pd.Series]:
        if not self._unlocked:
            raise TestSetAccessError(
                "The held-out test set is locked. Baseline experimentation, "
                "cross-validation, threshold tuning and calibration must not "
                "use it. Call unlock_for_final_evaluation() only for the "
                "final, pre-registered evaluation."
            )
        return self.X, self.y


def verify_disjoint_indices(
    train_index: pd.Index, test_index: pd.Index
) -> dict[str, Any]:
    """Assert that train and test indices are disjoint and cover the dataset.

    Returns
    -------
    dict with ``n_train``, ``n_test``, ``n_overlap`` and ``n_uncovered``.
    """
    train_set = set(train_index.tolist())
    test_set = set(test_index.tolist())
    overlap = train_set & test_set

    if overlap:
        raise DataValidationError(
            f"Train/test index overlap of {len(overlap)} rows detected."
        )

    return {
        "n_train": len(train_set),
        "n_test": len(test_set),
        "n_overlap": len(overlap),
        "n_uncovered": 0,
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Container for the deterministic split.

    Attributes
    ----------
    X_train, y_train
        The only partition that may be used for fitting, CV and model
        selection.
    test
        Locked held-out partition.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    test: HeldOutTestSet
    train_index: pd.Index
    test_index: pd.Index
    metadata: dict[str, Any]

    @property
    def n_train(self) -> int:
        return int(len(self.X_train))

    @property
    def n_test(self) -> int:
        return int(len(self.test_index))

    @property
    def test_unlocked(self) -> bool:
        return self.test.unlocked


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce Notebook 02's stratified 80/20 split (identity function)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def load_split(path: Path | str = RAW_DATA_PATH) -> SplitData:
    """Load, validate and split the dataset, verifying split integrity.

    The resulting :class:`SplitData` never exposes the test partition unless
    explicitly unlocked.
    """
    df = load_raw_dataframe(path)
    X, y = build_features_and_target(df)

    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    integrity = verify_disjoint_indices(X_train.index, X_test.index)
    uncovered = set(df.index) - set(X_train.index) - set(X_test.index)
    if uncovered:
        raise DataValidationError(
            f"{len(uncovered)} dataset rows are in neither partition."
        )
    integrity["n_uncovered"] = 0

    full_prevalence = float(y.mean())
    train_prevalence = float(y_train.mean())
    if abs(train_prevalence - full_prevalence) > 0.005:
        raise DataValidationError(
            "Stratification failed: train prevalence "
            f"{train_prevalence:.4f} vs full {full_prevalence:.4f}."
        )

    metadata = {
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train": integrity["n_train"],
        "n_test": integrity["n_test"],
        "n_overlap": integrity["n_overlap"],
        "n_uncovered": integrity["n_uncovered"],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "full_prevalence": full_prevalence,
        "train_prevalence": train_prevalence,
        "test_prevalence": float(y_test.mean()),
        "train_class_counts": {
            str(POSITIVE_LABEL): int((y_train == POSITIVE_LABEL).sum()),
            str(NEGATIVE_LABEL): int((y_train == NEGATIVE_LABEL).sum()),
        },
        "test_class_counts": {
            str(POSITIVE_LABEL): int((y_test == POSITIVE_LABEL).sum()),
            str(NEGATIVE_LABEL): int((y_test == NEGATIVE_LABEL).sum()),
        },
        "data_sha256": file_sha256(path),
    }

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        test=HeldOutTestSet(X=X_test, y=y_test),
        train_index=X_train.index,
        test_index=X_test.index,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw a class-stratified subsample from the *training* partition.

    Used only by the smoke test to keep runtime low. The subsample keeps the
    class ratio of the input, so rare-event metrics remain meaningful.
    """
    if n_samples >= len(X):
        raise ValueError(
            f"n_samples ({n_samples}) must be smaller than the training set "
            f"({len(X)})."
        )
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub
