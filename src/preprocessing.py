"""Leakage-safe preprocessing that reproduces Notebook 02.

Notebook 02 defined two preprocessing configurations:

linear / SVM models
    median imputation -> StandardScaler on continuous features;
    binary features passed through; one-hot encoding with ``drop="first"``.
    -> 16 predictors.

tree-based models
    median imputation on continuous features (no scaling);
    binary features passed through; full one-hot encoding.
    -> 21 predictors.

Both configurations wrap the transformer in a :class:`~sklearn.pipeline.Pipeline`
together with the estimator (see :func:`build_model_pipeline`).  This means the
imputer, scaler and encoder are re-fitted from scratch inside every
cross-validation training fold -- the full training set is never transformed
ahead of time.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FAMILY_LINEAR_SVM,
    MODEL_FAMILY_TREE,
    MODEL_FAMILIES,
    NUMERIC_FEATURES,
    SMOKING_COLUMN,
    SMOKING_UNKNOWN_VALUE,
)

# Preprocessing settings per model family, mirroring Notebook 02.
FAMILY_SETTINGS: dict[str, dict[str, bool]] = {
    MODEL_FAMILY_LINEAR_SVM: {"scale": True, "drop_first": True},
    MODEL_FAMILY_TREE: {"scale": False, "drop_first": False},
}

# One-hot encoding drops one level per categorical feature for the linear /
# SVM family.  scikit-learn orders levels lexicographically and "Unknown"
# starts with an upper-case letter, so with the default ordering it would be
# the level that gets dropped -- i.e. the missingness indicator would silently
# disappear.  All category orders are therefore pinned explicitly, with
# "Unknown" last for smoking status.  This preserves Notebook 02's 16-predictor
# design while keeping the unknown-smoking indicator as a real column.
CATEGORY_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male", "Other"],
    "ever_married": ["No", "Yes"],
    "work_type": [
        "Govt_job",
        "Never_worked",
        "Private",
        "Self-employed",
        "children",
    ],
    "Residence_type": ["Rural", "Urban"],
    SMOKING_COLUMN: [
        "formerly smoked",
        "never smoked",
        "smokes",
        SMOKING_UNKNOWN_VALUE,
    ],
}


def _categories_argument() -> list[list[str]]:
    """Explicit, ordered categories for every categorical feature.

    scikit-learn 1.9 no longer accepts a per-feature ``"auto"`` placeholder
    inside an explicit ``categories`` list, so every feature is listed.
    """
    return [CATEGORY_VALUES[feature] for feature in CATEGORICAL_FEATURES]


def build_numeric_pipeline(scale: bool) -> Pipeline:
    """Median imputation, optionally followed by standardisation."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def build_categorical_pipeline(drop_first: bool = False) -> Pipeline:
    """One-hot encoding that tolerates unseen categories at transform time.

    ``handle_unknown="ignore"`` emits an all-zero block for categories that
    were not observed during fitting, so production/edge-case rows cannot
    crash a fitted model.
    """
    return Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    categories=_categories_argument(),
                    handle_unknown="ignore",
                    drop="first" if drop_first else None,
                    sparse_output=False,
                ),
            )
        ]
    )


def build_preprocessor(family: str = MODEL_FAMILY_TREE) -> ColumnTransformer:
    """Build the unfitted ``ColumnTransformer`` for a model family.

    Parameters
    ----------
    family
        ``"linear_svm"`` (scaled, drop-first one-hot) or ``"tree"``
        (unscaled, full one-hot).

    Notes
    -----
    The transformer is returned **unfitted** on purpose. Fitting it on the
    full training set before cross-validation would leak fold information.
    """
    if family not in FAMILY_SETTINGS:
        raise ValueError(
            f"Unknown model family {family!r}; expected one of {MODEL_FAMILIES}."
        )

    settings = FAMILY_SETTINGS[family]
    return ColumnTransformer(
        transformers=[
            ("num", build_numeric_pipeline(settings["scale"]), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                build_categorical_pipeline(settings["drop_first"]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(
    estimator: BaseEstimator, family: str
) -> Pipeline:
    """Bundle preprocessing and estimator into one cross-validatable pipeline.

    ``Pipeline.fit`` fits the preprocessor on the data it receives; when the
    pipeline is cloned and fitted per CV fold, preprocessing is therefore
    fold-local by construction.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(family)),
            ("classifier", estimator),
        ]
    )


def get_fitted_preprocessor(pipeline: Pipeline) -> ColumnTransformer:
    """Return the fitted preprocessor from a fitted model pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'preprocessor' step.")
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("Preprocessor has not been fitted yet.")
    return preprocessor


def feature_names_out(pipeline_or_preprocessor: Any) -> list[str]:
    """Return output feature names for a fitted pipeline or preprocessor."""
    if isinstance(pipeline_or_preprocessor, Pipeline):
        preprocessor = get_fitted_preprocessor(pipeline_or_preprocessor)
    else:
        preprocessor = pipeline_or_preprocessor
    return list(preprocessor.get_feature_names_out())


def has_explicit_unknown_smoking_status(
    pipeline_or_preprocessor: Any,
) -> bool:
    """Whether ``smoking_status="Unknown"`` survived encoding as its own column.

    The dataset encodes unknown smoking status as the literal string
    ``"Unknown"``. Treating it as a missing value and imputing it would
    silently discard information, so the preprocessing keeps it as an explicit
    one-hot category.
    """
    expected = f"cat__{SMOKING_COLUMN}_{SMOKING_UNKNOWN_VALUE}"
    names = feature_names_out(pipeline_or_preprocessor)
    return expected in names
