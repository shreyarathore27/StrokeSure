"""Leakage-safe imbalance pipelines for mixed categorical/continuous data.

Why this module exists
----------------------
The resampling branch of :func:`src.imbalance.build_imbalance_pipeline` now
delegates here.  An earlier simple design placed the sampler after the model
preprocessor; that is safe for random over/under-sampling, but SMOTE and ADASYN
would then interpolate one-hot columns and create fractional category values.
This module provides the categorical-aware experiment path:

* the sampler sees imputed continuous features and integer-coded discrete
  features, never one-hot columns;
* SMOTE is implemented with :class:`imblearn.over_sampling.SMOTENC`, which
  keeps categorical values discrete;
* ADASYN has no native categorical implementation in imbalanced-learn, so
  :class:`CategoricalADASYN` performs ADASYN's adaptive neighbour weighting in
  a mixed feature space where discrete columns are one-hot encoded, then
  decodes each synthetic row to one valid category per discrete feature;
* decoding happens before the normal model preprocessor, so the classifier
  receives exactly the same feature representation as the baseline.

Leakage guarantees
------------------
Every step is a scikit-learn/imbalanced-learn transformer fitted inside the CV
training fold.  The sampler is only invoked during ``fit_resample``; the
imblearn pipeline skips sampler steps during ``predict``/``transform``.  The
validation fold is never resampled and the locked test set is never accepted by
these APIs.

SMOTENC investigation
---------------------
SMOTENC was selected because it supports mixed data directly and, unlike SMOTE
on one-hot data, returns values from the observed categorical levels.  Binary
clinical indicators are treated as categorical too, preventing fractional
values such as ``hypertension=0.37``.

ADASYN caveat
-------------
:class:`CategoricalADASYN` is a project-specific categorical-safe variant, not
the stock numeric-only ``ADASYN``.  It preserves ADASYN's adaptive sample
allocation but chooses each synthetic categorical value from the dominant
one-hot component of a valid neighbour pair.  Results should therefore be
reported as "categorical-aware ADASYN", not as an unmodified reference
implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    IMBALANCE_DEFAULT_K_NEIGHBORS,
    IMBALANCE_DEFAULT_SAMPLING_RATIO,
    IMBALANCE_RANDOM_STATE,
    NUMERIC_FEATURES,
)
from src.imbalance import (
    KIND_CLASS_WEIGHT,
    KIND_NONE,
    STRATEGIES,
    STRATEGY_ADASYN,
    STRATEGY_RANDOM_OVER_SAMPLER,
    STRATEGY_RANDOM_UNDER_SAMPLER,
    STRATEGY_SMOTE,
    ImbalanceStrategy,
    UnsupportedStrategyError,
    build_imbalance_pipeline,
    get_strategy,
    supports_strategy,
)
from src.models import BASELINE_MODEL_SPECS, build_estimator
from src.preprocessing import CATEGORY_VALUES, build_preprocessor

UNSEEN_CATEGORY = "__unseen_category__"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _resolved_groups(
    numeric_features: Sequence[str] | None,
    binary_features: Sequence[str] | None,
    categorical_features: Sequence[str] | None,
    category_values: Mapping[str, Sequence[Any]] | None,
) -> tuple[list[str], list[str], list[str], dict[str, list[Any]]]:
    numeric = list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    binary = list(BINARY_FEATURES if binary_features is None else binary_features)
    categorical = list(
        CATEGORICAL_FEATURES
        if categorical_features is None
        else categorical_features
    )
    values = {
        feature: list(levels)
        for feature, levels in (
            CATEGORY_VALUES if category_values is None else category_values
        ).items()
    }
    return numeric, binary, categorical, values


def class_counts(y: Sequence[Any]) -> dict[int, int]:
    """Return class counts keyed by integer label."""
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


# ---------------------------------------------------------------------------
# Sampling-side feature codec
# ---------------------------------------------------------------------------
class MixedFeatureEncoder(BaseEstimator, TransformerMixin):
    """Impute continuous features and integer-code discrete features.

    The output column order is ``numeric + binary + categorical``, matching
    :data:`src.config.FEATURE_COLUMNS`.  Categorical values are encoded with an
    explicit vocabulary so every fold uses the same codes.  Unknown validation
    categories become ``-1`` and are later mapped to a sentinel that the model
    one-hot encoder ignores.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] | None = None,
        binary_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        category_values: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        self.numeric_features = numeric_features
        self.binary_features = binary_features
        self.categorical_features = categorical_features
        self.category_values = category_values

    def _groups(self):
        return _resolved_groups(
            self.numeric_features,
            self.binary_features,
            self.categorical_features,
            self.category_values,
        )

    def output_feature_names(self) -> list[str]:
        """Return the encoded column names without requiring a fitted state."""
        numeric, binary, categorical, _ = self._groups()
        return numeric + binary + categorical

    def categorical_output_indices(self) -> np.ndarray:
        """Indices of binary/categorical columns in the encoded matrix."""
        numeric, binary, categorical, _ = self._groups()
        start = len(numeric)
        return np.arange(start, start + len(binary) + len(categorical), dtype=int)

    def _require_dataframe(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MixedFeatureEncoder expects a pandas DataFrame with named "
                "clinical features."
            )
        missing = [
            column
            for column in self.output_feature_names()
            if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MixedFeatureEncoder":
        self._require_dataframe(X)
        numeric, binary, categorical, values = self._groups()

        self.numeric_features_ = numeric
        self.binary_features_ = binary
        self.categorical_features_ = categorical
        self.category_values_ = values
        self.feature_columns_ = numeric + binary + categorical
        self.n_features_in_ = int(X.shape[1])

        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.numeric_imputer_.fit(X[numeric])

        discrete_columns = binary + categorical
        categories = [[0, 1] for _ in binary] + [
            values[column] for column in categorical
        ]
        self.discrete_encoder_ = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        self.discrete_encoder_.fit(X[discrete_columns])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "numeric_imputer_")
        self._require_dataframe(X)

        numeric = self.numeric_imputer_.transform(X[self.numeric_features_])
        discrete = self.discrete_encoder_.transform(
            X[self.binary_features_ + self.categorical_features_]
        )
        return np.column_stack([numeric, discrete]).astype(np.float64, copy=False)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_columns_")
        return np.asarray(self.feature_columns_, dtype=object)


class CategoricalCodeDecoder(BaseEstimator, TransformerMixin):
    """Map integer-coded discrete columns back to their original values.

    The decoder is deliberately strict: fractional categorical codes raise a
    ``ValueError`` instead of silently creating invalid categories or
    fractional one-hot columns downstream.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        numeric_features: Sequence[str] | None = None,
        binary_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        category_values: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.numeric_features = numeric_features
        self.binary_features = binary_features
        self.categorical_features = categorical_features
        self.category_values = category_values

    def _groups(self):
        columns = list(
            FEATURE_COLUMNS if self.feature_columns is None else self.feature_columns
        )
        numeric, binary, categorical, values = _resolved_groups(
            self.numeric_features,
            self.binary_features,
            self.categorical_features,
            self.category_values,
        )
        return columns, numeric, binary, categorical, values

    def fit(self, X: Any, y: Any = None) -> "CategoricalCodeDecoder":
        array = np.asarray(X)
        columns, numeric, binary, categorical, values = self._groups()
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError(
                f"Expected a 2-D array with {len(columns)} columns, got "
                f"shape {array.shape}."
            )
        self.feature_columns_ = columns
        self.numeric_features_ = numeric
        self.binary_features_ = binary
        self.categorical_features_ = categorical
        self.category_values_ = values
        self.n_features_in_ = len(columns)
        return self

    @staticmethod
    def _integer_codes(values: np.ndarray, column: str) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite categorical code in column {column!r}.")
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"Fractional categorical code in column {column!r}; synthetic "
                "sampling must preserve discrete categories."
            )
        return rounded.astype(int)

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected a 2-D array with {self.n_features_in_} columns, got "
                f"shape {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Sampled feature matrix contains NaN or infinity.")

        data: dict[str, Any] = {}
        for index, column in enumerate(self.feature_columns_):
            values = array[:, index]
            if column in self.numeric_features_:
                data[column] = values.astype(float)
            elif column in self.binary_features_:
                codes = self._integer_codes(values, column)
                if not np.isin(codes, [0, 1]).all():
                    raise ValueError(
                        f"Binary feature {column!r} has values outside {{0, 1}}."
                    )
                data[column] = codes.astype(int)
            else:
                codes = self._integer_codes(values, column)
                levels = self.category_values_[column]
                decoded: list[Any] = []
                for code in codes:
                    if code == -1:
                        decoded.append(UNSEEN_CATEGORY)
                    elif 0 <= code < len(levels):
                        decoded.append(levels[int(code)])
                    else:
                        raise ValueError(
                            f"Category code {int(code)} is outside the known "
                            f"range for {column!r} ({len(levels)} levels)."
                        )
                data[column] = decoded

        return pd.DataFrame(data, columns=self.feature_columns_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_columns_")
        return np.asarray(self.feature_columns_, dtype=object)


# ---------------------------------------------------------------------------
# Recording wrapper and categorical-aware ADASYN
# ---------------------------------------------------------------------------
class RecordingSampler(BaseEstimator):
    """Wrap a sampler and record actual before/after class counts.

    The wrapper is necessary because the imblearn pipeline does not expose the
    resampled target vector.  It delegates all sampling to the wrapped object,
    so the recorded counts describe the exact fold-level resampling that
    produced the fitted classifier.
    """

    def __init__(self, sampler: BaseSampler) -> None:
        self.sampler = sampler

    def fit(self, X: Any, y: Any) -> "RecordingSampler":
        self.sampler.fit(X, y)
        return self

    def fit_resample(self, X: Any, y: Any):
        self.sampling_class_counts_before_ = class_counts(y)
        self.n_samples_before_ = int(len(y))
        self.n_fit_resample_calls_ = int(
            getattr(self, "n_fit_resample_calls_", 0)
        ) + 1

        output = self.sampler.fit_resample(X, y)
        X_resampled, y_resampled = output[0], output[1]

        self.sampling_class_counts_after_ = class_counts(y_resampled)
        self.n_samples_after_ = int(len(y_resampled))
        return X_resampled, y_resampled

    def sampling_metadata(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the last resampling operation."""
        check_is_fitted(self, "sampling_class_counts_before_")
        return {
            "sampler": type(self.sampler).__name__,
            "sampler_random_state": getattr(self.sampler, "random_state", None),
            "sampling_strategy": getattr(self.sampler, "sampling_strategy", None),
            "sampling_strategy_": getattr(self.sampler, "sampling_strategy_", None),
            "n_samples_before": int(self.n_samples_before_),
            "n_samples_after": int(self.n_samples_after_),
            "class_counts_before": dict(self.sampling_class_counts_before_),
            "class_counts_after": dict(self.sampling_class_counts_after_),
            "n_fit_resample_calls": int(
                getattr(self, "n_fit_resample_calls_", 0)
            ),
        }


class CategoricalADASYN(BaseSampler):
    """Categorical-aware ADASYN for mixed continuous/discrete features.

    ADASYN's adaptive neighbour weights are computed on a mixed representation:
    continuous columns are standardized within each training fold and discrete
    columns are one-hot encoded (unit weight per one-hot column).
    Synthetic categorical values are then decoded from the dominant one-hot
    component of the nearest-neighbour interpolation.  The returned matrix uses
    integer codes for every discrete column, so no fractional category reaches
    the model preprocessor.
    """

    _sampling_type = "over-sampling"
    _parameter_constraints = {
        **ADASYN._parameter_constraints,
        "categorical_features": ["array-like"],
    }

    def __init__(
        self,
        categorical_features: Sequence[int],
        *,
        sampling_strategy: Any = "auto",
        random_state: int | None = None,
        n_neighbors: int = 5,
    ) -> None:
        super().__init__(sampling_strategy=sampling_strategy)
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def _fit_resample(self, X: Any, y: Any):
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("CategoricalADASYN expects a 2-D feature matrix.")
        categorical = np.asarray(self.categorical_features, dtype=int)
        if categorical.size == 0:
            raise ValueError("CategoricalADASYN requires categorical features.")
        if categorical.min() < 0 or categorical.max() >= array.shape[1]:
            raise ValueError("categorical_features contains an invalid index.")
        continuous = np.setdiff1d(np.arange(array.shape[1]), categorical)

        self.categorical_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=float,
        )
        encoded_categorical = self.categorical_encoder_.fit_transform(
            array[:, categorical]
        )
        # Fit scaling ONLY on the current training fold. This prevents raw-unit
        # variables such as glucose from overwhelming one-hot categorical
        # mismatches when ADASYN selects nearest neighbours. The synthetic
        # continuous features are restored to their original units below.
        self.continuous_scaler_ = StandardScaler()
        if len(continuous):
            scaled_continuous = self.continuous_scaler_.fit_transform(
                array[:, continuous]
            )
        else:
            scaled_continuous = np.empty((array.shape[0], 0), dtype=float)
        encoded = np.hstack([scaled_continuous, encoded_categorical])

        self.adasyn_ = ADASYN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )
        encoded_resampled, y_resampled = self.adasyn_.fit_resample(encoded, y)

        n_continuous = len(continuous)
        decoded_blocks: list[np.ndarray] = []
        offset = n_continuous
        for levels in self.categorical_encoder_.categories_:
            width = len(levels)
            block = encoded_resampled[:, offset : offset + width]
            indices = np.argmax(block, axis=1)
            # Map each one-hot index back to its actual ordinal code.  The
            # observed codes need not be contiguous (a category can be absent
            # from a particular training fold), so ``argmax`` alone is wrong.
            codes = np.asarray(levels, dtype=float)[indices]
            empty = np.isclose(block.sum(axis=1), 0.0, atol=1e-12)
            codes[empty] = -1.0
            decoded_blocks.append(codes)
            offset += width

        decoded_categorical = (
            np.column_stack(decoded_blocks)
            if decoded_blocks
            else np.empty((encoded_resampled.shape[0], 0))
        )

        resampled = np.empty(
            (encoded_resampled.shape[0], array.shape[1]), dtype=float
        )
        if n_continuous:
            resampled[:, continuous] = self.continuous_scaler_.inverse_transform(
                encoded_resampled[:, :n_continuous]
            )
            # Preserve the original observations byte-for-byte in numeric
            # value: inverse_transform can otherwise introduce floating noise.
            resampled[:len(array), continuous] = array[:, continuous]
        resampled[:, categorical] = decoded_categorical
        return resampled, y_resampled


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_leakage_safe_pipeline(
    model_name: str,
    strategy_name: str,
    *,
    sampling_strategy: Any | None = None,
    class_weight: str | Mapping[int, float] | None = None,
    random_state: int | None = None,
    k_neighbors: int | None = None,
    sampler_params: Mapping[str, Any] | None = None,
    record_sampling: bool = True,
) -> Pipeline:
    """Build one unfitted, fold-local model/strategy pipeline.

    For resampling strategies the pipeline is an
    :class:`imblearn.pipeline.Pipeline` with steps::

        sampling_encoder -> sampler -> sampling_decoder
        -> model_preprocessor -> classifier

    ``sampling_encoder`` imputes continuous features and integer-codes
    discrete features.  ``sampler`` sees only those encoded training rows.
    ``sampling_decoder`` restores valid categorical values, and
    ``model_preprocessor`` recreates the baseline feature representation.

    For ``none`` and class weighting, the ordinary two-step sklearn pipeline
    from :func:`src.imbalance.build_imbalance_pipeline` is returned.
    """
    strategy: ImbalanceStrategy = get_strategy(strategy_name)
    if model_name not in BASELINE_MODEL_SPECS:
        raise KeyError(
            f"Unknown model {model_name!r}; available: "
            f"{list(BASELINE_MODEL_SPECS)}"
        )
    if not supports_strategy(model_name, strategy_name):
        reason = strategy.exclusion_reason or "No compatible estimator API."
        raise UnsupportedStrategyError(
            f"Strategy {strategy_name!r} is not supported for model "
            f"{model_name!r}. {reason}"
        )

    if strategy.kind in (KIND_NONE, KIND_CLASS_WEIGHT):
        if sampler_params:
            raise ValueError(
                "sampler_params is only valid for resampling strategies."
            )
        if strategy.kind == KIND_NONE and class_weight is not None:
            raise ValueError(
                "class_weight is only valid for the class-weighting strategy."
            )
        return build_imbalance_pipeline(
            model_name,
            strategy_name,
            sampling_strategy=sampling_strategy,
            class_weight=class_weight,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    if class_weight is not None:
        raise ValueError(
            "class_weight cannot be combined with a resampling strategy in "
            "this experiment layer; keep balancing strategies separate."
        )

    estimator = build_estimator(model_name)
    family = BASELINE_MODEL_SPECS[model_name].family
    encoder = MixedFeatureEncoder()
    decoder = CategoricalCodeDecoder()
    categorical_indices = tuple(
        int(index) for index in encoder.categorical_output_indices()
    )

    ratio = (
        IMBALANCE_DEFAULT_SAMPLING_RATIO
        if sampling_strategy is None
        else sampling_strategy
    )
    seed = IMBALANCE_RANDOM_STATE if random_state is None else random_state
    neighbours = (
        IMBALANCE_DEFAULT_K_NEIGHBORS
        if k_neighbors is None
        else k_neighbors
    )

    sampler: BaseSampler
    if strategy.name == STRATEGY_RANDOM_OVER_SAMPLER:
        sampler = RandomOverSampler(
            sampling_strategy=ratio,
            random_state=seed,
        )
    elif strategy.name == STRATEGY_RANDOM_UNDER_SAMPLER:
        sampler = RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=seed,
        )
    elif strategy.name == STRATEGY_SMOTE:
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=ratio,
            random_state=seed,
            k_neighbors=neighbours,
        )
    elif strategy.name == STRATEGY_ADASYN:
        sampler = CategoricalADASYN(
            categorical_features=categorical_indices,
            sampling_strategy=ratio,
            random_state=seed,
            n_neighbors=neighbours,
        )
    else:
        raise ValueError(
            f"Unsupported resampling strategy {strategy.name!r}."
        )

    if sampler_params:
        if "categorical_features" in sampler_params:
            raise ValueError(
                "categorical_features is fixed by the pipeline's feature codec."
            )
        sampler.set_params(**sampler_params)

    sampler_step: BaseEstimator = (
        RecordingSampler(sampler) if record_sampling else sampler
    )

    return ImblearnPipeline(
        steps=[
            ("sampling_encoder", encoder),
            ("sampler", sampler_step),
            ("sampling_decoder", decoder),
            ("preprocessor", build_preprocessor(family)),
            ("classifier", estimator),
        ]
    )


def build_leakage_safe_grid(
    models: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    *,
    record_sampling: bool = True,
) -> OrderedDict[str, Pipeline]:
    """Build every supported model/strategy pipeline, keyed ``model__strategy``."""
    selected_models = tuple(
        BASELINE_MODEL_SPECS if models is None else models
    )
    selected_strategies = tuple(STRATEGIES if strategies is None else strategies)

    for model_name in selected_models:
        if model_name not in BASELINE_MODEL_SPECS:
            raise KeyError(
                f"Unknown model {model_name!r}; available: "
                f"{list(BASELINE_MODEL_SPECS)}"
            )
    for strategy_name in selected_strategies:
        get_strategy(strategy_name)

    grid: OrderedDict[str, Pipeline] = OrderedDict()
    for model_name in selected_models:
        for strategy_name in selected_strategies:
            if supports_strategy(model_name, strategy_name):
                grid[f"{model_name}__{strategy_name}"] = (
                    build_leakage_safe_pipeline(
                        model_name,
                        strategy_name,
                        record_sampling=record_sampling,
                    )
                )
    return grid


def clone_strategy_pipeline(pipeline: Pipeline) -> Pipeline:
    """Clone a pipeline before fitting it to a fold (explicit convenience)."""
    return clone(pipeline)


__all__ = [
    "CategoricalADASYN",
    "CategoricalCodeDecoder",
    "MixedFeatureEncoder",
    "RecordingSampler",
    "UNSEEN_CATEGORY",
    "build_leakage_safe_grid",
    "build_leakage_safe_pipeline",
    "class_counts",
    "clone_strategy_pipeline",
]
