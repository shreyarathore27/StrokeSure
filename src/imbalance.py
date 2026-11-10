"""Reusable class-imbalance strategies for the StrokeSure experiment.

This module defines the *strategy layer* only.  It does not run experiments,
tune thresholds, calibrate probabilities, or touch the held-out test set.  The
existing baseline pipelines and results remain unchanged.

Strategies
----------
``none``
    No balancing.  The model sees the natural training prevalence.
``class_weight_balanced``
    ``class_weight="balanced"`` on estimators that expose the parameter.
``random_over_sampler``
    Random duplication of minority rows.
``random_under_sampler``
    Random removal of majority rows.
``smote``
    Synthetic minority interpolation.
``adasyn``
    Adaptive synthetic minority generation.

Support matrix
--------------
==========================  =====  ====  ====  ====  ====
Strategy                    Dummy  LR    DT    RF    SVC
==========================  =====  ====  ====  ====  ====
none                          yes  yes   yes   yes   yes
class_weight_balanced          no  yes   yes   yes   yes
random_over_sampler            no  yes   yes   yes   yes
random_under_sampler           no  yes   yes   yes   yes
smote                          no  yes   yes   yes   yes
adasyn                         no  yes   yes   yes   yes
==========================  =====  ====  ====  ====  ====

Deliberate exclusions
---------------------
``dummy_prior`` is excluded from class weighting and from every resampling
strategy.  ``DummyClassifier`` has no ``class_weight`` parameter, and changing
the class distribution before fitting would destroy its role as a
natural-prevalence no-skill reference.  It is therefore only valid with
``none``.

No learned model is excluded.  SMOTE and ADASYN are supported through the
categorical-aware experiment layer in :mod:`src.imbalance_pipelines`, so
synthetic samples never contain fractional one-hot category values.  ``SVC``
with resampling is supported but comparatively expensive because its Platt
calibration performs an internal cross-validation for every fit.

Implementation notes
--------------------
* imbalanced-learn is imported lazily so that the dependency-free strategies
  remain usable if the package is absent.
* Resampling is placed inside an :class:`imblearn.pipeline.Pipeline` and is
  fitted only on the training portion of each cross-validation fold.  The
  categorical-aware path is implemented in :mod:`src.imbalance_pipelines`.
* The default sampling ratio (0.5) and neighbour count (5) are starting points
  recorded in :mod:`src.config`; they are not tuned values.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sklearn.pipeline import Pipeline

from src.config import (
    IMBALANCE_DEFAULT_K_NEIGHBORS,
    IMBALANCE_DEFAULT_SAMPLING_RATIO,
    IMBALANCE_RANDOM_STATE,
)
from src.models import BASELINE_MODEL_SPECS, build_estimator, model_names
from src.preprocessing import build_model_pipeline

# ---------------------------------------------------------------------------
# Strategy identifiers
# ---------------------------------------------------------------------------
STRATEGY_NONE = "none"
STRATEGY_CLASS_WEIGHT_BALANCED = "class_weight_balanced"
STRATEGY_RANDOM_OVER_SAMPLER = "random_over_sampler"
STRATEGY_RANDOM_UNDER_SAMPLER = "random_under_sampler"
STRATEGY_SMOTE = "smote"
STRATEGY_ADASYN = "adasyn"

KIND_NONE = "none"
KIND_CLASS_WEIGHT = "class_weight"
KIND_RESAMPLER = "resampler"


class ImbalanceDependencyError(RuntimeError):
    """Raised when a sampler strategy is requested without imbalanced-learn."""


class UnsupportedStrategyError(ValueError):
    """Raised when a strategy is combined with an incompatible model."""


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImbalanceStrategy:
    """Description of one class-imbalance strategy.

    Parameters
    ----------
    name
        Stable identifier used in experiment manifests and output filenames.
    kind
        ``"none"``, ``"class_weight"`` or ``"resampler"``.
    description
        Short human-readable explanation.
    supported_models
        Model names from :data:`src.models.BASELINE_MODEL_SPECS` that may use
        this strategy.
    excluded_models
        Model names that are deliberately unsupported.
    exclusion_reason
        Why the excluded models are unsupported.
    requires_imbalanced_learn
        Whether the strategy needs the optional imbalanced-learn dependency.
    default_parameters
        Parameters passed to the estimator or sampler unless overridden.
    notes
        Scientific or practical caveats.
    """

    name: str
    kind: str
    description: str
    supported_models: tuple[str, ...]
    excluded_models: tuple[str, ...]
    exclusion_reason: str
    requires_imbalanced_learn: bool
    default_parameters: Mapping[str, Any]
    notes: str = ""


_ALL_MODELS = tuple(model_names())
_DUMMY_MODELS = ("dummy_prior",)
_LEARNED_MODELS = tuple(
    name for name in _ALL_MODELS if name not in _DUMMY_MODELS
)

_DUMMY_EXCLUSION = (
    "DummyClassifier has no class_weight parameter, and resampling before "
    "fitting would change its natural-prevalence no-skill reference. The "
    "dummy is therefore meaningful only with the 'none' strategy."
)

STRATEGIES: OrderedDict[str, ImbalanceStrategy] = OrderedDict(
    [
        (
            STRATEGY_NONE,
            ImbalanceStrategy(
                name=STRATEGY_NONE,
                kind=KIND_NONE,
                description=(
                    "Train on the natural class distribution; no weights and "
                    "no resampling."
                ),
                supported_models=_ALL_MODELS,
                excluded_models=(),
                exclusion_reason="",
                requires_imbalanced_learn=False,
                default_parameters={},
                notes=(
                    "Reference strategy. It reproduces the unweighted baseline "
                    "distribution without changing estimator hyperparameters."
                ),
            ),
        ),
        (
            STRATEGY_CLASS_WEIGHT_BALANCED,
            ImbalanceStrategy(
                name=STRATEGY_CLASS_WEIGHT_BALANCED,
                kind=KIND_CLASS_WEIGHT,
                description=(
                    "Set class_weight='balanced' so the loss gives the minority "
                    "class proportionally more weight."
                ),
                supported_models=_LEARNED_MODELS,
                excluded_models=_DUMMY_MODELS,
                exclusion_reason=_DUMMY_EXCLUSION,
                requires_imbalanced_learn=False,
                default_parameters={"class_weight": "balanced"},
                notes=(
                    "For the current 4,088-row training partition with 199 "
                    "positives, 'balanced' implies class-1 weight ~10.27 and "
                    "class-0 weight ~0.526 (about 19.5:1). It changes the loss, "
                    "not the observed rows, so no synthetic data are created."
                ),
            ),
        ),
        (
            STRATEGY_RANDOM_OVER_SAMPLER,
            ImbalanceStrategy(
                name=STRATEGY_RANDOM_OVER_SAMPLER,
                kind=KIND_RESAMPLER,
                description=(
                    "Randomly duplicate minority rows until the requested "
                    "minority:majority ratio is reached."
                ),
                supported_models=_LEARNED_MODELS,
                excluded_models=_DUMMY_MODELS,
                exclusion_reason=_DUMMY_EXCLUSION,
                requires_imbalanced_learn=True,
                default_parameters={
                    "sampling_strategy": IMBALANCE_DEFAULT_SAMPLING_RATIO,
                    "random_state": IMBALANCE_RANDOM_STATE,
                },
                notes=(
                    "Duplicates existing minority rows and therefore adds no "
                    "new information; it mainly changes the effective loss. "
                    "Sampling runs inside each CV training fold only."
                ),
            ),
        ),
        (
            STRATEGY_RANDOM_UNDER_SAMPLER,
            ImbalanceStrategy(
                name=STRATEGY_RANDOM_UNDER_SAMPLER,
                kind=KIND_RESAMPLER,
                description=(
                    "Randomly remove majority rows until the requested "
                    "minority:majority ratio is reached."
                ),
                supported_models=_LEARNED_MODELS,
                excluded_models=_DUMMY_MODELS,
                exclusion_reason=_DUMMY_EXCLUSION,
                requires_imbalanced_learn=True,
                default_parameters={
                    "sampling_strategy": IMBALANCE_DEFAULT_SAMPLING_RATIO,
                    "random_state": IMBALANCE_RANDOM_STATE,
                },
                notes=(
                    "Discards majority information and can substantially reduce "
                    "the training set. On the current 4,088-row training "
                    "partition, ratio 0.5 leaves about 398 majority rows plus "
                    "199 minority rows; ratio 1.0 leaves only the 199 minority "
                    "rows. Moderate ratios are preferable."
                ),
            ),
        ),
        (
            STRATEGY_SMOTE,
            ImbalanceStrategy(
                name=STRATEGY_SMOTE,
                kind=KIND_RESAMPLER,
                description=(
                    "Create synthetic minority rows by interpolating between "
                    "near neighbours."
                ),
                supported_models=_LEARNED_MODELS,
                excluded_models=_DUMMY_MODELS,
                exclusion_reason=_DUMMY_EXCLUSION,
                requires_imbalanced_learn=True,
                default_parameters={
                    "sampling_strategy": IMBALANCE_DEFAULT_SAMPLING_RATIO,
                    "random_state": IMBALANCE_RANDOM_STATE,
                    "k_neighbors": IMBALANCE_DEFAULT_K_NEIGHBORS,
                },
                notes=(
                    "Built with SMOTENC in the leakage-safe experiment layer: "
                    "categorical and binary columns stay discrete, so no "
                    "fractional one-hot values are produced. Requires at least "
                    "k_neighbors+1 minority rows in each training fold."
                ),
            ),
        ),
        (
            STRATEGY_ADASYN,
            ImbalanceStrategy(
                name=STRATEGY_ADASYN,
                kind=KIND_RESAMPLER,
                description=(
                    "Adaptively generate synthetic minority rows, focusing on "
                    "minority examples that are harder to classify."
                ),
                supported_models=_LEARNED_MODELS,
                excluded_models=_DUMMY_MODELS,
                exclusion_reason=_DUMMY_EXCLUSION,
                requires_imbalanced_learn=True,
                default_parameters={
                    "sampling_strategy": IMBALANCE_DEFAULT_SAMPLING_RATIO,
                    "random_state": IMBALANCE_RANDOM_STATE,
                    "n_neighbors": IMBALANCE_DEFAULT_K_NEIGHBORS,
                },
                notes=(
                    "Uses the project-specific categorical-aware ADASYN variant "
                    "from :mod:`src.imbalance_pipelines`: adaptive neighbour "
                    "weights are computed in a mixed one-hot space and synthetic "
                    "categories are decoded to valid levels. With only 199 "
                    "positives, adaptive weights can amplify label noise; "
                    "interpret any gain cautiously."
                ),
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------
def imbalanced_learn_version() -> str | None:
    """Return the installed imbalanced-learn version, or ``None`` if absent."""
    try:
        module = importlib.import_module("imblearn")
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def require_imbalanced_learn() -> str:
    """Return the imbalanced-learn version, raising a clear error if absent."""
    version = imbalanced_learn_version()
    if version is None:
        raise ImbalanceDependencyError(
            "imbalanced-learn is required for RandomOverSampler, "
            "RandomUnderSampler, SMOTE and ADASYN. Install the pinned "
            "compatible release (imbalanced-learn==0.14.2) before using "
            "these strategies."
        )
    return version


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def strategy_names() -> list[str]:
    """Return strategy names in their documented order."""
    return list(STRATEGIES)


def get_strategy(name: str) -> ImbalanceStrategy:
    """Return the strategy definition for ``name``."""
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown imbalance strategy {name!r}; available: {strategy_names()}"
        ) from exc


def supports_strategy(model_name: str, strategy_name: str) -> bool:
    """Whether ``strategy_name`` is valid for ``model_name``."""
    if model_name not in BASELINE_MODEL_SPECS:
        raise KeyError(
            f"Unknown model {model_name!r}; available: {list(_ALL_MODELS)}"
        )
    return model_name in get_strategy(strategy_name).supported_models


def _validate_supported(model_name: str, strategy: ImbalanceStrategy) -> None:
    if model_name not in BASELINE_MODEL_SPECS:
        raise KeyError(
            f"Unknown model {model_name!r}; available: {list(_ALL_MODELS)}"
        )
    if model_name not in strategy.supported_models:
        reason = strategy.exclusion_reason or "No compatible estimator API."
        raise UnsupportedStrategyError(
            f"Strategy {strategy.name!r} is not supported for model "
            f"{model_name!r}. {reason}"
        )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_imbalance_pipeline(
    model_name: str,
    strategy_name: str,
    *,
    sampling_strategy: Any | None = None,
    class_weight: str | Mapping[int, float] | None = None,
    random_state: int | None = None,
    k_neighbors: int | None = None,
    sampler_params: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build an unfitted pipeline for one model/strategy combination.

    Parameters
    ----------
    model_name
        Key in :data:`src.models.BASELINE_MODEL_SPECS`.
    strategy_name
        Key in :data:`STRATEGIES`.
    sampling_strategy
        Optional override for the sampler's ``sampling_strategy``.  Semantics
        are imbalanced-learn's; the project default is a 0.5 target ratio.
    class_weight
        Optional override for the class-weighting strategy.  ``None`` keeps the
        strategy default (``"balanced"``); pass a mapping for custom weights.
    random_state
        Optional sampler seed override.
    k_neighbors
        Optional neighbour-count override for SMOTE (``k_neighbors``) or
        ADASYN (``n_neighbors``).
    sampler_params
        Optional extra sampler keyword arguments.  These override generated
        parameters, so use the sampler's own argument names.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A two-step sklearn pipeline for ``none``/class weighting.  Resampling
        strategies return the categorical-aware five-step
        :class:`imblearn.pipeline.Pipeline` built by
        :func:`src.imbalance_pipelines.build_leakage_safe_pipeline`
        (sampling encoder -> sampler -> decoder -> preprocessor -> classifier).
        All objects are fresh and unfitted.
    """
    strategy = get_strategy(strategy_name)
    _validate_supported(model_name, strategy)

    estimator = build_estimator(model_name)
    family = BASELINE_MODEL_SPECS[model_name].family

    if strategy.kind == KIND_NONE:
        return build_model_pipeline(estimator, family)

    if strategy.kind == KIND_CLASS_WEIGHT:
        if "class_weight" not in estimator.get_params(deep=False):
            raise UnsupportedStrategyError(
                f"Estimator for model {model_name!r} does not expose "
                "class_weight."
            )
        weight = (
            strategy.default_parameters["class_weight"]
            if class_weight is None
            else class_weight
        )
        estimator.set_params(class_weight=weight)
        return build_model_pipeline(estimator, family)

    if strategy.kind == KIND_RESAMPLER:
        ratio = (
            strategy.default_parameters["sampling_strategy"]
            if sampling_strategy is None
            else sampling_strategy
        )
        seed = (
            strategy.default_parameters["random_state"]
            if random_state is None
            else random_state
        )
        neighbours = (
            IMBALANCE_DEFAULT_K_NEIGHBORS
            if k_neighbors is None
            else k_neighbors
        )

        # Lazy import avoids a module cycle: the experiment layer imports this
        # registry for its strategy definitions.
        require_imbalanced_learn()
        from src.imbalance_pipelines import build_leakage_safe_pipeline

        return build_leakage_safe_pipeline(
            model_name,
            strategy.name,
            sampling_strategy=ratio,
            random_state=seed,
            k_neighbors=neighbours,
            sampler_params=sampler_params,
            record_sampling=False,
        )

    raise ValueError(
        f"Unsupported strategy kind {strategy.kind!r} for {strategy.name!r}."
    )


def build_strategy_grid(
    models: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
) -> OrderedDict[str, Pipeline]:
    """Build every supported model/strategy pipeline, keyed as ``model__strategy``.

    Unsupported combinations (for example, the dummy with resampling) are
    skipped rather than constructed.  Building a pipeline does not fit it.
    For experiments that must audit before/after sampling counts, use
    :func:`src.imbalance_pipelines.build_leakage_safe_grid` instead; it wraps
    samplers with :class:`src.imbalance_pipelines.RecordingSampler`.
    """
    selected_models = tuple(_ALL_MODELS if models is None else models)
    selected_strategies = tuple(
        STRATEGIES if strategies is None else strategies
    )

    for model_name in selected_models:
        if model_name not in BASELINE_MODEL_SPECS:
            raise KeyError(
                f"Unknown model {model_name!r}; available: {list(_ALL_MODELS)}"
            )
    for strategy_name in selected_strategies:
        get_strategy(strategy_name)

    grid: OrderedDict[str, Pipeline] = OrderedDict()
    for model_name in selected_models:
        for strategy_name in selected_strategies:
            if supports_strategy(model_name, strategy_name):
                key = f"{model_name}__{strategy_name}"
                grid[key] = build_imbalance_pipeline(
                    model_name, strategy_name
                )
    return grid


# ---------------------------------------------------------------------------
# Documentation helpers
# ---------------------------------------------------------------------------
def strategy_table() -> list[dict[str, Any]]:
    """Return a serialisable description of every strategy and its exclusions."""
    rows: list[dict[str, Any]] = []
    for spec in STRATEGIES.values():
        rows.append(
            {
                "strategy": spec.name,
                "kind": spec.kind,
                "description": spec.description,
                "requires_imbalanced_learn": spec.requires_imbalanced_learn,
                "supported_models": ", ".join(spec.supported_models),
                "excluded_models": ", ".join(spec.excluded_models) or "none",
                "exclusion_reason": spec.exclusion_reason or "none",
                "default_parameters": ", ".join(
                    f"{key}={value!r}"
                    for key, value in sorted(spec.default_parameters.items())
                ),
                "notes": spec.notes,
            }
        )
    return rows


def support_matrix() -> list[dict[str, Any]]:
    """Return one row per strategy/model pair with a boolean support flag."""
    return [
        {
            "strategy": strategy_name,
            "model": model_name,
            "supported": supports_strategy(model_name, strategy_name),
        }
        for strategy_name in STRATEGIES
        for model_name in _ALL_MODELS
    ]


__all__ = [
    "ImbalanceDependencyError",
    "ImbalanceStrategy",
    "KIND_CLASS_WEIGHT",
    "KIND_NONE",
    "KIND_RESAMPLER",
    "STRATEGIES",
    "STRATEGY_ADASYN",
    "STRATEGY_CLASS_WEIGHT_BALANCED",
    "STRATEGY_NONE",
    "STRATEGY_RANDOM_OVER_SAMPLER",
    "STRATEGY_RANDOM_UNDER_SAMPLER",
    "STRATEGY_SMOTE",
    "UnsupportedStrategyError",
    "build_imbalance_pipeline",
    "build_strategy_grid",
    "get_strategy",
    "imbalanced_learn_version",
    "require_imbalanced_learn",
    "strategy_names",
    "strategy_table",
    "support_matrix",
    "supports_strategy",
]
