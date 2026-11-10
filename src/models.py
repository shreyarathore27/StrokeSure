"""Documented baseline estimators and their preprocessing families.

Scope (Steps 18-20)
-------------------
Five baselines are defined: a prior-probability dummy, logistic regression, a
decision tree, a random forest and an RBF support vector machine.

Deliberate exclusions
---------------------
* No ``class_weight`` (neither ``"balanced"`` nor custom weights).
* No oversampling, undersampling, SMOTE or ADASYN.
* No threshold tuning and no probability calibration.

Class imbalance is handled later, as a separate experimental factor, so that
the baseline numbers reflect the natural 4.87% prevalence of the source data.

All estimators use ``random_state=42``.  Each call to
:func:`build_baseline_pipelines` constructs *fresh* estimator instances, so a
pipeline can never accidentally reuse a previously fitted object.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    MODEL_FAMILY_LINEAR_SVM,
    MODEL_FAMILY_TREE,
    RANDOM_STATE,
)
from src.preprocessing import build_model_pipeline

# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------
# Parameters are documented starting points for the unbalanced baseline. They
# are intentionally modest: this step establishes a reference, not a tuned
# result. Hyperparameter search belongs to a later step.


def make_dummy() -> DummyClassifier:
    """Predict the training class prior (majority-class baseline).

    ``strategy="prior"`` returns the training prevalence for both classes, so
    ROC-AUC must be 0.5 and average precision must approximate prevalence.
    """
    return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)


def make_logistic_regression() -> LogisticRegression:
    """L2-regularised logistic regression (unweighted, C=1.0).

    ``penalty`` is left at its default (L2) because scikit-learn 1.8+
    deprecates the explicit ``penalty="l2"`` argument in favour of ``C``.
    """
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def make_decision_tree() -> DecisionTreeClassifier:
    """Shallow CART tree to limit variance on a 4.87%-positive dataset."""
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )


def make_random_forest() -> RandomForestClassifier:
    """Bagged trees with sqrt feature sampling and a small leaf size."""
    return RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_svc() -> SVC:
    """RBF SVM with Platt-scaled probabilities.

    ``probability=True`` enables internal 5-fold Platt scaling, which costs
    extra time but is required for probability-based metrics (Brier score,
    ROC-AUC, average precision). ``random_state`` makes that step repeatable.
    """
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        cache_size=500,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class _ModelSpec:
    """Internal description of a baseline model."""

    __slots__ = ("name", "family", "factory", "rationale")

    def __init__(
        self,
        name: str,
        family: str,
        factory: Callable[[], BaseEstimator],
        rationale: str,
    ) -> None:
        self.name = name
        self.family = family
        self.factory = factory
        self.rationale = rationale


BASELINE_MODEL_SPECS: OrderedDict[str, _ModelSpec] = OrderedDict(
    [
        (
            "dummy_prior",
            _ModelSpec(
                name="dummy_prior",
                family=MODEL_FAMILY_TREE,
                factory=make_dummy,
                rationale=(
                    "Class-prior reference; establishes the floor that any "
                    "learned model must beat on ROC-AUC and average precision."
                ),
            ),
        ),
        (
            "logistic_regression",
            _ModelSpec(
                name="logistic_regression",
                family=MODEL_FAMILY_LINEAR_SVM,
                factory=make_logistic_regression,
                rationale=(
                    "Linear baseline with scaled features and drop-first "
                    "one-hot encoding, as specified in Notebook 02."
                ),
            ),
        ),
        (
            "decision_tree",
            _ModelSpec(
                name="decision_tree",
                family=MODEL_FAMILY_TREE,
                factory=make_decision_tree,
                rationale=(
                    "Non-linear, interpretable baseline on unscaled features."
                ),
            ),
        ),
        (
            "random_forest",
            _ModelSpec(
                name="random_forest",
                family=MODEL_FAMILY_TREE,
                factory=make_random_forest,
                rationale=(
                    "Variance-reduced tree ensemble; the usual strong tabular "
                    "baseline on unscaled features."
                ),
            ),
        ),
        (
            "svc_rbf",
            _ModelSpec(
                name="svc_rbf",
                family=MODEL_FAMILY_LINEAR_SVM,
                factory=make_svc,
                rationale=(
                    "Margin-based non-linear baseline; probabilities come from "
                    "internal Platt scaling so probability metrics are valid."
                ),
            ),
        ),
    ]
)


def model_names() -> list[str]:
    """Return baseline model names in evaluation order."""
    return list(BASELINE_MODEL_SPECS)


def model_family(name: str) -> str:
    """Return the preprocessing family for a model name."""
    try:
        return BASELINE_MODEL_SPECS[name].family
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}; available: {model_names()}"
        ) from exc


def build_estimator(name: str) -> BaseEstimator:
    """Create a fresh, unfitted estimator for the named baseline."""
    if name not in BASELINE_MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}; available: {model_names()}")
    return BASELINE_MODEL_SPECS[name].factory()


def build_baseline_pipelines() -> OrderedDict[str, Pipeline]:
    """Build one unfitted ``Pipeline`` per baseline, keyed by model name."""
    pipelines: OrderedDict[str, Pipeline] = OrderedDict()
    for name, spec in BASELINE_MODEL_SPECS.items():
        pipelines[name] = build_model_pipeline(spec.factory(), spec.family)
    return pipelines


def model_parameter_table() -> list[dict[str, Any]]:
    """Return a serialisable description of each baseline configuration."""
    rows: list[dict[str, Any]] = []
    for name, spec in BASELINE_MODEL_SPECS.items():
        estimator = spec.factory()
        params = estimator.get_params(deep=False)
        rows.append(
            {
                "model": name,
                "classifier": type(estimator).__name__,
                "preprocessing_family": spec.family,
                "parameters": ", ".join(
                    f"{key}={value!r}" for key, value in sorted(params.items())
                ),
                "class_weight": str(params.get("class_weight", "not applicable")),
                "rationale": spec.rationale,
            }
        )
    return rows
