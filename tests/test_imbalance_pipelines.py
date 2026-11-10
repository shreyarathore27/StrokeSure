"""Tests for leakage-safe imbalance pipelines and the fold-level runner.

Run from the project root with::

    python -m unittest discover -s tests -v

The tests use small synthetic data; they never fit the full StrokeSure dataset
and never touch the held-out test partition.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURE_COUNTS,
    FEATURE_COLUMNS,
    MODEL_FAMILY_LINEAR_SVM,
    MODEL_FAMILY_TREE,
    NUMERIC_FEATURES,
)
from src.imbalance import build_imbalance_pipeline
from src.imbalance_experiment import (
    run_imbalance_cv,
    validate_shared_fold_plan,
)
from src.imbalance_pipelines import (
    CategoricalADASYN,
    CategoricalCodeDecoder,
    MixedFeatureEncoder,
    RecordingSampler,
    build_leakage_safe_grid,
    build_leakage_safe_pipeline,
    class_counts,
)
from src.preflight import build_fold_plan
from src.preprocessing import CATEGORY_VALUES


def make_mixed_data(
    n_rows: int = 240,
    n_positive: int = 48,
    random_state: int = 42,
    missing_bmi: int = 6,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic mixed-type data with all configured categories."""
    rng = np.random.default_rng(random_state)
    y = np.zeros(n_rows, dtype=int)
    y[:n_positive] = 1
    rng.shuffle(y)

    data: dict[str, object] = {
        "age": rng.normal(52.0, 12.0, n_rows),
        "avg_glucose_level": rng.normal(105.0, 28.0, n_rows),
        "bmi": rng.normal(28.0, 5.0, n_rows),
        "hypertension": rng.integers(0, 2, n_rows),
        "heart_disease": rng.integers(0, 2, n_rows),
    }
    for column, levels in CATEGORY_VALUES.items():
        data[column] = [levels[index % len(levels)] for index in range(n_rows)]

    X = pd.DataFrame(data)
    X.loc[X.index[:missing_bmi], "bmi"] = np.nan
    return X, pd.Series(y, name="stroke")


class ImbalancePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, cls.y = make_mixed_data()
        cls.plan = build_fold_plan(
            cls.X,
            cls.y,
            n_splits=3,
            n_repeats=2,
            random_state=42,
        )

    def first_fold(self):
        spec = self.plan[0]
        return (
            self.X.iloc[spec.train_positions],
            self.X.iloc[spec.valid_positions],
            self.y.iloc[spec.train_positions],
            self.y.iloc[spec.valid_positions],
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def test_resampling_pipelines_use_imblearn_and_categorical_samplers(self):
        expected = {
            "random_over_sampler": RandomOverSampler,
            "random_under_sampler": RandomUnderSampler,
            "smote": SMOTENC,
            "adasyn": CategoricalADASYN,
        }
        expected_indices = tuple(
            range(len(NUMERIC_FEATURES), len(FEATURE_COLUMNS))
        )
        for strategy, sampler_class in expected.items():
            with self.subTest(strategy=strategy):
                pipeline = build_leakage_safe_pipeline(
                    "logistic_regression", strategy
                )
                self.assertIsInstance(pipeline, ImblearnPipeline)
                self.assertEqual(
                    list(pipeline.named_steps),
                    [
                        "sampling_encoder",
                        "sampler",
                        "sampling_decoder",
                        "preprocessor",
                        "classifier",
                    ],
                )
                sampler = pipeline.named_steps["sampler"]
                self.assertIsInstance(sampler, RecordingSampler)
                self.assertIsInstance(sampler.sampler, sampler_class)
                if strategy in {"smote", "adasyn"}:
                    self.assertEqual(
                        tuple(sampler.sampler.categorical_features),
                        expected_indices,
                    )

    def test_legacy_builder_now_uses_categorical_aware_pipeline(self):
        pipeline = build_imbalance_pipeline(
            "logistic_regression", "smote"
        )
        self.assertEqual(
            list(pipeline.named_steps),
            [
                "sampling_encoder",
                "sampler",
                "sampling_decoder",
                "preprocessor",
                "classifier",
            ],
        )
        self.assertIsInstance(
            pipeline.named_steps["sampler"], SMOTENC
        )

    def test_none_and_class_weight_remain_two_step_pipelines(self):
        none_pipeline = build_leakage_safe_pipeline(
            "logistic_regression", "none"
        )
        self.assertIsInstance(none_pipeline, Pipeline)
        self.assertEqual(
            list(none_pipeline.named_steps), ["preprocessor", "classifier"]
        )
        self.assertNotIn("sampler", none_pipeline.named_steps)

        weighted = build_leakage_safe_pipeline(
            "logistic_regression", "class_weight_balanced"
        )
        self.assertEqual(
            weighted.named_steps["classifier"].class_weight, "balanced"
        )

    def test_grid_support_and_dummy_exclusion(self):
        grid = build_leakage_safe_grid()
        self.assertEqual(len(grid), 25)
        for pipeline in grid.values():
            clone(pipeline)
        self.assertIn("dummy_prior__none", grid)
        self.assertFalse(
            any(
                key.startswith("dummy_prior__")
                and not key.endswith("__none")
                for key in grid
            )
        )
        with self.assertRaises(ValueError):
            build_leakage_safe_pipeline("dummy_prior", "smote")

    # ------------------------------------------------------------------
    # Fold-local fitting and validation isolation
    # ------------------------------------------------------------------
    def test_fit_is_fold_local_and_prediction_never_resamples(self):
        X_train, X_validation, y_train, y_validation = self.first_fold()
        pipeline = build_leakage_safe_pipeline(
            "logistic_regression", "smote"
        )
        pipeline.fit(X_train, y_train)

        sampler = pipeline.named_steps["sampler"]
        self.assertEqual(sampler.n_samples_before_, len(y_train))
        self.assertEqual(
            sampler.sampling_class_counts_before_,
            class_counts(y_train),
        )
        self.assertEqual(sampler.n_fit_resample_calls_, 1)
        self.assertGreater(sampler.n_samples_after_, len(y_train))

        before = dict(sampler.sampling_class_counts_before_)
        after = dict(sampler.sampling_class_counts_after_)
        calls = sampler.n_fit_resample_calls_
        validation_before = class_counts(y_validation)

        probabilities = pipeline.predict_proba(X_validation)
        self.assertEqual(probabilities.shape[0], len(y_validation))

        self.assertEqual(sampler.n_fit_resample_calls_, calls)
        self.assertEqual(sampler.sampling_class_counts_before_, before)
        self.assertEqual(sampler.sampling_class_counts_after_, after)
        self.assertEqual(class_counts(y_validation), validation_before)
        self.assertEqual(sampler.n_samples_before_, len(y_train))
        self.assertNotEqual(
            sampler.n_samples_before_, len(y_train) + len(y_validation)
        )

    def test_preprocessing_is_fit_on_training_fold_only(self):
        X_train, X_validation, y_train, _ = self.first_fold()
        pipeline = build_leakage_safe_pipeline(
            "decision_tree", "smote"
        )
        pipeline.fit(X_train, y_train)

        encoder: MixedFeatureEncoder = pipeline.named_steps[
            "sampling_encoder"
        ]
        expected_medians = np.nanmedian(
            X_train[NUMERIC_FEATURES].to_numpy(), axis=0
        )
        np.testing.assert_allclose(
            encoder.numeric_imputer_.statistics_, expected_medians
        )

        expected_categories = [[0, 1] for _ in BINARY_FEATURES] + [
            list(CATEGORY_VALUES[column]) for column in CATEGORICAL_FEATURES
        ]
        for actual, expected in zip(
            encoder.discrete_encoder_.categories_, expected_categories
        ):
            self.assertEqual(list(actual), list(expected))

        shifted = X_validation.copy()
        shifted.iloc[0, shifted.columns.get_loc("gender")] = "Unknown"
        probabilities = pipeline.predict_proba(shifted)
        self.assertEqual(probabilities.shape[0], len(shifted))

        # The unseen validation category did not alter the fitted vocabulary.
        for actual, expected in zip(
            encoder.discrete_encoder_.categories_, expected_categories
        ):
            self.assertEqual(list(actual), list(expected))

    # ------------------------------------------------------------------
    # Categorical integrity
    # ------------------------------------------------------------------
    def test_synthetic_sampling_keeps_categories_discrete(self):
        X_train, _, y_train, _ = self.first_fold()
        for strategy in ("smote", "adasyn"):
            with self.subTest(strategy=strategy):
                pipeline = build_leakage_safe_pipeline(
                    "decision_tree", strategy
                )
                pipeline.fit(X_train, y_train)

                encoder: MixedFeatureEncoder = pipeline.named_steps[
                    "sampling_encoder"
                ]
                decoder: CategoricalCodeDecoder = pipeline.named_steps[
                    "sampling_decoder"
                ]
                sampler = pipeline.named_steps["sampler"].sampler

                encoded = encoder.transform(X_train)
                resampled, _ = sampler.fit_resample(encoded, y_train)
                decoded = decoder.transform(resampled)

                for column in BINARY_FEATURES:
                    self.assertTrue(
                        set(decoded[column].unique()).issubset({0, 1}),
                        msg=f"{strategy}: fractional binary {column}",
                    )
                for column, levels in CATEGORY_VALUES.items():
                    self.assertTrue(
                        set(decoded[column].unique()).issubset(set(levels)),
                        msg=f"{strategy}: invalid category in {column}",
                    )
                self.assertTrue(
                    np.isfinite(
                        decoded[NUMERIC_FEATURES].to_numpy()
                    ).all()
                )

                preprocessor = pipeline.named_steps["preprocessor"]
                model_matrix = preprocessor.transform(decoded)
                feature_names = [
                    str(name)
                    for name in preprocessor.get_feature_names_out()
                ]
                categorical_columns = [
                    index
                    for index, name in enumerate(feature_names)
                    if name.startswith("cat__")
                ]
                values = model_matrix[:, categorical_columns]
                self.assertTrue(
                    np.all((values == 0.0) | (values == 1.0)),
                    msg=f"{strategy}: fractional one-hot category values",
                )
                self.assertFalse(
                    np.any((values > 1e-9) & (values < 1.0 - 1e-9))
                )

                # Tree-family one-hot keeps every category, so each row must
                # activate exactly one level per categorical feature.
                for feature in CATEGORICAL_FEATURES:
                    columns = [
                        index
                        for index, name in enumerate(feature_names)
                        if name.startswith(f"cat__{feature}_")
                    ]
                    self.assertGreater(len(columns), 0)
                    np.testing.assert_allclose(
                        model_matrix[:, columns].sum(axis=1), 1.0
                    )

    def test_decoder_rejects_fractional_discrete_codes(self):
        decoder = CategoricalCodeDecoder()
        decoder.fit(np.zeros((2, len(FEATURE_COLUMNS))))

        fractional = np.zeros((1, len(FEATURE_COLUMNS)))
        fractional[
            0, FEATURE_COLUMNS.index("gender")
        ] = 1.5
        with self.assertRaises(ValueError):
            decoder.transform(fractional)

        fractional_binary = np.zeros((1, len(FEATURE_COLUMNS)))
        fractional_binary[
            0, FEATURE_COLUMNS.index("hypertension")
        ] = 0.25
        with self.assertRaises(ValueError):
            decoder.transform(fractional_binary)

        valid = np.zeros((1, len(FEATURE_COLUMNS)))
        valid[0, FEATURE_COLUMNS.index("gender")] = 2
        decoded = decoder.transform(valid)
        self.assertEqual(decoded.loc[0, "gender"], "Other")

    def test_categorical_adasyn_preserves_non_contiguous_codes(self):
        rng = np.random.default_rng(123)
        n_rows = 180
        n_positive = 45
        y = np.array([1] * n_positive + [0] * (n_rows - n_positive))
        rng.shuffle(y)
        continuous = rng.normal(size=n_rows)
        # Only ordinal codes 0 and 2 are observed; code 1 is absent.
        categorical = np.where(np.arange(n_rows) % 2 == 0, 0.0, 2.0)
        encoded = np.column_stack([continuous, categorical])

        sampler = CategoricalADASYN(
            categorical_features=(1,),
            sampling_strategy=0.5,
            random_state=42,
            n_neighbors=5,
        )
        resampled, resampled_y = sampler.fit_resample(encoded, y)

        self.assertGreater(len(resampled_y), len(y))
        self.assertTrue(
            np.isin(resampled[:, 1], [0.0, 2.0]).all(),
            msg="ADASYN emitted an ordinal code absent from the training fold",
        )

    # ------------------------------------------------------------------
    # Shared folds, seeds and counts
    # ------------------------------------------------------------------
    def test_shared_fold_plan_and_validation_distribution(self):
        pipelines = {
            "logistic_regression__smote": build_leakage_safe_pipeline(
                "logistic_regression", "smote"
            ),
            "decision_tree__random_under_sampler": (
                build_leakage_safe_pipeline(
                    "decision_tree", "random_under_sampler"
                )
            ),
            "logistic_regression__none": build_leakage_safe_pipeline(
                "logistic_regression", "none"
            ),
        }
        cv, oof, metadata = run_imbalance_cv(
            self.X,
            self.y,
            pipelines,
            self.plan,
            random_state=42,
        )

        self.assertEqual(metadata["fold_plan"]["n_splits"], 3)
        self.assertEqual(metadata["fold_plan"]["n_repeats"], 2)
        self.assertFalse(metadata["validation_resampling"])
        self.assertFalse(metadata["test_set_used"])
        self.assertEqual(metadata["seeds"]["cv_random_state"], 42)
        self.assertEqual(
            {int(value) for value in metadata["seeds"]["sampler_random_state_values"]},
            {42},
        )
        self.assertEqual(
            {
                int(value)
                for value in metadata["seeds"][
                    "estimator_random_state_values"
                ]
            },
            {42},
        )
        self.assertEqual(len(cv), len(pipelines) * len(self.plan))
        self.assertTrue(
            set(oof["row_index"]).issubset(set(self.X.index))
        )
        for _, row in oof.iterrows():
            self.assertEqual(
                row["y_true"], self.y.loc[row["row_index"]]
            )

        self.assertEqual(
            set(
                cv.loc[
                    cv["model"] == "logistic_regression", "n_features"
                ]
            ),
            {EXPECTED_FEATURE_COUNTS[MODEL_FAMILY_LINEAR_SVM]},
        )
        self.assertEqual(
            set(cv.loc[cv["model"] == "decision_tree", "n_features"]),
            {EXPECTED_FEATURE_COUNTS[MODEL_FAMILY_TREE]},
        )

        grouped = cv.groupby(["repeat", "fold"])
        self.assertEqual(
            grouped["train_index_fingerprint"].nunique().max(), 1
        )
        self.assertEqual(
            grouped["validation_index_fingerprint"].nunique().max(), 1
        )

        for _, group in grouped:
            counts = {
                tuple(sorted(counts.items()))
                for counts in group["validation_class_counts_before"]
            }
            self.assertEqual(len(counts), 1)
            for _, row in group.iterrows():
                self.assertEqual(
                    row["validation_class_counts_before"],
                    row["validation_class_counts_after"],
                )

    def test_fold_observer_receives_fitted_pipeline_and_sampler(self):
        pipeline = {
            "logistic_regression__smote": build_leakage_safe_pipeline(
                "logistic_regression", "smote"
            )
        }
        observations: list[dict[str, object]] = []

        def observer(**kwargs):
            observations.append(
                {
                    "repeat": int(kwargs["spec"].repeat),
                    "fold": int(kwargs["spec"].fold),
                    "n_train": len(kwargs["y_train"]),
                    "n_validation": len(kwargs["y_validation"]),
                    "sampler_before": int(
                        kwargs["sampler_step"].n_samples_before_
                    ),
                    "probability_count": int(
                        kwargs["probabilities"].shape[0]
                    ),
                    "has_sampling_encoder": (
                        "sampling_encoder" in kwargs["pipeline"].named_steps
                    ),
                }
            )

        cv, _, _ = run_imbalance_cv(
            self.X,
            self.y,
            pipeline,
            self.plan,
            random_state=42,
            fold_observer=observer,
        )

        self.assertEqual(len(observations), len(self.plan))
        self.assertEqual(len(cv), len(self.plan))
        self.assertEqual(
            [(row["repeat"], row["fold"]) for row in observations],
            [(spec.repeat, spec.fold) for spec in self.plan],
        )
        for row in observations:
            self.assertEqual(row["sampler_before"], row["n_train"])
            self.assertEqual(row["probability_count"], row["n_validation"])
            self.assertTrue(row["has_sampling_encoder"])

    def test_runner_records_seeds_and_before_after_counts(self):
        pipeline = {
            "logistic_regression__smote": build_leakage_safe_pipeline(
                "logistic_regression", "smote"
            )
        }
        cv, _, _ = run_imbalance_cv(
            self.X,
            self.y,
            pipeline,
            self.plan,
            random_state=42,
        )

        for _, row in cv.iterrows():
            spec = next(
                spec
                for spec in self.plan
                if spec.repeat == row["repeat"]
                and spec.fold == row["fold"]
            )
            expected_before = class_counts(
                self.y.iloc[spec.train_positions]
            )
            self.assertEqual(
                row["sampling_class_counts_before"], expected_before
            )
            self.assertEqual(
                sum(row["sampling_class_counts_after"].values()),
                row["n_train_after_sampling"],
            )
            self.assertEqual(
                row["n_train_before_sampling"], len(spec.train_positions)
            )
            self.assertEqual(row["sampler_random_state"], 42)
            self.assertEqual(row["cv_random_state"], 42)
            self.assertEqual(row["estimator_random_state"], 42)
            self.assertEqual(row["n_fit_resample_calls"], 1)
            self.assertTrue(row["sampling_applied"])
            self.assertNotEqual(
                row["sampling_class_counts_before"],
                row["sampling_class_counts_after"],
            )

        plain = {
            "logistic_regression__none": build_leakage_safe_pipeline(
                "logistic_regression", "none"
            )
        }
        plain_cv, _, _ = run_imbalance_cv(
            self.X,
            self.y,
            plain,
            self.plan,
            random_state=42,
        )
        for _, row in plain_cv.iterrows():
            self.assertFalse(row["sampling_applied"])
            self.assertIsNone(row["sampler"])
            self.assertEqual(row["n_fit_resample_calls"], 0)
            self.assertEqual(
                row["sampling_class_counts_before"],
                row["sampling_class_counts_after"],
            )
            self.assertEqual(
                row["n_train_before_sampling"],
                row["n_train_after_sampling"],
            )

    def test_fold_plan_validator_rejects_overlap(self):
        bad_plan = list(self.plan)
        first = bad_plan[0]
        bad_plan[0] = type(first)(
            repeat=first.repeat,
            fold=first.fold,
            train_positions=first.train_positions,
            valid_positions=np.concatenate(
                [first.valid_positions, first.train_positions[:1]]
            ),
        )
        with self.assertRaises(ValueError):
            validate_shared_fold_plan(
                bad_plan,
                n_rows=len(self.X),
                n_splits=3,
                n_repeats=2,
            )


if __name__ == "__main__":
    unittest.main()
