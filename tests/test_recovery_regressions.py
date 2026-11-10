"""Recovery regressions; synthetic data only, including real artifact persistence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src import preflight_full_size as pf
from src.imbalance_outputs import (
    file_sha256, load_training_partition, source_file_hashes,
    run_imbalance_experiment,
)
from src.imbalance_pipelines import CategoricalADASYN, build_leakage_safe_pipeline
from tests.test_imbalance_pipelines import make_mixed_data


class RecoveryRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="recovery_regression_")
        cls.X, cls.y = make_mixed_data()
        metadata = {
            "data_sha256": "ab" * 32, "n_rows": len(cls.X), "n_train": len(cls.X),
            "heldout_test_was_locked": True,
            "heldout_test_data_fields_accessed": False,
        }
        grid = {f"logistic_regression__{s}": build_leakage_safe_pipeline(
            "logistic_regression", s) for s in ("none", "smote", "adasyn")}
        # Execute the actual persistence path. Only the data source and grid
        # are substituted; report, hashing, verification and manifest are real.
        with patch.object(pf, "PREFLIGHT_RESULTS_DIR", Path(cls.temp.name)), \
             patch.object(pf, "load_training_partition", return_value=(cls.X, cls.y, metadata)), \
             patch.object(pf, "supported_grid_keys", return_value=grid):
            cls.result = pf.run_full_size_preflight(
                run_id="preflight_recovery_test", expect_full_grid=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _verify(self, **overrides):
        r = self.result
        args = {k: r[k] for k in (
            "run_directory", "cv", "oof", "timings", "sampling_audit",
            "warnings_table", "errors_table", "statuses", "manifest",
            "plan_info", "spec", "geometry_report")}
        args.update(grid_keys=list(r["statuses"]), expect_full_grid=False,
                    data_metadata=r["manifest"]["data"],
                    pin=r["manifest"]["repeatability_pin"],
                    n_splits_plan=5, n_repeats_plan=1)
        args.update(overrides)
        return pf.verify_preflight_run(**args)

    def test_report_exists_before_hash_and_manifest_hashes_match(self):
        r = self.result
        self.assertTrue(r["overall_passed"], r["verification"])
        manifest = json.loads(r["paths"]["manifest"].read_text(encoding="utf-8"))
        for key, path in r["paths"].items():
            self.assertTrue(path.is_file(), key)
            if key != "manifest":
                self.assertEqual(manifest["files"][key]["sha256"], file_sha256(path))
        report = r["paths"]["report"].read_text(encoding="utf-8")
        self.assertIn("cosine similarity", report)
        self.assertIn("PREFLIGHT ONLY", report)
        self.assertIn("src/preflight_full_size.py", manifest["source_files_start"])
        self.assertEqual(manifest["source_files_start"], manifest["source_files_end"])

    def test_smotenc_target_is_original_minority_plus_requested_additions(self):
        g = self.result["geometry_report"]["smotenc_generation"]
        spec = self.result["spec"]
        before = self.y.iloc[spec.train_positions]
        target = int((before == 0).sum() * 0.5)
        self.assertEqual(g["intended_minority_target"], target)
        self.assertEqual(g["final_minority_count"], target)
        self.assertEqual(g["n_synthetic"], target - int((before == 1).sum()))

    def test_geometry_replication_checks_every_generated_row(self):
        geometry = self.result["geometry_report"]
        self.assertTrue(all(c["passed"] for c in geometry["checks"]), geometry)
        g = geometry["categoricaladasyn_generation"]
        self.assertTrue(g["replication"]["shape_ok"])
        self.assertTrue(g["replication"]["is_exact"])
        self.assertEqual(g["replication"]["max_abs_deviation_categ_codes"], 0)
        self.assertEqual(g["category_source_fractions"]["other"], 0)

    def test_any_failed_geometry_check_blocks_verification(self):
        geometry = copy.deepcopy(self.result["geometry_report"])
        geometry["checks"].append({"check": "extra_integrity_check", "passed": False})
        v = self._verify(geometry_report=geometry)
        self.assertFalse(v["overall_passed"])
        self.assertFalse(next(c for c in v["checks"] if c["check"] == "sampling_geometry_verified")["passed"])

    def test_failed_pipeline_cannot_pass_even_with_consistent_error_capture(self):
        statuses = dict(self.result["statuses"])
        statuses["logistic_regression__none"] = "failed"
        v = self._verify(statuses=statuses)
        self.assertFalse(next(c for c in v["checks"] if c["check"] == "all_pipelines_fit_successfully")["passed"])

    def test_source_mutation_is_detected(self):
        manifest = copy.deepcopy(self.result["manifest"])
        manifest["source_files_end"]["src/imbalance_pipelines.py"] = "00" * 32
        v = self._verify(manifest=manifest)
        self.assertFalse(next(c for c in v["checks"] if c["check"] == "source_freeze_stable")["passed"])
        self.assertEqual(source_file_hashes()["src/imbalance_outputs.py"],
                         file_sha256(Path("src/imbalance_outputs.py")))

    def test_adasyn_scaling_and_fitted_generation_geometry_are_fold_local(self):
        spec = self.result["spec"]
        X = self.X.iloc[spec.train_positions]
        y = self.y.iloc[spec.train_positions]
        pipeline = build_leakage_safe_pipeline("logistic_regression", "adasyn")
        pipeline.fit(X, y)
        sampler = pipeline.named_steps["sampler"].sampler
        encoder = pipeline.named_steps["sampling_encoder"]
        Z = encoder.transform(X)
        categorical = np.asarray(sampler.categorical_features)
        continuous = np.setdiff1d(np.arange(Z.shape[1]), categorical)
        np.testing.assert_allclose(sampler.continuous_scaler_.mean_, Z[:, continuous].mean(axis=0))
        self.assertEqual(sampler.continuous_scaler_.n_samples_seen_, len(X))
        mixed = np.hstack([sampler.continuous_scaler_.transform(Z[:, continuous]),
                           sampler.categorical_encoder_.transform(Z[:, categorical])])
        np.testing.assert_array_equal(sampler.adasyn_.nn_._fit_X, mixed[np.asarray(y) == 1])
        full_nn = NearestNeighbors(n_neighbors=6).fit(mixed)
        neighbours = full_nn.kneighbors(mixed[np.asarray(y) == 1], return_distance=False)[:, 1:]
        hardness = (np.asarray(y)[neighbours] != 1).sum(axis=1) / 5
        allocation = np.rint(hardness / hardness.sum() * sampler.adasyn_.sampling_strategy_[1]).astype(int)
        resampled, labels = sampler.fit_resample(Z, y)
        self.assertEqual(len(labels) - len(y), int(allocation.sum()))
        np.testing.assert_array_equal(resampled[:len(Z)], Z)
        self.assertTrue(np.all(resampled[:, categorical] == np.rint(resampled[:, categorical])))
        # Changing continuous units must leave this standardized geometry and
        # decoded synthetic categories invariant (within floating precision).
        converted = Z.copy()
        converted[:, continuous] = converted[:, continuous] * 100 + 17
        other = CategoricalADASYN(categorical_features=categorical,
                                 sampling_strategy=0.5, random_state=42)
        converted_resampled, other_y = other.fit_resample(converted, y)
        np.testing.assert_array_equal(other_y, labels)
        np.testing.assert_array_equal(converted_resampled[:, categorical], resampled[:, categorical])
        np.testing.assert_allclose((converted_resampled[:, continuous] - 17) / 100,
                                   resampled[:, continuous], atol=1e-10)

    def test_training_loader_never_reads_locked_test_fields(self):
        class LockedSplit:
            X_train = self.X
            y_train = self.y
            test_unlocked = False
            metadata = dict(data_sha256="ab" * 32, n_rows=300, n_features=10,
                            n_train=240, n_test=60, test_size=0.2, random_state=42,
                            full_prevalence=0.2)

            def __getattr__(self, name):
                raise AssertionError(f"prohibited split field accessed: {name}")

        with patch("src.imbalance_outputs.load_split", return_value=LockedSplit()):
            X, y, metadata = load_training_partition()
        self.assertIs(X, self.X)
        self.assertIs(y, self.y)
        self.assertFalse(metadata["heldout_test_data_fields_accessed"])

    def test_repeated_cv_oof_verification_and_within_repeat_pooling(self):
        with tempfile.TemporaryDirectory(prefix="recovery_oof_") as directory:
            result = run_imbalance_experiment(
                self.X, self.y,
                {"logistic_regression__none": build_leakage_safe_pipeline(
                    "logistic_regression", "none")},
                data_metadata=self.result["manifest"]["data"],
                step="recovery_synthetic_repeated_oof", n_splits=3, n_repeats=2,
                random_state=42, results_dir=Path(directory), run_id="repeated_oof",
            )
            # Release this test's experiment logger before any assertions so
            # an assertion failure also leaves Windows temp cleanup possible.
            import logging
            for logger in list(logging.Logger.manager.loggerDict.values()):
                if isinstance(logger, logging.Logger):
                    for handler in list(logger.handlers):
                        if getattr(handler, "baseFilename", "").startswith(directory):
                            handler.close()
                            logger.removeHandler(handler)
            self.assertTrue(result.overall_passed, result.verification)
            import pandas as pd
            oof = pd.read_csv(result.paths["oof_predictions"])
            self.assertEqual(len(oof), 2 * len(self.X))
            self.assertFalse(oof.duplicated(["pipeline", "repeat", "row_index"]).any())
            for _, group in oof.groupby("repeat"):
                self.assertEqual(set(group["row_index"]), set(self.X.index))
                np.testing.assert_array_equal(
                    group["y_true"], self.y.loc[group["row_index"]])
            pooled = pd.read_csv(result.paths["pooled_by_repeat"])
            self.assertEqual(set(pooled["repeat"]), {1, 2})
            self.assertEqual(len(pooled), 2)

    def test_preflight_failure_releases_log_and_restores_output_scope(self):
        import logging
        original = pf.PREFLIGHT_RESULTS_DIR
        with tempfile.TemporaryDirectory(prefix="recovery_failure_") as directory:
            with patch.object(pf, "load_training_partition", return_value=(
                    self.X, self.y, self.result["manifest"]["data"])), \
                 patch.object(pf, "supported_grid_keys", side_effect=RuntimeError("injected grid failure")):
                with self.assertRaisesRegex(RuntimeError, "injected grid failure"):
                    pf.run_full_size_preflight(results_dir=Path(directory),
                                               run_id="preflight_failure")
            self.assertEqual(pf.PREFLIGHT_RESULTS_DIR, original)
            self.assertEqual(logging.getLogger("preflight_full_size.preflight_failure").handlers, [])
