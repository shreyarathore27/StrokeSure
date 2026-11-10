"""Lightweight tests for the run-scoped imbalance output system (Step 31).

These tests verify the G1/G2/G3 implementation in
:mod:`src.imbalance_outputs` on small synthetic data only:

* output isolation (G1): fresh run directories, protected-path refusal,
  escaping-path refusal;
* manifest completeness (G2): source hashes, data hash, versions, git state,
  seeds, model parameters, sampling parameters, CV configuration;
* strict warning/error capture (G3): failed pipelines and unexpected warning
  categories fail verification and are never silently converted into
  successful experiments;
* repeatability: identical configuration reproduces identical fold metrics.

No experiment touches the real dataset; every run writes into a temporary
directory passed via ``results_dir``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.imbalance import build_imbalance_pipeline
from src.imbalance_outputs import (
    EXPECTED_WARNING_CATEGORIES,
    OutputIsolationError,
    classify_warning,
    create_run_directory,
    protected_paths,
    run_imbalance_experiment,
)
from tests.test_imbalance_pipelines import make_mixed_data


class WarningSimulationEstimator(BaseEstimator, ClassifierMixin):
    """Classifier that raises an *unexpected* numerical warning, then fits."""

    def fit(self, X, y):
        warnings.warn(
            "simulated numerical instability (RuntimeWarning)",
            RuntimeWarning,
            stacklevel=2,
        )
        self.model_ = LogisticRegression(max_iter=200)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class BrokenEstimator(BaseEstimator, ClassifierMixin):
    """Classifier whose fit always fails with a traceable error."""

    def fit(self, X, y):
        raise RuntimeError("simulated catastrophic fit failure")

    def predict_proba(self, X):  # pragma: no cover - never reached
        raise RuntimeError("predict_proba on a broken estimator")


class ImbalanceOutputsIsolationTests(unittest.TestCase):
    """G1: run directories can never overwrite or escape their scope."""

    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="imb_outputs_tests_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def test_create_run_directory_is_fresh_and_never_overwrites(self) -> None:
        first = create_run_directory(
            "test_run_a", results_dir=self.base / "imbalance"
        )
        self.assertTrue((first.path / ".keep").exists() is False)
        self.assertEqual(first.path.name, "test_run_a")
        with self.assertRaises(OutputIsolationError):
            create_run_directory("test_run_a", results_dir=self.base / "imbalance")

    def test_run_directory_inside_protected_path_is_refused(self) -> None:
        # The real data/ and results/tables trees are protected; a run may
        # never be created inside them.
        for protected in protected_paths():
            target = Path(protected)
            if not target.exists():
                continue  # only test roots that exist on disk
            issues = _overlap_issues(target)
            self.assertTrue(
                any("inside protected" in issue for issue in issues),
                msg=f"no overlap issue reported for {target}",
            )

    def test_baseline_prefix_run_name_is_refused(self) -> None:
        with self.assertRaises(OutputIsolationError):
            create_run_directory(
                "baseline_results", results_dir=self.base / "imbalance"
            )

    def test_check_path_rejects_escaping_and_baseline_names(self) -> None:
        run_dir = create_run_directory(
            "test_run_b", results_dir=self.base / "imbalance"
        )
        with self.assertRaises(OutputIsolationError):
            run_dir.check_path(self.base / "results" / "tables" / "x.csv")
        with self.assertRaises(OutputIsolationError):
            run_dir.check_path(run_dir.path / "baseline_something.csv")
        with self.assertRaises(OutputIsolationError):
            run_dir.check_path(run_dir.path / "sub" / "x.csv")
        with self.assertRaises(OutputIsolationError):
            run_dir.check_path(run_dir.path / "not allowed name.csv")
        # A legitimate artifact name passes.
        self.assertEqual(
            run_dir.check_path(run_dir.path / "test_run_b_metrics.csv").name,
            "test_run_b_metrics.csv",
        )


# ---------------------------------------------------------------------------
# Small experiment harness (synthetic data only)
# ---------------------------------------------------------------------------
def _overlap_issues(path: Path):
    from src.imbalance_outputs import _protected_overlap_issues

    return _protected_overlap_issues(Path(path))


def _data_metadata() -> dict:
    return {
        "data_sha256": "0f" * 32,  # tests never touch the real dataset
        "n_rows": 240,
        "n_train": 240,
        "train_class_counts": {"1": 48, "0": 192},
        "train_prevalence": 0.2,
        "heldout_test_was_locked": True,
        "heldout_test_data_fields_accessed": False,
    }


def _tiny_grid() -> dict:
    return {
        "logistic_regression__none": build_imbalance_pipeline(
            "logistic_regression", "none"
        ),
        "logistic_regression__random_over_sampler": build_imbalance_pipeline(
            "logistic_regression", "random_over_sampler"
        ),
    }


def _run_tiny_experiment(
    results_dir,
    *,
    grid=None,
    n_rows: int = 240,
    n_positive: int = 48,
    run_id: str | None = None,
):
    X, y = make_mixed_data(n_rows=n_rows, n_positive=n_positive)
    return run_imbalance_experiment(
        X,
        y,
        grid if grid is not None else _tiny_grid(),
        data_metadata=_data_metadata(),
        step="step31_unit_test",
        n_splits=3,
        n_repeats=1,
        random_state=42,
        results_dir=results_dir,
        run_id=run_id,
    )


class ImbalanceOutputsExperimentTests(unittest.TestCase):
    """The full run path on synthetic data: artifacts, manifest, verification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path(tempfile.mkdtemp(prefix="imb_outputs_run_"))
        cls.result = _run_tiny_experiment(cls.base / "imbalance")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_all_expected_artifacts_exist(self) -> None:
        run_dir = self.result.run_directory.path
        expected = {
            "fold_metrics",
            "oof_predictions",
            "summary",
            "summary_wide",
            "pooled_by_repeat",
            "sampling_audit",
            "warnings",
            "errors",
            "fold_plan",
            "verification",
            "report",
            "manifest",
        }
        self.assertEqual(set(self.result.paths), expected)
        for key, path in self.result.paths.items():
            self.assertTrue(
                Path(path).is_file(), msg=f"{key} was not written: {path}"
            )
            self.assertTrue(
                str(path).startswith(str(run_dir)),
                msg=f"{key} escaped the run directory: {path}",
            )
        self.assertTrue((run_dir / f"{self.result.run_directory.run_id}.log").is_file())

    def test_verification_passes_for_a_clean_run(self) -> None:
        self.assertTrue(
            self.result.overall_passed,
            msg=json.dumps(self.result.verification, indent=2)[:2000],
        )
        self.assertEqual(self.result.verification["n_failed"], 0)

    def test_sampling_audit_is_complete_and_passing(self) -> None:
        audit = self.result.sampling_audit
        # one audit per resampling fold (3 folds x 1 repeat)
        self.assertEqual(len(audit), 3)
        self.assertEqual(set(audit["audit_status"]), {"PASS"})
        self.assertEqual(
            set(audit["pipeline"]),
            {"logistic_regression__random_over_sampler"},
        )
        self.assertTrue((audit["n_fractional_one_hot_values"] == 0).all())
        self.assertTrue(audit["appended_rows_checked"].all())

    def test_manifest_completeness_g2(self) -> None:
        manifest = self.result.manifest
        # package versions
        env = manifest["environment"]
        for key in (
            "python",
            "platform",
            "numpy",
            "scipy",
            "pandas",
            "scikit_learn",
            "joblib",
            "threadpoolctl",
            "imbalanced_learn",
            "git_commit",
            "git_branch",
        ):
            self.assertTrue(env.get(key), msg=f"environment.{key} missing")
        self.assertIsInstance(env["git_dirty"], bool)
        self.assertTrue(env["pip_freeze"], "pip_freeze missing")
        # source hashes (G2 provenance freeze)
        source_files = manifest["source_files"]
        self.assertGreaterEqual(len(source_files), 10)
        for relative, digest in source_files.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", msg=f"{relative}")
        # dataset hash and seeds, model/sampling/CV configuration
        self.assertRegex(manifest["data"]["data_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["cv"]["n_splits"], 3)
        self.assertEqual(manifest["cv"]["n_repeats"], 1)
        self.assertEqual(manifest["cv"]["cv_random_state"], 42)
        self.assertIn("decision_threshold", manifest["cv"])
        self.assertEqual(manifest["sampling"]["sampling_strategy_ratio"], 0.5)
        self.assertEqual(manifest["sampling"]["k_neighbors"], 5)
        self.assertEqual(manifest["sampling"]["sampler_seed"], 42)
        self.assertEqual(manifest["random_state"], 42)
        # per-pipeline model parameters
        described = {record["pipeline"]: record for record in manifest["pipelines"]}
        self.assertEqual(
            set(described),
            {
                "logistic_regression__none",
                "logistic_regression__random_over_sampler",
            },
        )
        for key, record in described.items():
            self.assertTrue(record["classifier"]["params"], msg=f"{key}")
            self.assertIn("preprocessor_family", record)
        # repeatability pin + git + final wall time + file hashes
        self.assertRegex(manifest["repeatability_pin"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(manifest["total_seconds"], float)
        # G2: start/end source snapshots, configuration hashes, fold fingerprints
        self.assertEqual(
            set(manifest["source_files_start"]),
            set(manifest["source_files"]),
            msg="start and end source hashes must cover the same files",
        )
        for key in ("source_files_start", "source_files_end"):
            for digest in manifest[key].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$", msg=key)
        self.assertEqual(
            manifest["source_files_start"],
            manifest["source_files_end"],
            msg="no source file may change during a run",
        )
        self.assertRegex(
            manifest["configuration_hashes"]["requirements.txt"],
            r"^[0-9a-f]{64}$",
        )
        fingerprints = manifest["fold_plan"]["repeat_assignment_fingerprints"]
        self.assertTrue(fingerprints, "fold-plan repeat fingerprints missing")
        for fingerprint in fingerprints.values():
            self.assertRegex(fingerprint, r"^[0-9a-f]{16}$")
        # artifact hashes in the files map
        files = manifest["files"]
        self.assertEqual(files["manifest"]["sha256"], None)
        run_dir = self.result.run_directory.path.resolve()
        for key, entry in files.items():
            if key == "manifest":
                continue
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", msg=f"{key}")
            self.assertEqual(
                Path(entry["path"]).resolve().parent,
                run_dir,
                msg=f"{key}: artifact must live inside the run directory",
            )

    def test_repeatability_pin_matches_runtime(self) -> None:
        # Covered inside verification for a clean run; additionally confirm
        # the recorded cv/sampler seeds in the fold metrics.
        cv = pd.read_csv(self.result.paths["fold_metrics"])
        if len(cv):
            self.assertTrue((cv["cv_random_state"] == 42).all())


class ImbalanceOutputsStrictCaptureTests(unittest.TestCase):
    """G3: failures and unexpected warnings must never pass silently."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path(tempfile.mkdtemp(prefix="imb_strict_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_broken_pipeline_is_reported_and_fails_the_run(self) -> None:
        grid = _tiny_grid()
        grid["decision_tree__none"] = build_imbalance_pipeline(
            "decision_tree", "none"
        ).set_params(classifier=BrokenEstimator())
        result = _run_tiny_experiment(base := self.base / "imbalance_error", grid=grid)

        self.assertEqual(result.pipeline_status["decision_tree__none"], "failed")
        self.assertEqual(result.pipeline_status["logistic_regression__none"], "ok")
        self.assertFalse(result.overall_passed)

        errors = pd.read_csv(result.paths["errors"])
        # one explicit error row per failed fold (3 folds x 1 repeat)
        self.assertEqual(len(errors), 3)
        self.assertEqual(set(errors["error_type"]), {"RuntimeError"})
        self.assertTrue(errors["message"].str.contains("simulated").all())
        self.assertEqual(
            set(zip(errors["repeat"], errors["fold"])),
            {(1, 1), (1, 2), (1, 3)},
        )
        self.assertTrue(errors["traceback"].str.contains("Traceback").all())
        fold_counts = result.manifest["pipeline_fold_counts"]
        self.assertEqual(
            fold_counts["decision_tree__none"],
            {"successful": 0, "failed": 3},
        )

        # no OOF predictions may exist for the failed pipeline
        oof = pd.read_csv(result.paths["oof_predictions"])
        self.assertNotIn("decision_tree__none", set(oof["pipeline"]))

        passed = {
            check["check"] for check in result.verification["checks"] if check["passed"]
        }
        failed = {
            check["check"] for check in result.verification["checks"] if not check["passed"]
        }
        self.assertIn("errors_reported_consistently", passed)
        self.assertIn("all_folds_attempted", passed)
        self.assertIn("all_pipelines_fit_successfully", failed)

    def test_unexpected_numerical_warning_fails_the_run(self) -> None:
        grid = {
            "logistic_regression__none": build_imbalance_pipeline(
                "logistic_regression", "none"
            ).set_params(classifier=WarningSimulationEstimator())
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # capture happens inside the runner
            result = _run_tiny_experiment(
                self.base / "imbalance_warning", grid=grid
            )

        self.assertFalse(result.overall_passed)
        self.assertGreater(len(result.warnings_table), 0)
        self.assertTrue(
            (result.warnings_table["category"] == "RuntimeWarning").all()
        )
        failed_checks = {
            check["check"] for check in result.verification["checks"] if not check["passed"]
        }
        self.assertIn("warnings_expected_categories_only", failed_checks)

    def test_classify_warning_exposes_the_expected_allow_list(self) -> None:
        self.assertEqual(EXPECTED_WARNING_CATEGORIES, frozenset({"FutureWarning"}))
        expected = {
            "category": "FutureWarning",
            "message": "The `probability` parameter was deprecated in 1.9.",
        }
        status, _ = classify_warning(expected)
        self.assertEqual(status, "EXPECTED")
        for record in (
            {"category": "RuntimeWarning", "message": "overflow"},
            {"category": "UserWarning", "message": "probability mishap"},
            {"category": "FutureWarning", "message": "some other deprecation"},
        ):
            status, reason = classify_warning(record)
            self.assertEqual(status, "UNEXPECTED", msg=str(record))


class ImbalanceOutputsRepeatabilityTests(unittest.TestCase):
    """Identical configuration must reproduce identical fold metrics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path(tempfile.mkdtemp(prefix="imb_repeat_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_two_runs_produce_identical_fold_metrics_for_the_none_pipeline(
        self,
    ) -> None:
        first = _run_tiny_experiment(
            self.base / "imbalance", run_id="repeat_check_1"
        )
        second = _run_tiny_experiment(
            self.base / "imbalance", run_id="repeat_check_2"
        )
        self.assertNotEqual(
            first.run_directory.path, second.run_directory.path
        )
        a = pd.read_csv(first.paths["fold_metrics"])
        b = pd.read_csv(second.paths["fold_metrics"])
        subset = a[a["pipeline"] == "logistic_regression__none"].reset_index(drop=True)
        subset_b = b[b["pipeline"] == "logistic_regression__none"].reset_index(drop=True)
        metric_columns = [
            "average_precision",
            "roc_auc",
            "f1",
            "precision",
            "recall_sensitivity",
            "specificity",
            "balanced_accuracy",
            "n_train_before_sampling",
            "n_train_after_sampling",
        ]
        for column in metric_columns:
            np.testing.assert_allclose(
                subset[column].to_numpy(dtype=float),
                subset_b[column].to_numpy(dtype=float),
                rtol=0,
                atol=0,
                err_msg=f"non-deterministic metric column {column}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
