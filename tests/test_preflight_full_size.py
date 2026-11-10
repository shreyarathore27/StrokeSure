"""Step 32 tests: G9 full-size preflight runner (lightweight, synthetic data).

The real 25-attempt preflight on the training partition is executed once out
of band; these tests exercise the same code paths on tiny synthetic data and
never write into the real ``results/`` tree.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imbalance_outputs import (
    CapturingRecordingSampler,
    OutputIsolationError,
    build_audited_grid,
)
from src.imbalance import build_imbalance_pipeline
from src.imbalance_pipelines import build_leakage_safe_pipeline
from src.preflight import build_fold_plan
from src.preflight_full_size import (
    PREFLIGHT_STEP,
    analyze_sampling_geometry,
    create_preflight_directory,
    expected_supported_grid_keys,
    run_preflight_attempts,
    supported_grid_keys,
    verify_preflight_run,
)


def make_mixed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Re-use the established synthetic mixed-feature fixture."""
    from tests.test_imbalance_pipelines import make_mixed_data as fixture

    return fixture()


def _tiny_grid() -> dict[str, object]:
    grid = {
        "logistic_regression__none": build_imbalance_pipeline(
            "logistic_regression", "none"
        ),
        "logistic_regression__smote": build_leakage_safe_pipeline(
            "logistic_regression", "smote", record_sampling=True
        ),
    }
    return build_audited_grid(grid)


class PreflightRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        import src.preflight_full_size as preflight_module

        self.base = Path(tempfile.mkdtemp(prefix="preflight_test_"))
        self._original_results_dir = preflight_module.PREFLIGHT_RESULTS_DIR
        self._preflight_module = preflight_module
        preflight_module.PREFLIGHT_RESULTS_DIR = (
            self.base / "preflight_full_size"
        )
        self.addCleanup(self._restore_results_dir)

    def _restore_results_dir(self) -> None:
        self._preflight_module.PREFLIGHT_RESULTS_DIR = (
            self._original_results_dir
        )

    def _plan(self, X: pd.DataFrame, y: pd.Series):
        return build_fold_plan(X, y, n_splits=3, n_repeats=1, random_state=42)

    def test_attempts_record_status_timing_memory_and_audits(self) -> None:
        X, y = make_mixed_data()
        plan = self._plan(X, y)
        spec = plan[0]
        grid = _tiny_grid()
        attempt = run_preflight_attempts(X, y, grid, spec)

        self.assertEqual(set(attempt["statuses"]), set(grid))
        self.assertEqual(set(attempt["statuses"].values()), {"ok"})
        self.assertEqual(len(attempt["cv"]), len(grid))
        cv = attempt["cv"]
        self.assertTrue((cv["repeat"] == spec.repeat).all())
        self.assertTrue((cv["fold"] == spec.fold).all())
        self.assertEqual(cv["train_index_fingerprint"].nunique(), 1)
        self.assertEqual(cv["validation_index_fingerprint"].nunique(), 1)

        timings = attempt["timings"]
        self.assertEqual(len(timings), len(grid))
        self.assertTrue((timings["fit_seconds"] > 0).all())
        self.assertTrue((timings["predict_seconds"] > 0).all())
        self.assertTrue((timings["rss_peak_mb"] >= timings["rss_baseline_mb"]).all())
        # the shared fold's OOF is complete for both pipelines
        self.assertEqual(
            len(attempt["oof"]),
            2 * int(np.asarray(spec.valid_positions).size),
        )
        resampling_rows = attempt["sampling_audit"]
        self.assertEqual(
            set(resampling_rows["pipeline"]),
            {"logistic_regression__smote"},
        )
        self.assertTrue((resampling_rows["audit_status"] == "PASS").all())

    def test_broken_attempt_is_reported_with_fold_attribution(self) -> None:
        X, y = make_mixed_data()
        plan = self._plan(X, y)
        spec = plan[0]
        grid = _tiny_grid()
        from sklearn.base import BaseEstimator

        class BrokenEstimator(BaseEstimator):
            def fit(self, X, y):
                raise RuntimeError("simulated catastrophic preflight failure")

            def predict_proba(self, X):
                raise RuntimeError("unreachable")

        grid["decision_tree__none"] = build_imbalance_pipeline(
            "decision_tree", "none"
        ).set_params(classifier=BrokenEstimator())
        attempt = run_preflight_attempts(X, y, grid, spec)

        self.assertEqual(attempt["statuses"]["decision_tree__none"], "failed")
        self.assertEqual(attempt["statuses"]["logistic_regression__none"], "ok")
        errors = attempt["errors"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors.iloc[0]["error_type"], "RuntimeError")
        self.assertIn("simulated", errors.iloc[0]["message"])
        self.assertEqual(int(errors.iloc[0]["repeat"]), int(spec.repeat))
        self.assertEqual(int(errors.iloc[0]["fold"]), int(spec.fold))
        failed_row = attempt["timings"][
            attempt["timings"]["status"] == "failed"
        ]
        self.assertEqual(len(failed_row), 1)
        self.assertEqual(len(attempt["warnings"]), 0)

    def test_expected_attempt_count_and_grid_gate_in_verification(self) -> None:
        X, y = make_mixed_data()
        plan = self._plan(X, y)
        spec = plan[0]
        grid = _tiny_grid()
        attempt = run_preflight_attempts(X, y, grid, spec)
        common = dict(
            run_directory=SimpleNamespace(run_id="preflight_test"),
            cv=attempt["cv"],
            oof=attempt["oof"],
            timings=attempt["timings"],
            sampling_audit=attempt["sampling_audit"],
            warnings_table=attempt["warnings"],
            errors_table=attempt["errors"],
            statuses=attempt["statuses"],
            manifest={},
            plan_info={"n_splits": 3, "n_repeats": 1, "repeat_assignment_fingerprints": {1: "x"}},
            spec=spec,
            grid_keys=list(grid),
            geometry_report={"checks": []},
            data_metadata={"heldout_test_was_locked": True, "heldout_test_data_fields_accessed": False},
            pin={"sha256": "0" * 64},
            n_splits_plan=3,
            n_repeats_plan=1,
        )
        strict = verify_preflight_run(expect_full_grid=True, **common)
        relaxed = verify_preflight_run(expect_full_grid=False, **common)
        strict_checks = {
            check["check"]: check["passed"] for check in strict["checks"]
        }
        relaxed_checks = {
            check["check"]: check["passed"] for check in relaxed["checks"]
        }
        self.assertFalse(strict_checks["grid_matches_supported_set"])
        self.assertTrue(relaxed_checks["grid_matches_supported_set"])
        self.assertFalse(relaxed["overall_passed"])  # manifest is empty here
        self.assertTrue(relaxed_checks["expected_attempt_count_recorded"])
        self.assertTrue(relaxed_checks["single_shared_fold_used"])

    def test_expected_grid_keys_match_the_grid_builder(self) -> None:
        self.assertEqual(
            sorted(expected_supported_grid_keys()), sorted(supported_grid_keys())
        )

    def test_geometry_analysis_validates_smotenc_and_adasyn(self) -> None:
        X, y = make_mixed_data()
        plan = self._plan(X, y)
        spec = plan[0]
        report = analyze_sampling_geometry(X, y, spec, random_state=42)

        for section in (
            "representation",
            "smotenc_space",
            "smotenc_generation",
            "adasyn_space",
            "categoricaladasyn_generation",
            "categoricaladasyn_hardness",
            "checks",
            "standards_compliance_note",
        ):
            self.assertIn(section, report, msg=section)
        passed = {entry["check"] for entry in report["checks"] if entry["passed"]}
        self.assertIn("smotenc_synthetic_codes_integer_and_valid", passed)
        self.assertIn("adasyn_allocation_replication_is_exact", passed)
        self.assertIn("adasyn_synthetic_categories_come_only_from_seed_or_neighbour", passed)
        # the smotenc generation recorded a genuine measured count
        generation = report["smotenc_generation"]
        self.assertGreater(generation["n_synthetic"], 0)
        self.assertTrue(generation["all_synthetic_codes_integer"])
        # replication details expose the measured deviations, never silence
        replication = report["categoricaladasyn_generation"]["replication"]
        self.assertTrue(replication["shape_ok"])
        self.assertLessEqual(replication["max_abs_deviation_continuous"], 1e-9)

    def test_preflight_directory_is_fresh_and_prefixed(self) -> None:
        created = create_preflight_directory("preflight_unit_test_x")
        self.assertIn("preflight_unit_test_x", str(created.path))
        self.assertEqual(created.path.parent.name, "preflight_full_size")
        with self.assertRaises(OutputIsolationError):
            create_preflight_directory("preflight_unit_test_x")
        with self.assertRaises(OutputIsolationError):
            create_preflight_directory("baseline_sneaky")


if __name__ == "__main__":
    unittest.main()
