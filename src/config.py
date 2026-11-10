"""Central configuration for the StrokeSure research project.

This module is the single source of truth for file locations, the feature
groups defined in Notebook 01/02, the deterministic train/test split and the
default cross-validation settings.

It deliberately contains no data loading, fitting or evaluation logic so that
it can be imported safely from notebooks, scripts and tests.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "healthcare-dataset-stroke-data.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Smoke-test artefacts are stored outside results/tables and results/figures
# so that they can never be mistaken for publication results.
SMOKE_DIR = RESULTS_DIR / "smoke_test"
SMOKE_TABLES_DIR = SMOKE_DIR / "tables"
SMOKE_FIGURES_DIR = SMOKE_DIR / "figures"

# ---------------------------------------------------------------------------
# Dataset schema (from Notebook 01 and the raw CSV header)
# ---------------------------------------------------------------------------
TARGET_COLUMN = "stroke"
ID_COLUMN = "id"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
ALLOWED_TARGET_VALUES = (0, 1)

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

# Feature groups reproduce Notebook 02 exactly.
NUMERIC_FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
]
BINARY_FEATURES = [
    "hypertension",
    "heart_disease",
]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# "Unknown" is a genuine data category in the source dataset (not a missing
# value marker), so it must survive one-hot encoding as its own column.
SMOKING_COLUMN = "smoking_status"
SMOKING_UNKNOWN_VALUE = "Unknown"

# ---------------------------------------------------------------------------
# Split configuration (must match Notebook 02)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Cross-validation and evaluation defaults
# ---------------------------------------------------------------------------
CV_N_SPLITS = 5
CV_SHUFFLE = True
DECISION_THRESHOLD = 0.5

# Repeated-CV protocol for the baseline experiment (Steps 21-24).
# 5 folds x 5 repeats, with a fresh shuffle per repeat (RepeatedStratifiedKFold).
CV_N_REPEATS = 5

# Smoke test (deliberately small and cheap).
SMOKE_SAMPLE_SIZE = 600
SMOKE_CV_SPLITS = 3

# ---------------------------------------------------------------------------
# Class-imbalance strategy defaults (Step 26)
# ---------------------------------------------------------------------------
# These values define the *starting point* for the imbalance experiment, not a
# tuned result. Any change must be pre-registered before evaluation. The
# sampling ratio is passed directly to imbalanced-learn; 0.5 is a moderate
# minority:majority target, well below the natural 1:19.5 ratio.
IMBALANCE_RANDOM_STATE = RANDOM_STATE
IMBALANCE_DEFAULT_SAMPLING_RATIO = 0.5
IMBALANCE_DEFAULT_K_NEIGHBORS = 5

# ---------------------------------------------------------------------------
# Preprocessing families (see Notebook 02, "Final Preprocessing Strategy")
# ---------------------------------------------------------------------------
MODEL_FAMILY_LINEAR_SVM = "linear_svm"
MODEL_FAMILY_TREE = "tree"
MODEL_FAMILIES = (MODEL_FAMILY_LINEAR_SVM, MODEL_FAMILY_TREE)

# Notebook 02 produced 16 predictors for linear/SVM models and 21 for trees.
EXPECTED_FEATURE_COUNTS = {
    MODEL_FAMILY_LINEAR_SVM: 16,
    MODEL_FAMILY_TREE: 21,
}

# Metric columns used throughout evaluation.
METRIC_NAMES = [
    "roc_auc",
    "average_precision",
    "recall_sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
    "brier_score",
]

# Human-readable labels. Average precision is NOT trapezoidal PR-AUC.
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision (step-wise PR curve, not trapezoidal PR-AUC)",
    "recall_sensitivity": "Recall / sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "brier_score": "Brier score (lower is better)",
}

# ---------------------------------------------------------------------------
# Experiment prefixes
# ---------------------------------------------------------------------------
SMOKE_RUN_PREFIX = "smoke"
BASELINE_RUN_PREFIX = "baseline_cv"


def ensure_output_directories() -> None:
    """Create output directories if they do not exist (never deletes data)."""
    for path in (
        TABLES_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        LOGS_DIR,
        SMOKE_TABLES_DIR,
        SMOKE_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
