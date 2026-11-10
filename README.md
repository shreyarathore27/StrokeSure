# StrokeSure
### Beyond Accuracy: A Robust and Explainable Machine Learning Framework for Stroke Risk Prediction Under Class Imbalance

**Authors:** Shreya Bisht, Shreya Rathore, Vaibhavi Singh  
**Institution:** Indira Gandhi Delhi Technical University for Women  
**Department:** Computer Science and Engineering (Artificial Intelligence)

## Overview

StrokeSure is a machine learning research project investigating stroke-label prediction under severe class imbalance.

Rather than relying on classification accuracy alone, the project evaluates discrimination, minority-class detection, probability calibration, statistical robustness, model explainability, and classification-threshold trade-offs.

The contribution is a systematic, reproducible experimental comparison—not a new clinical diagnostic system or a claim of algorithmic novelty.

## Dataset

- **Dataset:** Healthcare Stroke Prediction Dataset
- **Source:** [Kaggle — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Original filename:** `healthcare-dataset-stroke-data.csv`
- **Date accessed:** 28 September 2026
- **Task:** Binary classification
- **Target:** `stroke`
- **Observations:** 5,110
- **Original predictors:** 10
- **Stroke-positive observations:** 249 (4.87%)
- **Missing BMI values:** 201
- **Unknown smoking-status entries:** 1,544
- **Duplicate records identified:** 0

### Dataset Partition

| Partition | Observations | Stroke-positive |
|---|---:|---:|
| Development | 4,088 | 199 |
| Held-out test | 1,022 | 50 |

The established experiment used a stratified 80/20 split with random state 42.

## Evaluated Models

- Dummy baseline
- Logistic Regression
- Decision Tree
- Random Forest
- RBF Support Vector Machine

## Class-Imbalance Strategies

1. No correction
2. Balanced class weighting
3. Random oversampling
4. Random undersampling
5. SMOTENC
6. Custom CategoricalADASYN

CategoricalADASYN is a project-specific implementation using fold-local scaling for continuous features and categorical-aware reconstruction. It must not be confused with unmodified ADASYN.

The full experiment evaluated 25 supported configurations using five-fold, five-repeat stratified cross-validation.

**625 fold fits completed successfully, generating 511,000 out-of-fold prediction records.**

Preprocessing and resampling were fitted inside training folds to prevent validation leakage.

## Research Components

The completed study includes:

- Dataset-quality assessment
- Baseline model comparison
- Class-imbalance experiments
- Repeated cross-validation
- ROC-AUC and Average Precision
- Minority-class detection metrics
- Probability calibration
- Classification-threshold analysis
- Exploratory dependence-aware statistical comparisons
- Global and local SHAP explanations
- Frozen, one-shot held-out evaluation

## Selected Held-Out Results

The final held-out partition contained 1,022 observations, including 50 stroke-positive cases.

| Model / operating point | ROC-AUC | AP | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Uncorrected LR, threshold 0.50 | 0.8421 | 0.2700 | 2% | 100% |
| Weighted LR, threshold 0.50 | 0.8436 | 0.2682 | 80% | 74.28% |
| Sigmoid-calibrated weighted LR, threshold 0.061 | 0.8436 | 0.2682 | 80% | 77.78% |

At the frozen 0.061 operating threshold, calibrated weighted Logistic Regression detected 40 of 50 positive observations, with 216 false positives and 10 false negatives.

The threshold was selected on development data before final test access.

This is an exploratory representative comparison, not a declaration of a universally superior or clinically validated model.

## Main Findings

- High ROC-AUC does not guarantee useful sensitivity at the default threshold.
- Class weighting can substantially improve sensitivity while reducing specificity and probability accuracy.
- Calibration can improve probability accuracy without creating new ranking information.
- Changing operating thresholds affects false-positive and false-negative trade-offs.
- SHAP attribution patterns vary across model families.
- No AP or ROC-AUC comparison survived the study's exploratory multiplicity adjustment.

## Repository Structure

```text
StrokeSure/
├── src/                 # ML pipelines and experiment implementations
├── tests/               # Automated verification tests
├── notebooks/           # Baseline research notebook
├── scripts/             # Copies of later-stage analysis scripts
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the public dataset from Kaggle and place the CSV at the path expected by `src/config.py`.

Run the available test suite using the project's configured Python environment.

### Reproducibility Notes

The repository contains core experiment implementations and copies of analysis scripts for the later research stages.

Large generated outputs, fitted models, logs, experiment manifests, and local results are excluded from Git by default.

Some later-stage scripts depend on saved experiment artifacts and their original directory structure. Consequently, a fresh clone alone does not reproduce every reported table and figure.

The final held-out test partition has already been evaluated under frozen selection rules. The historical evaluation script is retained for auditability and should not be rerun for model tuning or further test-guided comparisons.

## Scientific Limitations

- Single public dataset with limited clinical provenance
- Only 50 positive cases in the final test partition
- Exploratory configuration selection
- Correlated observations across repeated cross-validation
- Approximate statistical comparisons
- In-sample SHAP analysis
- Independent seed-sensitivity analysis not performed
- No independent external or prospective clinical validation

StrokeSure is a research framework and should not be used as a medical diagnostic or clinical decision-support system without appropriate further validation.

## Research Paper

**Beyond Accuracy: A Robust and Explainable Machine Learning Framework for Stroke Risk Prediction Under Class Imbalance**

Authors: Shreya Rathore, Vaibhavi Singh.

The manuscript documents the completed experimental study, findings and limitations.
