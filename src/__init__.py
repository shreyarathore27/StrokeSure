"""StrokeSure source package.

Modules
-------
config          Project paths, schema, feature groups and CV defaults.
data_loading    Validation, identifier handling, deterministic split.
preprocessing   Leakage-safe, per-fold preprocessing pipelines.
models          Documented baseline estimator definitions.
imbalance       Class-imbalance strategy registry and pipeline builder.
imbalance_pipelines  Categorical-aware, fold-local imbalance pipelines.
imbalance_experiment Fold-sharing CV runner with sampling audit records.
evaluation      Metrics, cross-validation and reproducible logging.
smoke_test      Cheap end-to-end checks (Step 20).
"""
