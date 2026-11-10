"""Complete descriptive probability audit before any new calibration fitting."""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import pandas as pd
from calibration_metrics import probability_metrics,reliability_bins,LOSS_EPS,LOGIT_EPS

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN='imbalance_cv_step33_20261109T165943Z'
SOURCE=ROOT/'results/imbalance_cv'/RUN
STEP34=ROOT/'results/step34_imbalance_analysis'
SELECTED=[
    'logistic_regression__none','logistic_regression__class_weight_balanced',
    'logistic_regression__random_under_sampler','logistic_regression__adasyn',
    'random_forest__none','svc_rbf__random_under_sampler']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(obj,p): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def main():
    snapshot={str(p.relative_to(ROOT)):sha(p) for d in [SOURCE,STEP34] for p in d.iterdir() if p.is_file()}
    manifest=json.loads((SOURCE/f'{RUN}_manifest.json').read_text())
    for item in manifest['files'].values():
        if item['sha256']: assert sha(ROOT/item['path'])==item['sha256']
    inventory=pd.read_csv(STEP34/'artifact_inventory.csv')
    for row in inventory.itertuples(): assert sha(STEP34/row.file)==row.sha256
    predictions=pd.read_csv(SOURCE/f'{RUN}_oof_predictions.csv',float_precision='round_trip')
    summary=pd.read_csv(STEP34/'comparison_pooled_repeat_mean_sd.csv',float_precision='round_trip')
    assert len(predictions)==511000 and predictions.pipeline.nunique()==25
    assert not predictions.duplicated(['pipeline','repeat','row_index']).any()
    reference=predictions[(predictions.pipeline=='dummy_prior__none')&(predictions.repeat==1)].set_index('row_index').y_true.sort_index()
    metrics=[]; bins=[]
    for (pipeline,repeat),g in predictions.groupby(['pipeline','repeat']):
        assert len(g)==4088 and int(g.y_true.sum())==199 and g.set_index('row_index').y_true.sort_index().equals(reference)
        record=probability_metrics(g.y_true,g.y_proba)
        record.update(pipeline=pipeline,repeat=int(repeat),scope='Step33_pooled_within_repeat')
        metrics.append(record)
        for n,strategy in [(5,'quantile'),(10,'quantile'),(20,'quantile'),(10,'uniform')]:
            for row in reliability_bins(g.y_true,g.y_proba,n,strategy):
                bins.append(dict(pipeline=pipeline,repeat=int(repeat),**row))
    metrics=pd.DataFrame(metrics)
    for key,g in metrics.groupby('pipeline'):
        saved=summary[summary.pipeline==key].iloc[0]
        assert abs(g.brier_score.mean()-saved.brier_score_mean)<1e-12
        assert abs(g.roc_auc.mean()-saved.roc_auc_mean)<1e-12
        assert abs(g.average_precision.mean()-saved.average_precision_mean)<1e-12
    metrics.to_csv(OUT/'uncalibrated_metrics_by_repeat.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'uncalibrated_reliability_bins.csv',index=False)
    fields=[c for c in metrics.select_dtypes(include='number').columns if c!='repeat']
    stats=metrics.groupby('pipeline')[fields].agg(['mean','std','min','max'])
    stats.columns=['_'.join(c) for c in stats.columns]
    stats.reset_index().to_csv(OUT/'uncalibrated_reliability_summary.csv',index=False)
    reasons={
        SELECTED[0]:'Strong discrimination, zero-threshold-recall uncorrected LR reference; probability-error comparator.',
        SELECTED[1]:'Large observed minority detection and MCC, with high Brier/probability-error burden; paired LR weighting contrast.',
        SELECTED[2]:'Competitive AP and distinct undersampling approach; paired LR strategy contrast, not a final winner.',
        SELECTED[3]:'Custom categorical-aware synthetic strategy; mechanistic contrast with same LR estimator despite not maximizing AP.',
        SELECTED[4]:'Uncorrected nonlinear reference with low observed Brier but zero threshold recall; family diversity.',
        SELECTED[5]:'Nonlinear margin model and undersampling comparator with competitive AP; SVC already has internal Platt probabilities.'}
    pd.DataFrame([dict(pipeline=k,exploratory_reason=v) for k,v in reasons.items()]).to_csv(OUT/'exploratory_configuration_set.csv',index=False)
    design=dict(phase='frozen_after_completed_descriptive_audit',source_run=RUN,selected=SELECTED,
        exploratory_selection=True,selection_not_nested=True,
        selection_limitation='Configurations chosen from prior CV on these same development patients; nested calibration protects fitting/evaluation separation but does not remove configuration-selection optimism.',
        outer_cv=dict(n_splits=5,n_repeats=5,random_state=42,reuse_Step33_plan=True),
        inner_cv=dict(n_splits=5,shuffle=True,random_state=42),
        base_parameters='Exact existing build_leakage_safe_pipeline defaults; sampler ratio .5, k=5, random_state42; no tuning.',
        calibration_design='For each outer fold/configuration, five inner fits produce OOF scores for all natural-prevalence outer training observations. Fit each map on those scores/labels, refit the base on all outer training rows, evaluate three paired methods on disjoint outer validation rows. Share base fits across methods.',
        score_rule='decision_function if available (LR/SVC), else positive predict_proba (RF); SVC internal probability=True preserved.',
        methods=['uncalibrated_reference','sigmoid','isotonic'],
        sigmoid='Pinned sklearn.calibration._SigmoidCalibration (Platt target smoothing), source hashed; p=expit(-(a*score+b)).',
        isotonic='sklearn.isotonic.IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds=clip); exploratory with 3270/3271 calibration scores and 159/160 positives per map.',
        counts=dict(configurations=6,outer_folds_per_configuration=25,inner_base_fits=750,outer_refits=150,total_base_fits=900,sigmoid_maps=150,isotonic_maps=150,total_calibration_maps=300),
        binning='Primary 10 quantile bins per configuration/method/repeat; tied quantiles collapse. Sensitivity: 5/20 quantile and 10 equal-width bins. ECE/MCE are empirical bin-dependent descriptive diagnostics.',
        uncertainty='Display five repeat curves and bin event counts; repeat SD/range is descriptive, not CI. No naive independent-fold tests or binomial CIs ignoring CV dependence.',
        coefficient_estimation='Descriptive calibration-in-large intercept with slope fixed1; separate unpenalized joint intercept/slope on clipped logit(p). These evaluated-outcome fits are diagnostics only and never used to adjust or evaluate probabilities.',
        coefficients_estimable_rule='two outcome classes; logit SD >=1e-8; no complete separation; converged optimizer or gradient<1e-7; Hessian condition<1e12; standardized coefficients<50. Near-flat logit SD<.05 flagged.',
        log_loss_clip=LOSS_EPS,diagnostic_logit_clip=LOGIT_EPS,
        no_threshold_selection=True,heldout_access=False,
        original_artifact_hashes=snapshot,
        descriptive_coefficients_attempted=125,
        original_unchanged=all(sha(ROOT/k)==v for k,v in snapshot.items()))
    assert design['original_unchanged']
    dump(design,OUT/'design_and_provenance.json')
    dump(dict(overall_passed=True,configurations=25,repeats=5,n_prediction_records=511000,
              n_reliability_metric_rows=len(metrics),saved_scores_match=True,original_artifacts_unchanged=True),OUT/'descriptive_verification.json')
    print(stats[['brier_score_mean','log_loss_mean','ece_quantile_10_mean','mean_probability_mean','calibration_in_large_intercept_mean','calibration_slope_mean']].to_string(),flush=True)
    print('DESCRIPTIVE COMPLETE; frozen calibration design saved.',flush=True)

if __name__=='__main__': main()
