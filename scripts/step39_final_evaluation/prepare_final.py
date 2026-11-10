"""Phase A/B and training-only fitting. Never imports the held-out loader."""
from final_common import *
import platform,inspect,warnings,time,joblib,scipy,sklearn
from scipy.special import expit
from sklearn.calibration import _SigmoidCalibration
from src.config import RAW_DATA_PATH,FEATURE_COLUMNS
from src.imbalance_pipelines import build_leakage_safe_pipeline

def main():
    assert not (OUT/'selection_freeze.json').exists(),'No overwrite or automatic second final selection'
    start=time.perf_counter();before=prior_snapshot();checks=[]
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    # Independently certify all prior scientific artifacts using latest verified ledger.
    m=json.loads((ROOT/'results/step38_shap_explainability/analysis_manifest.json').read_text())
    for p,h in m['original_artifact_hashes'].items():check('prior hash '+p,sha(ROOT/p)==h)
    for row in pd.read_csv(ROOT/'results/step38_shap_explainability/artifact_inventory.csv').itertuples():
        check('Step38 inventory '+row.file,sha(ROOT/'results/step38_shap_explainability'/row.file)==row.sha256)
    stages={33:S33/(RUN+'_verification.json')}
    for n,folder in [(34,'step34_imbalance_analysis'),(35,'step35_calibration'),(36,'step36_threshold_analysis'),(37,'step37_statistical_robustness'),(38,'step38_shap_explainability')]:stages[n]=ROOT/'results'/folder/'verification.json'
    gates={}
    for n,path in stages.items():
        v=json.loads(path.read_text());check('Step '+str(n)+' verified',v['overall_passed']);gates[n]={k:v[k] for k in ['n_checks','n_passed','n_failed']}
    dataset=json.loads((S33/(RUN+'_manifest.json')).read_text())['data']
    check('unchanged raw dataset bytes',sha(RAW_DATA_PATH)==dataset['data_sha256'])
    oof=pd.read_csv(S33/(RUN+'_oof_predictions.csv'),float_precision='round_trip')
    check('625-fold repeated OOF coverage',len(oof)==511000 and oof.pipeline.nunique()==25 and oof['repeat'].nunique()==5 and all(len(g)==4088 and g.row_index.nunique()==4088 for _,g in oof.groupby(['pipeline','repeat'])))
    members=oof[(oof.pipeline==BASES[1])&(oof['repeat']==1)].sort_values('row_index');ids=members.row_index.to_numpy(int);allowed=set(ids)
    raw=pd.read_csv(RAW_DATA_PATH,skiprows=lambda line:line>0 and line-1 not in allowed,usecols=FEATURE_COLUMNS+['stroke']);raw.index=ids;X=raw[FEATURE_COLUMNS];y=raw.stroke
    check('training-only identity/labels',len(X)==4088 and int(y.sum())==199 and np.array_equal(y,members.y_true))
    nested=pd.read_csv(S35/'nested/outer_predictions.csv',float_precision='round_trip')
    refs=nested[(nested.pipeline==BASES[2])&(nested.method=='uncalibrated_reference')&(nested['repeat']==1)].sort_values('row_index')
    check('reusable 5-fold calibration scores',len(refs)==4088 and refs.row_index.nunique()==4088 and refs.fold.nunique()==5 and np.array_equal(refs.row_index,ids) and np.array_equal(refs.y_true,y) and set(refs.score_method)=={'decision_function'})
    check('score/probability consistency',np.allclose(expit(refs.base_score),refs.y_proba,atol=1e-14,rtol=0))
    check('no test access before protocol freeze',not (OUT/'test_access_started.json').exists())
    reasons={BASES[0]:'Natural-prevalence no-skill reference.',BASES[1]:'Linear probability/discrimination reference, no established AP/AUC inferiority to RF in Step37.',BASES[2]:'Exposes sensitivity/specificity/Brier trade-off of weighting; not assumed calibrated.',BASES[3]:'Nonlinear probability reference; RF sigmoid calibration showed a small Brier penalty, so retain raw RF.',SIG:'Nested sigmoid improved weighted-LR Brier while preserving within-fold ranking. Prefer parsimonious sigmoid to isotonic; not a universal winner.'}
    protocol=dict(phase='A/B protocol sealed before final fitting and any held-out loader',created_utc=now(),source_run=RUN,
        selection_status='Exploratory representative comparison chosen after development results, not a nested selection procedure or preregistration; no declared winner',
        selection_rule='Joint evidence review: AP/AUC ranking, sensitivity/specificity/precision at fixed and selected cutoffs, Brier/logloss/reliability and fold/repeat stability. No clinical cost margin exists, so no numerical composite utility or arbitrary winner. Exactly the five streams below; no post-test additions.',
        configurations=CONFIGS,rationale=reasons,baseline_hyperparameters='Exact existing build_leakage_safe_pipeline defaults; model seed42, ratio .5 and k5 passed unchanged; selected none/weight strategies perform no sampling',
        calibration=dict(method='sklearn.calibration._SigmoidCalibration, Platt smoothing; ensemble=False-style deployment',
            source='Reuse verified Step35 weighted-LR uncalibrated_reference base_score from repeat1, 5 outer folds; each patient score produced without fitting that patient. Fit one sigmoid on these4088 scores/labels. Refit weighted LR on all4088 and apply same map; no in-sample base scores for calibrator fitting.',
            reuse='Existing fits reused, no inner base fits necessary. 4 final base fits +1 calibration map.',
            implementation_sha256=sha(inspect.getfile(_SigmoidCalibration))),
        threshold=dict(reference=.5,additional_stream=SIG,policy='maximum_grid_mcc',
            training_scores='Saved Step35 sigmoid nested outer predictions, five separate pooled repeats, no test. Evaluate grid per repeat, average five correlated-repeat MCC scores, choose highest threshold within absolute1e-12 of maximum. No treating repeated patients as independent.',
            grid=GRID.tolist(),tie_break='highest threshold within abs1e-12 of max mean MCC; no secondary metric',
            limitation='Threshold is development-selected after evidence review; this selection-score maximum is optimistic and is not an unbiased development performance estimate. Full-training final map differs from outer-fold maps; fixed policy evaluated only on frozen test.'),
        fitting='Fit exactly four full-training pipelines, share weighted base for raw/sigmoid; no Step38 fitted models reused and SHAP not a primary selection criterion',
        test_protocol='Validate both freeze hashes and model hashes, exclusive one-shot access marker, load_split() after freezes, check locked, unlock_for_final_evaluation(), one get(), predict each base once and transform weighted scores once. Persist all5110 probabilities, then compute exactly six frozen model/policy rows. No test-guided refits or thresholds.',
        evaluation_metrics=BOOT_METRICS+['tn','fp','fn','tp','mean_probability','ece_quantile_5','ece_quantile_10','calibration_in_large_intercept','joint_intercept','calibration_slope'],
        diagnostics='Calibration coefficient fits use test predictions only as descriptive diagnostics, never applied to predictions; not new calibrator training.',
        uncertainty=dict(patient_bootstrap_draws=5000,seed=39042,scheme='ordinary paired multinomial test-patient resampling; same weights for all streams; fixed models/thresholds; no refits; percentile95 intervals, marginal/unadjusted; no significance tests',
            binomial='Exact95% Clopper-Pearson sensitivity/specificity/PPV conditional denominators; undefined PPV if no positive predictions. Bootstrap endpoint-degenerate intervals flagged; exact binomial intervals preferred for sensitivity/specificity.',
            limitations='Only50 events; no training/selection uncertainty, no external-population coverage guarantee, no multiplicity-adjusted inference or post-test hypothesis family.'),
        source_hashes=source_hashes(),original_artifact_hashes=before,training_ids_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest(),raw_data_sha256=dataset['data_sha256'],
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        known_discrepancies=['Dummy pooled-repeat AUC can depart from0.5 because fold-specific training priors differ; fold Dummy AUC0.5 is not contradicted.',
            'Step37 corrected-CV tests are approximate/exploratory, not confirmatory; H4 and independent model-seed sensitivity untested.',
            'Step38 Windows blocked SHAP library; exact analytical LR/original-feature RF enumeration used, with in-sample explanations; no generalization inference.',
            'Boosting candidates XGBoost/LightGBM/CatBoost have no evaluated artifacts in this verified experiment and must not appear as evaluated results.',
            'No identifiable repository manuscript document supplied for line-specific edits; corrections are a writing-team checklist.'])
    protocol_hash=seal(protocol,'selection_freeze.json');print('PROTOCOL SEALED',protocol_hash,flush=True)
    # All following fitting uses training only, and is constrained by the sealed protocol.
    models={};fitlog=[];warn=[]
    for key in BASES:
        t=time.perf_counter();model=build_leakage_safe_pipeline(*key.split('__'),random_state=42,sampling_strategy=.5,k_neighbors=5)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');model.fit(X,y)
        warn.extend(dict(base=key,category=w.category.__name__,message=str(w.message)) for w in seen)
        model_file=OUT/(key+'.joblib');joblib.dump(model,model_file);models[key]=dict(file=model_file.name,sha256=sha(model_file))
        fitlog.append(dict(base=key,n_training=4088,n_positive=199,random_state=42,seconds=time.perf_counter()-t,parameters=model.named_steps['classifier'].get_params()))
    calibrator=_SigmoidCalibration().fit(refs.base_score.to_numpy(),y.to_numpy())
    joblib.dump(calibrator,OUT/'weighted_lr_sigmoid.joblib')
    write(refs,'calibration_training_oof_scores')
    selected=nested[(nested.pipeline==BASES[2])&(nested.method=='sigmoid')]
    gridrows=[]
    for repeat,g in selected.groupby('repeat'):
        check('nested threshold selection coverage '+str(repeat),len(g)==4088 and g.row_index.nunique()==4088)
        values=grid_metrics(g.y_true,g.y_proba)
        gridrows.append(pd.DataFrame(dict(repeat=repeat,threshold=GRID,mcc=values['mcc'],sensitivity=values['sensitivity'],specificity=values['specificity'])))
    grid=pd.concat(gridrows);mean=grid.groupby('threshold',sort=True).mcc.mean();maximum=float(mean.max());threshold=float(mean.index[np.isclose(mean,maximum,rtol=0,atol=1e-12)][-1])
    write(grid,'training_threshold_grid_by_repeat');write(mean.reset_index(name='mean_correlated_repeat_mcc'),'training_threshold_selection')
    policies=[dict(id=c['id']+'::fixed_050',stream=c['id'],threshold=.5,policy='fixed_0.50') for c in CONFIGS]+[dict(id=SIG+'::training_mcc',stream=SIG,threshold=threshold,policy='maximum_grid_mcc')]
    dump(fitlog,'fitting_provenance.json');dump(warn,'training_warnings.json');check('no unexpected fit warnings',len(warn)==0)
    check('all original sources and artifacts unchanged',prior_snapshot()==before)
    check('frozen evaluation sources unchanged',source_hashes()==protocol['source_hashes'])
    deployment=dict(created_utc=now(),phase='All model/calibration/threshold choices numerical and sealed; test still untouched',selection_freeze_sha256=protocol_hash,models=models,
        calibration=dict(file='weighted_lr_sigmoid.joblib',sha256=sha(OUT/'weighted_lr_sigmoid.joblib'),a=float(calibrator.a_),b=float(calibrator.b_),n_oof=4088,positive=199),
        policies=policies,training_threshold=float(threshold),training_grid_maximum_mcc=maximum,threshold_selection_score_is_optimistic=True,
        training_membership=ids.tolist(),model_fits=4,calibrator_fits=1,calibration_oof_base_fits_reused=5,source_hashes=source_hashes())
    seal(deployment,'model_and_threshold_freeze.json')
    dump(dict(overall_passed=True,n_checks=len(checks),n_passed=len(checks),n_failed=0,checks=checks,seconds=time.perf_counter()-start),'pretest_verification.json')
    (OUT/'selection_protocol.md').write_text('# Final pre-test selection protocol\n\n'+json.dumps({k:protocol[k] for k in ['selection_status','selection_rule','rationale','calibration','threshold','uncertainty','known_discrepancies']},indent=2)+'\n\nBoth freeze files are exclusive-created, read-only and SHA-256 sealed; the evaluator validates them and refuses repeat access. This is an application-level audit seal, not adversarial write-proof storage. No winner or clinical policy is asserted.\n',encoding='utf-8')
    print('READY FOR LOCKED TEST; THRESHOLD',threshold,'PRETEST CHECKS',len(checks),flush=True)
if __name__=='__main__':main()
