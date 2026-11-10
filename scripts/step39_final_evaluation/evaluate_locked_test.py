"""One-shot locked test access. Refuses repeat execution; no model fitting."""
from final_common import *
import warnings,time,joblib,traceback
from scipy.stats import binomtest
from src.data_loading import load_split
sys.path.insert(0,str(ROOT/'results/step37_statistical_robustness'))
from statistical_methods import WeightedPredictionScorer

def main():
    start=time.perf_counter();protocol=validate_seal('selection_freeze.json');deployment=validate_seal('model_and_threshold_freeze.json')
    assert deployment['selection_freeze_sha256']==sha(OUT/'selection_freeze.json')
    assert source_hashes()==protocol['source_hashes']==deployment['source_hashes']
    assert json.loads((OUT/'pretest_verification.json').read_text())['overall_passed']
    assert prior_snapshot()==protocol['original_artifact_hashes']
    for item in deployment['models'].values():assert sha(OUT/item['file'])==item['sha256']
    assert sha(OUT/deployment['calibration']['file'])==deployment['calibration']['sha256']
    assert not (OUT/'test_access_started.json').exists(),'One-shot marker exists; no repeat test access allowed'
    with (OUT/'test_access_started.json').open('x',encoding='utf-8') as h:
        json.dump(dict(created_utc=now(),selection_freeze_sha256=sha(OUT/'selection_freeze.json'),deployment_freeze_sha256=sha(OUT/'model_and_threshold_freeze.json'),note='Marker written before load_split. No rerun or test-guided changes permitted.'),h,indent=2)
    # First and only held-out API load/get in this analysis.
    split=load_split();assert not split.test_unlocked
    assert set(split.train_index)==set(deployment['training_membership'])
    split.test.unlock_for_final_evaluation();X,y=split.test.get()
    assert len(X)==1022 and int(y.sum())==50 and not set(X.index)&set(deployment['training_membership'])
    probabilities={};base_scores={};warn=[];prediction_events=[]
    for key in BASES:
        model=joblib.load(OUT/deployment['models'][key]['file']);assert np.array_equal(model.named_steps['classifier'].classes_,[0,1])
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');p=model.predict_proba(X)[:,1]
            if key==BASES[2]:base_scores[key]=model.decision_function(X)
        warn.extend(dict(base=key,phase='predict',category=w.category.__name__,message=str(w.message)) for w in seen)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        probabilities[key+'::original']=p;prediction_events.append(dict(base=key,predict_proba_calls=1,n_test=1022,utc=now()))
    calibrated=joblib.load(OUT/deployment['calibration']['file']).predict(base_scores[BASES[2]])
    assert np.isfinite(calibrated).all() and np.all((calibrated>=0)&(calibrated<=1));probabilities[SIG]=calibrated
    rows=pd.concat([pd.DataFrame(dict(stream=s,row_index=X.index.to_numpy(),y_true=y.to_numpy(),positive_probability=p)) for s,p in probabilities.items()])
    write(rows,'locked_test_predictions');dump(prediction_events,'prediction_call_provenance.json')
    # After this point, use saved arrays only. No fitting or test reload occurs.
    test_ids=X.index.to_numpy();yy=y.to_numpy();del X,y,split
    result=[];bins=[];binomial=[]
    for policy in deployment['policies']:
        p=probabilities[policy['stream']]
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');metrics=all_metrics(yy,p,policy['threshold'])
        warn.extend(dict(base=policy['id'],phase='diagnostic_metrics',category=w.category.__name__,message=str(w.message)) for w in seen)
        result.append(dict(**policy,**{k:v for k,v in metrics.items() if k!='threshold'}))
        for metric,k,n in [('sensitivity',metrics['tp'],metrics['tp']+metrics['fn']),('specificity',metrics['tn'],metrics['tn']+metrics['fp']),('precision',metrics['tp'],metrics['tp']+metrics['fp'])]:
            interval=binomtest(k,n).proportion_ci(method='exact') if n else None
            binomial.append(dict(id=policy['id'],metric=metric,k=k,n=n,estimate=k/n if n else None,exact95_low=interval.low if interval else None,exact95_high=interval.high if interval else None,defined=bool(n)))
    for stream,p in probabilities.items():
        for nb in [5,10]:
            for row in reliability_bins(yy,p,n_bins=nb,strategy='quantile'):
                ci=binomtest(row['n_positive'],row['n']).proportion_ci(method='exact') if row['n'] else None
                bins.append(dict(stream=stream,**row,observed_fraction_exact95_low=ci.low if ci else None,observed_fraction_exact95_high=ci.high if ci else None))
    write(pd.DataFrame(result),'locked_test_metrics');write(pd.DataFrame(binomial),'exact_binomial_intervals');write(pd.DataFrame(bins),'locked_test_reliability_bins')
    # Ordinary paired test-patient bootstrap, fixed fitted pipelines and thresholds.
    rng=np.random.default_rng(39042);counts=rng.multinomial(1022,np.full(1022,1/1022),size=5000).astype(np.uint16)
    assert np.all(counts@yy>0) and np.all(counts@(1-yy)>0),'Do not silently drop one-class resamples'
    weights=counts.astype(float);boot=np.empty((5000,6,len(BOOT_METRICS)));undefined=[]
    for j,policy in enumerate(deployment['policies']):
        p=probabilities[policy['stream']];scorer=WeightedPredictionScorer(yy,p)
        score,_=scorer.score(weights)
        positive=p>=policy['threshold'];tp=weights@(positive&(yy==1));fp=weights@(positive&(yy==0));fn=weights@((~positive)&(yy==1));tn=weights@((~positive)&(yy==0))
        sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(tp),where=den>0)
        precision=np.divide(tp,tp+fp,out=np.full_like(tp,np.nan),where=tp+fp>0)
        clipped=np.clip(p,1e-15,1-1e-15);loss=weights@(-yy*np.log(clipped)-(1-yy)*np.log1p(-clipped))/1022
        boot[:,j,:]=np.column_stack([score[:,0],score[:,1],score[:,2],sensitivity,specificity,precision,2*tp/(2*tp+fp+fn),mcc,.5*(sensitivity+specificity),loss])
        undefined.append(dict(id=policy['id'],precision_undefined_draws=int(np.isnan(precision).sum()),mcc_zero_denominator_draws=int((den==0).sum())))
    intervals=[]
    for j,policy in enumerate(deployment['policies']):
        for k,metric in enumerate(BOOT_METRICS):
            vals=boot[:,j,k];finite=np.isfinite(vals)
            # Never condition away undefined PPV draws: interval unavailable if any.
            ci=np.quantile(vals,[.025,.975]) if finite.all() else [None,None]
            intervals.append(dict(id=policy['id'],metric=metric,estimate=result[j][metric],percentile95_low=ci[0],percentile95_high=ci[1],n_draws=5000,n_defined=int(finite.sum()),interval_available=bool(finite.all()),degenerate=bool(finite.all() and vals.min()==vals.max()),scope='fixed_model_test_patient_mix_only; marginal_unadjusted; no_refitting'))
    np.savez_compressed(OUT/'test_patient_bootstrap_counts.npz',counts=counts,row_index=test_ids)
    np.savez_compressed(OUT/'test_patient_bootstrap_metrics.npz',values=boot,metrics=np.array(BOOT_METRICS),policies=np.array([p['id'] for p in deployment['policies']]))
    write(pd.DataFrame(intervals),'test_bootstrap_intervals');write(pd.DataFrame(undefined),'bootstrap_undefined_diagnostics');dump(warn,'test_warning_records.json')
    checks=[dict(check='exact held-out sample counts',passed=len(test_ids)==1022 and int(yy.sum())==50),
        dict(check='all5110 predictions and six frozen operating rows',passed=len(rows)==5110 and len(result)==6),
        dict(check='all original artifacts unchanged',passed=prior_snapshot()==protocol['original_artifact_hashes']),
        dict(check='frozen sources unchanged',passed=source_hashes()==protocol['source_hashes']),
        dict(check='all bootstrap draws retained',passed=counts.shape==(5000,1022)),
        dict(check='no unexpected warnings',passed=len(warn)==0)]
    verdict=dict(overall_passed=all(c['passed'] for c in checks),n_checks=len(checks),n_passed=sum(c['passed'] for c in checks),n_failed=sum(not c['passed'] for c in checks),checks=checks)
    dump(verdict,'test_execution_verification.json');dump(dict(completed_utc=now(),test_n=1022,test_events=50,test_nonevents=972,locked_api_get_calls=1,base_probability_calls=4,probability_streams=5,operating_rows=6,seconds=time.perf_counter()-start),'test_access_completed.json')
    dump([],'test_error_records.json');print(pd.DataFrame(result)[['id','threshold','roc_auc','average_precision','sensitivity','specificity','precision','mcc','brier_score','fp','fn']].to_string(index=False),flush=True)
    assert verdict['overall_passed'],'Preserve failed verification; no test rerun'
if __name__=='__main__':
    try:main()
    except Exception:
        dump([dict(utc=now(),traceback=traceback.format_exc())],'test_error_records.json');raise
