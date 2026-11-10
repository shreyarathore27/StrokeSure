"""Paired nested probability calibration, frozen Step 35 development design."""
from pathlib import Path
import sys,json,csv,time,traceback,warnings,hashlib,inspect
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from src.imbalance_outputs import (load_training_partition,source_file_hashes,
    build_audited_grid,audit_resampling_fold,describe_pipelines,classify_warning)
from src.imbalance_pipelines import build_leakage_safe_pipeline
from src.preflight import build_fold_plan
from src.preflight_full_size import RssProbe

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(obj,p): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def fingerprint(ids): return hashlib.sha256(np.asarray(sorted(map(int,ids)),dtype=np.int64).tobytes()).hexdigest()[:16]
def append_frame(frame,path): frame.to_csv(path,mode='a',header=not path.exists(),index=False,float_format='%.17g')

def main():
    design=json.loads((OUT/'design_and_provenance.json').read_text())
    assert json.loads((OUT/'descriptive_verification.json').read_text())['overall_passed']
    target=OUT/'nested'
    target.mkdir(exist_ok=False)
    assert all(sha(ROOT/k)==v for k,v in design['original_artifact_hashes'].items())
    source_start=source_file_hashes()
    code_start={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'calibration_metrics.py',OUT/'step35_descriptive.py']}
    library_source={str(Path(inspect.getfile(obj)).resolve()):sha(inspect.getfile(obj)) for obj in [_SigmoidCalibration,IsotonicRegression]}
    X,y,metadata=load_training_partition()
    assert len(X)==4088 and int(y.sum())==199 and not metadata['heldout_test_data_fields_accessed']
    outer=build_fold_plan(X,y,n_splits=5,n_repeats=5,random_state=42)
    saved_step33=pd.read_csv(ROOT/f'results/imbalance_cv/{design["source_run"]}/{design["source_run"]}_oof_predictions.csv',float_precision='round_trip')
    bases={key:build_leakage_safe_pipeline(*key.split('__'),random_state=42,
                                          sampling_strategy=0.5,k_neighbors=5,record_sampling=True)
           for key in design['selected']}
    bases=build_audited_grid(bases)
    # None/weight builders reject sampling settings? Existing builders accept
    # the documented defaults and ignore sampling for nonsampler strategies.
    split_records=[]
    for spec in outer:
        train=np.asarray(spec.train_positions); valid=np.asarray(spec.valid_positions)
        assert not set(train)&set(valid) and len(train)+len(valid)==4088
        inners=[]
        for i,(a,b) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X.iloc[train],y.iloc[train]),1):
            actual_a,actual_b=train[a],train[b]
            assert not set(actual_a)&set(actual_b) and not set(actual_a)&set(valid) and not set(actual_b)&set(valid)
            inners.append(dict(inner_fold=i,fit_row_ids=list(map(int,X.index[actual_a])),
                               calibration_score_row_ids=list(map(int,X.index[actual_b]))))
        split_records.append(dict(repeat=int(spec.repeat),fold=int(spec.fold),
            outer_fit_row_ids=list(map(int,X.index[train])),outer_evaluation_row_ids=list(map(int,X.index[valid])),
            n_calibration=len(train),n_calibration_positive=int(y.iloc[train].sum()),inners=inners))
    dump(split_records,target/'nested_split_plan.json')
    dump(dict(design=design,source_files_start=source_start,execution_code_hashes=code_start,
              calibration_library_source_hashes=library_source,data=metadata,
              pipeline_parameters=describe_pipelines(bases)),target/'manifest_start.json')
    counters=dict(base_attempted=0,base_successful=0,base_failed=0,
                  sigmoid_attempted=0,sigmoid_successful=0,isotonic_attempted=0,isotonic_successful=0)
    fit_records=[]; warning_records=[]; error_records=[]; audits=[]; maps=[]
    max_raw_deviation=0.0
    probe=RssProbe(); probe.start(); started=time.perf_counter()

    def fit_base(key,spec,phase,inner,train,valid):
        nonlocal max_raw_deviation
        label=dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),phase=phase,inner_fold=inner)
        record=dict(**label,n_fit=len(train),n_fit_positive=int(y.iloc[train].sum()),
                    n_score=len(valid),n_score_positive=int(y.iloc[valid].sum()),
                    train_fingerprint=fingerprint(X.index[train]),score_fingerprint=fingerprint(X.index[valid]))
        counters['base_attempted']+=1
        caught=[]; begin=time.perf_counter()
        try:
            assert not set(train)&set(valid)
            assert not set(train)&set(spec.valid_positions) if phase=='inner' else True
            pipeline=clone(bases[key])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipeline.fit(X.iloc[train],y.iloc[train])
                probabilities=pipeline.predict_proba(X.iloc[valid])[:,list(pipeline.classes_).index(1)]
                response='decision_function' if hasattr(pipeline,'decision_function') else 'predict_proba'
                scores=(pipeline.decision_function(X.iloc[valid]) if response=='decision_function' else probabilities)
                sampler=pipeline.named_steps.get('sampler')
                if sampler is not None:
                    expected={int(c):int(n) for c,n in y.iloc[train].value_counts().items()}
                    assert sampler.sampling_class_counts_before_==expected and sampler.n_samples_before_==len(train)
                    assert sampler.n_fit_resample_calls_==1
                    audit=audit_resampling_fold(pipeline_key=key,model_name=key.split('__')[0],
                        strategy_name=key.split('__')[1],spec=spec,pipeline=pipeline,sampler_step=sampler,
                        y_train=y.iloc[train],y_validation=y.iloc[valid])
                    audit.update(phase=phase,inner_fold=inner)
                    audits.append(audit)
                    append_frame(pd.DataFrame([audit]),target/'sampling_audits.csv')
                    assert audit['audit_status']=='PASS',audit
            assert np.isfinite(scores).all() and np.isfinite(probabilities).all()
            assert ((probabilities>=0)&(probabilities<=1)).all()
            record.update(status='ok',score_method=response)
            counters['base_successful']+=1
            if phase=='outer':
                original=saved_step33[(saved_step33.pipeline==key)&(saved_step33.repeat==spec.repeat)&(saved_step33.fold==spec.fold)].set_index('row_index')
                deviation=float(np.max(np.abs(probabilities-original.loc[X.index[valid],'y_proba'].to_numpy())))
                record['max_abs_deviation_Step33_reference']=deviation
                max_raw_deviation=max(max_raw_deviation,deviation)
                assert deviation<=1e-12,('raw Step33 reproduction failed',key,deviation)
            return probabilities,np.asarray(scores,dtype=float),response
        except Exception as exc:
            counters['base_failed']+=1
            record.update(status='failed',error_type=type(exc).__name__,message=str(exc))
            error_records.append(dict(**label,error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            print('BASE FAILURE',label,str(exc),flush=True)
            return None
        finally:
            record['seconds']=time.perf_counter()-begin
            fit_records.append(record)
            # Fixed-schema final fit table is also checkpointed after each outer block.
            for item in caught:
                warning_records.append(dict(**label,category=item.category.__name__,message=str(item.message),
                                             filename=item.filename,lineno=int(item.lineno)))

    try:
        for key in design['selected']:
            print(time.strftime('%H:%M:%S'),key,'START',flush=True)
            for spec in outer:
                train=np.asarray(spec.train_positions); valid=np.asarray(spec.valid_positions)
                inner_score=np.full(len(train),np.nan); assigned=np.zeros(len(train),dtype=int)
                inner_fold=np.zeros(len(train),dtype=int)
                response=None
                for number,(a,b) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X.iloc[train],y.iloc[train]),1):
                    fitted=fit_base(key,spec,'inner',number,train[a],train[b])
                    if fitted is not None:
                        _,scores,response=fitted
                        inner_score[b]=scores; assigned[b]+=1; inner_fold[b]=number
                outer_result=fit_base(key,spec,'outer',0,train,valid)
                if np.isfinite(inner_score).all() and np.all(assigned==1):
                    inner_table=pd.DataFrame(dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),
                        inner_fold=inner_fold,row_index=X.index[train],y_true=y.iloc[train].to_numpy(),
                        calibration_score=inner_score,score_method=response))
                    append_frame(inner_table,target/'inner_oof_calibration_scores.csv')
                else:
                    error_records.append(dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),phase='calibration_guard',
                        error_type='IncompleteInnerCoverage',message='Calibration maps not fitted to incomplete inner scores',traceback='guard'))
                    continue
                if outer_result is None: continue
                raw,outer_scores,response=outer_result
                evaluated=[('uncalibrated_reference',raw)]
                for method in ['sigmoid','isotonic']:
                    counters[method+'_attempted']+=1
                    caught=[]
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            calibrator=(_SigmoidCalibration() if method=='sigmoid' else
                                IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip'))
                            calibrator.fit(inner_score,y.iloc[train].to_numpy())
                            calibrated=calibrator.predict(outer_scores)
                        assert np.isfinite(calibrated).all() and ((calibrated>=0)&(calibrated<=1)).all()
                        parameters=(dict(a=float(calibrator.a_),b=float(calibrator.b_)) if method=='sigmoid' else
                            dict(x_thresholds=calibrator.X_thresholds_.tolist(),y_thresholds=calibrator.y_thresholds_.tolist()))
                        maps.append(dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),method=method,
                            score_method=response,n_calibration=len(train),n_calibration_positive=int(y.iloc[train].sum()),
                            n_evaluation=len(valid),n_evaluation_positive=int(y.iloc[valid].sum()),
                            calibration_fingerprint=fingerprint(X.index[train]),evaluation_fingerprint=fingerprint(X.index[valid]),
                            n_distinct_inner_scores=int(len(np.unique(inner_score))),parameters=parameters))
                        evaluated.append((method,calibrated))
                        counters[method+'_successful']+=1
                    except Exception as exc:
                        error_records.append(dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),phase=method,
                            error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                    finally:
                        for item in caught:
                            warning_records.append(dict(pipeline=key,repeat=int(spec.repeat),fold=int(spec.fold),phase=method,
                                inner_fold=0,category=item.category.__name__,message=str(item.message),filename=item.filename,lineno=int(item.lineno)))
                for method,probs in evaluated:
                    append_frame(pd.DataFrame(dict(pipeline=key,method=method,repeat=int(spec.repeat),fold=int(spec.fold),
                        row_index=X.index[valid],y_true=y.iloc[valid].to_numpy(),y_proba=probs,
                        base_score=outer_scores,score_method=response)),target/'outer_predictions.csv')
                pd.DataFrame(fit_records).to_csv(target/'base_fit_records.csv',index=False)
                pd.DataFrame(warning_records,columns=['pipeline','repeat','fold','phase','inner_fold','category','message','filename','lineno']).to_csv(target/'warnings.csv',index=False)
                pd.DataFrame(error_records,columns=['pipeline','repeat','fold','phase','error_type','message','traceback']).to_csv(target/'errors.csv',index=False)
                dump(maps,target/'calibration_maps.json')
                print(time.strftime('%H:%M:%S'),key,'r',spec.repeat,'f',spec.fold,
                      'base fits',counters['base_successful'],'/',counters['base_attempted'],
                      'maps',counters['sigmoid_successful']+counters['isotonic_successful'],flush=True)
            print(time.strftime('%H:%M:%S'),key,'COMPLETE',flush=True)
    finally:
        memory=probe.stop()
        dump(dict(counters=counters,seconds=time.perf_counter()-started,memory=memory,
                  memory_scope='sampled process RSS for all nested fits and streamed predictions; 20ms probe',
                  max_raw_reference_deviation=max_raw_deviation),target/'execution_observations.json')
        pd.DataFrame(fit_records).to_csv(target/'base_fit_records.csv',index=False)
        pd.DataFrame(warning_records,columns=['pipeline','repeat','fold','phase','inner_fold','category','message','filename','lineno']).to_csv(target/'warnings.csv',index=False)
        pd.DataFrame(error_records,columns=['pipeline','repeat','fold','phase','error_type','message','traceback']).to_csv(target/'errors.csv',index=False)
        dump(maps,target/'calibration_maps.json')
    unexpected=[r for r in warning_records if classify_warning(r)[0]!='EXPECTED']
    source_end=source_file_hashes()
    end_code={k:sha(ROOT/k) for k in code_start}
    assert source_start==source_end and code_start==end_code
    unchanged=all(sha(ROOT/k)==v for k,v in design['original_artifact_hashes'].items())
    verdict=dict(overall_passed=(counters['base_successful']==900 and counters['base_failed']==0 and
        counters['sigmoid_successful']==150 and counters['isotonic_successful']==150 and
        not error_records and not unexpected and unchanged and len(audits)==450),
        counters=counters,unexpected_warning_count=len(unexpected),expected_warning_count=len(warning_records)-len(unexpected),
        error_count=len(error_records),sampling_audit_count=len(audits),
        sampling_audits_all_pass=all(r['audit_status']=='PASS' for r in audits),
        max_raw_reference_deviation=max_raw_deviation,original_artifacts_unchanged=unchanged,
        source_hashes_end=source_end,execution_code_hashes_end=end_code)
    dump(verdict,target/'execution_verification.json')
    print(json.dumps(verdict,indent=2),flush=True)
    if not verdict['overall_passed']: raise SystemExit(2)

if __name__=='__main__': main()
