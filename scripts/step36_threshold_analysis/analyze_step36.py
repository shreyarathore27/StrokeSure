"""Step 36: saved-prediction grid analysis and inner-only threshold policies.

No raw-data loader, estimator fitting, calibration fitting or test-set access.
"""
from pathlib import Path
import sys,os,json,hashlib,math,inspect
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (confusion_matrix,recall_score,precision_score,f1_score,
    matthews_corrcoef,balanced_accuracy_score,roc_auc_score,average_precision_score)
from threshold_metrics import GRID,POLICIES,METRICS,grid_metrics,metrics_from_counts,select_index

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN='imbalance_cv_step33_20261109T165943Z'
S33=ROOT/'results/imbalance_cv'/RUN
S34=ROOT/'results/step34_imbalance_analysis'
S35=ROOT/'results/step35_calibration'
NEST=S35/'nested'
METHODS=['original','sigmoid','isotonic']
COLORS=dict(zip(METHODS,['#666666','#0072B2','#D55E00']))
SHORT={'logistic_regression__none':'LR / none','logistic_regression__class_weight_balanced':'LR / weight',
 'logistic_regression__random_under_sampler':'LR / RUS','logistic_regression__adasyn':'LR / CatADASYN',
 'random_forest__none':'RF / none','svc_rbf__random_under_sampler':'SVM / RUS'}
CHECKS=[]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fp(ids): return hashlib.sha256(np.sort(np.asarray(list(ids),dtype=np.int64)).tobytes()).hexdigest()[:16]
def dump(obj,name): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def check(name,passed,**details):
    CHECKS.append(dict(check=name,passed=bool(passed),**details))
    if not passed: raise AssertionError((name,details))
def write(frame,name): frame.to_csv(OUT/(name+'.csv'),index=False,float_format='%.15g')
def original_files():
    excluded={'_plot_dependencies','_matplotlib_config','__pycache__'}
    return {str(p.relative_to(ROOT)):sha(p) for d in [S33,S34,S35] for p in d.rglob('*')
            if p.is_file() and not any(part in excluded for part in p.parts)}
def flatten_stats(frame,keys,columns):
    result=frame.groupby(keys,dropna=False)[columns].agg(['mean','std','min','max','count'])
    result.columns=['_'.join(k) for k in result.columns]
    return result.reset_index()
def table(frame,columns):
    text=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        text.append('| '+' | '.join(f'{row[k]:.4f}' if isinstance(row[k],(float,np.floating)) else str(row[k]) for k in columns)+' |')
    return '\n'.join(text)
def row_at(values,index):
    return {k:(v[index].item() if hasattr(v[index],'item') else v[index]) for k,v in values.items()}
def independent_count_check(y,p,values):
    decisions=np.asarray(p)[:,None]>=GRID[None,:]
    positive=np.asarray(y)==1
    tp=(decisions&positive[:,None]).sum(axis=0)
    fp_=(decisions&(~positive)[:,None]).sum(axis=0)
    return np.array_equal(tp,values['tp']) and np.array_equal(fp_,values['fp'])
def mapped_scores(method,key,scores,params):
    if method=='original':
        if key.startswith('logistic_regression__'): return expit(scores)
        if key=='random_forest__none': return np.asarray(scores,dtype=float)
        raise ValueError('Inner original SVM probabilities are unavailable')
    if method=='sigmoid': return expit(-(params['a']*scores+params['b']))
    return np.interp(scores,params['x_thresholds'],params['y_thresholds'])


def main():
    # Freeze all analysis rules before calculating threshold outcomes.
    check('191 predefined thresholds including 0.50',len(GRID)==191 and len(set(GRID))==191 and .5 in GRID)
    definition=dict(source_run=RUN,grid=GRID.tolist(),decision_rule='positive if y_proba >= threshold',
      grid_rationale='Fixed 0.000–0.100 in steps of 0.001, then 0.110–1.000 in steps of 0.010; finer low-risk resolution appropriate to rare outcomes; identical for every stream. Not outcome-adapted.',
      policies=dict(fixed_0_50='Predefined probability threshold 0.50; no selection.',
        sensitivity_ge_0_80='Hypothetical target: choose highest grid threshold achieving development sensitivity >=0.80.',
        specificity_ge_0_90='Hypothetical target: choose lowest grid threshold achieving development specificity >=0.90.',
        maximum_grid_mcc='Maximize signed development MCC on fixed grid; ties within 1e-12 choose highest threshold.'),
      targets_are_hypothetical=True,clinical_costs_available=False,
      descriptive_selection='Policy thresholds chosen on each evaluated repeat OOF outcome set are apparent/optimistically selected; not unbiased evaluations.',
      heldout_selection='Use Step35 outer-training inner OOF scores only, mapped by that outer fold calibration map; choose threshold without outer labels, then evaluate existing outer validation predictions.',
      calibration_selection_reuse='Calibration maps and thresholds may use the same outer-training labels; development metrics can overfit. Corresponding outer validation outcomes are never used for either fit or threshold choice.',
      original_svm_inner_limit='Step35 stored SVM decision margins, not each inner SVC internal Platt probability. No valid original-probability threshold selection for SVM can be reconstructed. Calibrated SVM streams are available via saved maps.',
      configuration_selection='Earlier development-data configuration selection is not nested; leakage-safe threshold evaluation is conditional and exploratory, not unbiased final model selection.',
      new_base_fits=0,new_calibrator_fits=0,locked_test_access=False,
      research_question='RQ7: How does changing the classification threshold affect sensitivity, specificity, precision, false positives and false negatives?',
      hypothesis='H5: Changing the classification threshold will produce meaningful sensitivity-specificity and false-positive/false-negative trade-offs.')
    dump(definition,'threshold_design.json')
    snapshot=original_files()
    m33=json.loads((S33/f'{RUN}_manifest.json').read_text())
    for key,entry in m33['files'].items():
        p=ROOT/entry['path'];check('Step33 artifact exists '+key,p.is_file())
        if entry['sha256']:check('Step33 hash '+key,sha(p)==entry['sha256'])
    for folder in [S34,S35]:
        for record in pd.read_csv(folder/'artifact_inventory.csv').itertuples():
            check('saved analysis artifact hash '+record.file,sha(folder/record.file)==record.sha256)
    for path,value in m33['source_files_end'].items(): check('experiment source unchanged '+path,sha(ROOT/path)==value)
    for folder in [S34,S35]: check('saved analysis verdict '+folder.name,json.loads((folder/'verification.json').read_text())['overall_passed'])
    raw=pd.read_csv(S33/f'{RUN}_oof_predictions.csv',float_precision='round_trip')
    raw['method']='original'
    outer=pd.read_csv(NEST/'outer_predictions.csv',float_precision='round_trip')
    selected=json.loads((S35/'design_and_provenance.json').read_text())['selected']
    fields=['pipeline','method','repeat','fold','row_index','y_true','y_proba']
    data=pd.concat([raw[fields],outer[outer.method!='uncalibrated_reference'][fields]],ignore_index=True)
    data['stream']=data.pipeline+'::'+data.method
    check('25 configurations and 37 unique streams',data.pipeline.nunique()==25 and data.stream.nunique()==37)
    check('no duplicate stream/repeat/patient',not data.duplicated(['stream','repeat','row_index']).any())
    check('valid probabilities',np.isfinite(data.y_proba).all() and data.y_proba.between(0,1).all())
    reference=raw[(raw.pipeline=='dummy_prior__none')&(raw.repeat==1)].set_index('row_index').y_true.sort_index()
    duplicate_max=0.
    for key in selected:
        ref=raw[raw.pipeline==key].set_index(['repeat','fold','row_index']).sort_index()
        duplicate=outer[(outer.pipeline==key)&(outer.method=='uncalibrated_reference')].set_index(['repeat','fold','row_index']).sort_index()
        check('Step35 reference identities '+key,ref.index.equals(duplicate.index))
        duplicate_max=max(duplicate_max,float(np.max(np.abs(ref.y_proba.to_numpy()-duplicate.y_proba.to_numpy()))))
    check('six duplicate reference streams verified not double counted',duplicate_max<=1e-12,max_abs_deviation=duplicate_max)
    keys=['stream','pipeline','method']
    stream_inventory=data[keys].drop_duplicates().sort_values(keys)
    stream_inventory['source']=np.where(stream_inventory.method=='original','Step33 original OOF','Step35 nested outer calibrated OOF')
    write(stream_inventory,'probability_stream_inventory')
    pooled_parts=[];fold_parts=[];rankings=[];apparent=[]
    for (stream,key,method,repeat),g in data.groupby(keys+['repeat']):
        check('4088/199 repeated cohort coverage',len(g)==4088 and int(g.y_true.sum())==199 and
              g.set_index('row_index').y_true.sort_index().equals(reference) and set(g.fold)==set(range(1,6)))
        values=grid_metrics(g.y_true,g.y_proba)
        check('independent Boolean counts for every repeat/grid point',independent_count_check(g.y_true,g.y_proba,values))
        frame=pd.DataFrame(values).assign(stream=stream,pipeline=key,method=method,repeat=int(repeat),scope='pooled_within_repeat')
        pooled_parts.append(frame)
        rankings.append(dict(stream=stream,pipeline=key,method=method,repeat=int(repeat),
            roc_auc=float(roc_auc_score(g.y_true,g.y_proba)),average_precision=float(average_precision_score(g.y_true,g.y_proba))))
        for policy in POLICIES:
            idx=select_index(values,policy)
            rec=dict(stream=stream,pipeline=key,method=method,repeat=int(repeat),policy=policy,
                     selection_scope='same_evaluation_repeat_apparent' if policy!='fixed_0.50' else 'predefined_fixed',
                     feasible=idx is not None)
            if idx is not None:rec.update(row_at(values,idx))
            apparent.append(rec)
        for threshold in [0.,.001,.05,.1,.5,1.]:
            idx=int(np.flatnonzero(GRID==threshold)[0]);pred=g.y_proba.to_numpy()>=threshold
            tn,fp_,fn,tp=map(int,confusion_matrix(g.y_true,pred,labels=[0,1]).ravel())
            check('sklearn confusion spot check',all(values[k][idx]==v for k,v in dict(tn=tn,fp=fp_,fn=fn,tp=tp).items()))
            independent=[recall_score(g.y_true,pred),precision_score(g.y_true,pred,zero_division=0),
                         f1_score(g.y_true,pred,zero_division=0),matthews_corrcoef(g.y_true,pred),balanced_accuracy_score(g.y_true,pred)]
            check('sklearn score spot check',np.allclose(independent,[values[k][idx] for k in ['sensitivity','precision','f1','mcc','balanced_accuracy']],atol=1e-12,rtol=0))
        for fold,gf in g.groupby('fold'):
            fv=grid_metrics(gf.y_true,gf.y_proba)
            check('independent Boolean counts every fold/grid point',independent_count_check(gf.y_true,gf.y_proba,fv))
            fold_parts.append(pd.DataFrame(fv).assign(stream=stream,pipeline=key,method=method,
                repeat=int(repeat),fold=int(fold),scope='individual_validation_fold'))
    pooled=pd.concat(pooled_parts,ignore_index=True);folds=pd.concat(fold_parts,ignore_index=True)
    apparent=pd.DataFrame(apparent);rankings=pd.DataFrame(rankings)
    check('expected pooled/fold grid row counts',len(pooled)==35335 and len(folds)==176675)
    write(pooled,'threshold_metrics_by_repeat');write(folds,'threshold_metrics_by_fold')
    summary=flatten_stats(pooled,keys+['threshold'],METRICS)
    write(summary,'threshold_summary_across_repeats')
    write(flatten_stats(folds,keys+['threshold'],METRICS),'threshold_summary_across_folds')
    write(rankings,'threshold_independent_roc_auc_ap_by_repeat')
    write(apparent,'apparent_operating_points_by_repeat')
    apparent_summary=flatten_stats(apparent,keys+['policy'],['threshold']+METRICS)
    write(apparent_summary,'apparent_operating_points_summary')
    # Golden fixed-.50 comparisons: do not alter earlier experiment metrics.
    prior=pd.read_csv(S34/'comparison_pooled_repeat_mean_sd.csv',float_precision='round_trip').set_index('pipeline')
    at50=summary[(summary.method=='original')&(summary.threshold==.5)]
    for row in at50.itertuples():
        prior_row=prior.loc[row.pipeline]
        check('fixed .50 reproduces Step34',all(abs(getattr(row,k+'_mean')-prior_row[old+'_mean'])<1e-12
              for k,old in [('sensitivity','recall_sensitivity'),('specificity','specificity'),('precision','precision'),
                           ('f1','f1'),('mcc','mcc'),('balanced_accuracy','balanced_accuracy')]))
    write(summary[summary.threshold==.5],'fixed_050_reference_summary')
    # Comparisons at the same probability cutoff are paired by repeat.
    comparisons=[]
    for key in selected:
        base=pooled[(pooled.pipeline==key)&(pooled.method=='original')].set_index(['repeat','threshold'])
        for method in ['sigmoid','isotonic']:
            part=pooled[(pooled.pipeline==key)&(pooled.method==method)].set_index(['repeat','threshold'])
            for (repeat,t),row in part.iterrows():
                comparisons.append(dict(pipeline=key,method=method,repeat=int(repeat),threshold=float(t),
                    **{'delta_'+m:float(row[m]-base.loc[(repeat,t),m]) for m in METRICS}))
    comparisons=pd.DataFrame(comparisons)
    write(comparisons,'paired_original_calibrated_threshold_differences')
    # Genuinely held-out threshold policies: choices use inner scores/labels only.
    inner=pd.read_csv(NEST/'inner_oof_calibration_scores.csv',float_precision='round_trip')
    maps=json.loads((NEST/'calibration_maps.json').read_text())
    plans=json.loads((NEST/'nested_split_plan.json').read_text())
    plan={(r['repeat'],r['fold']):r for r in plans}
    map_index={(m['pipeline'],m['method'],m['repeat'],m['fold']):m for m in maps}
    eligible=[(key,m) for key in selected for m in METHODS
              if not (key=='svc_rbf__random_under_sampler' and m=='original')]
    check('17 inner-supported streams',len(eligible)==17)
    inner_grids=[];heldout=[];provenance=[]
    for key,method in eligible:
        for (repeat,fold),r in plan.items():
            dev=inner[(inner.pipeline==key)&(inner.repeat==repeat)&(inner.fold==fold)]
            train_ids=set(r['outer_fit_row_ids']);eval_ids=set(r['outer_evaluation_row_ids'])
            check('threshold development/evaluation disjoint',not train_ids&eval_ids and
                  set(dev.row_index)==train_ids and len(dev)==len(train_ids) and dev.row_index.is_unique)
            check('inner labels match original cohort',np.array_equal(dev.y_true.to_numpy(),reference.loc[dev.row_index].to_numpy()))
            params=None
            if method!='original':
                saved=map_index[(key,method,repeat,fold)]
                check('calibration map excludes outer outcomes',saved['calibration_fingerprint']==fp(train_ids)
                      and saved['evaluation_fingerprint']==fp(eval_ids))
                params=saved['parameters']
            dev_p=mapped_scores(method,key,dev.calibration_score.to_numpy(),params)
            dv=grid_metrics(dev.y_true,dev_p)
            check('inner grid count verification',independent_count_check(dev.y_true,dev_p,dv))
            inner_grids.append(pd.DataFrame(dv).assign(pipeline=key,method=method,repeat=repeat,fold=fold,
                                                      scope='outer_training_inner_score_development'))
            # Freeze choices before selecting or reading corresponding evaluation outcomes.
            choices={policy:select_index(dv,policy) for policy in POLICIES}
            thresholds={policy:(float(GRID[i]) if i is not None else None) for policy,i in choices.items()}
            eval_part=data[(data.pipeline==key)&(data.method==method)&(data.repeat==repeat)&(data.fold==fold)]
            check('heldout stream row identities',set(eval_part.row_index)==eval_ids and len(eval_part)==len(eval_ids))
            if method=='original':
                raw_base=outer[(outer.pipeline==key)&(outer.repeat==repeat)&(outer.fold==fold)&
                               (outer.method=='uncalibrated_reference')]
                rebuilt=mapped_scores(method,key,raw_base.base_score.to_numpy(),None)
                check('original inner score-to-probability rule validated on saved outer scores',
                      np.max(np.abs(rebuilt-raw_base.y_proba.to_numpy()))<1e-12)
            for policy,index in choices.items():
                record=dict(stream=key+'::'+method,pipeline=key,method=method,repeat=repeat,fold=fold,
                    policy=policy,selection_scope='inner_development_only' if policy!='fixed_0.50' else 'predefined_fixed',
                    feasible=index is not None,selected_threshold=thresholds[policy],
                    n_development=len(dev),development_positive=int(dev.y_true.sum()),
                    development_fingerprint=fp(train_ids),evaluation_fingerprint=fp(eval_ids),
                    map_fit_uses_outer_evaluation=False)
                if index is None:
                    record['infeasibility_reason']='No feasible threshold in predefined grid; no observations discarded'
                    heldout.append(record);continue
                selected_threshold=thresholds[policy]
                # Replay policy definition independently to check tie-break and constraints.
                if policy=='sensitivity_ge_0.80':
                    check('sensitivity policy rule',dv['sensitivity'][index]>=.8 and
                          all(dv['sensitivity'][index+1:]<.8))
                if policy=='specificity_ge_0.90':
                    check('specificity policy rule',dv['specificity'][index]>=.9 and
                          all(dv['specificity'][:index]<.9))
                if policy=='maximum_grid_mcc':
                    check('MCC policy rule',abs(dv['mcc'][index]-np.max(dv['mcc']))<=1e-12 and
                          index==np.flatnonzero(np.isclose(dv['mcc'],np.max(dv['mcc']),atol=1e-12,rtol=0))[-1])
                ev=grid_metrics(eval_part.y_true,eval_part.y_proba,np.asarray([selected_threshold]))
                record.update({k:v[0].item() for k,v in ev.items()})
                record.update(development_sensitivity=float(dv['sensitivity'][index]),
                    development_specificity=float(dv['specificity'][index]),development_mcc=float(dv['mcc'][index]))
                record['outer_constraint_achieved']=(bool(record['sensitivity']>=.8) if policy=='sensitivity_ge_0.80'
                    else bool(record['specificity']>=.9) if policy=='specificity_ge_0.90' else None)
                # Independent direct replay of the selected evaluation threshold.
                pred=eval_part.y_proba.to_numpy()>=selected_threshold
                tn,fp_,fn,tp=map(int,confusion_matrix(eval_part.y_true,pred,labels=[0,1]).ravel())
                check('heldout selected-threshold confusion replay',all(record[k]==v for k,v in dict(tn=tn,fp=fp_,fn=fn,tp=tp).items()))
                heldout.append(record)
                provenance.append({k:record[k] for k in ['stream','pipeline','method','repeat','fold','policy',
                    'selection_scope','selected_threshold','n_development','development_positive',
                    'development_fingerprint','evaluation_fingerprint','map_fit_uses_outer_evaluation']})
    heldout=pd.DataFrame(heldout)
    check('1700 inner-supported policy/fold evaluations retained',len(heldout)==1700)
    write(pd.concat(inner_grids,ignore_index=True),'inner_development_threshold_metrics')
    write(heldout,'inner_selected_policy_outer_fold_results')
    write(pd.DataFrame(provenance),'threshold_selection_provenance')
    check('all planned policies feasible (not a requirement of heldout target achievement)',heldout.feasible.all())
    policy_repeats=[]
    for k,g in heldout.groupby(keys+['policy','repeat']):
        stream,key,method,policy,repeat=k
        check('five outer folds per selected policy repeat',len(g)==5 and set(g.fold)==set(range(1,6)))
        counts=metrics_from_counts(*[np.asarray([int(g[m].sum())]) for m in ['tn','fp','fn','tp']])
        record=dict(stream=stream,pipeline=key,method=method,policy=policy,repeat=int(repeat),
            scope='pooled_within_repeat_with_fold_specific_inner_selected_thresholds',
            threshold_mean=float(g.selected_threshold.mean()),threshold_sd=float(g.selected_threshold.std(ddof=1)),
            threshold_min=float(g.selected_threshold.min()),threshold_max=float(g.selected_threshold.max()),
            **{m:v[0].item() for m,v in counts.items()})
        check('selected policy repeat denominator',record['n']==4088 and record['n_positive']==199)
        policy_repeats.append(record)
    policy_repeats=pd.DataFrame(policy_repeats)
    policy_summary=flatten_stats(policy_repeats,keys+['policy'],METRICS+['threshold_mean','threshold_sd','threshold_min','threshold_max'])
    write(policy_repeats,'inner_selected_policy_pooled_repeat_results')
    write(policy_summary,'inner_selected_policy_summary')
    write(flatten_stats(heldout,keys+['policy'],METRICS+['selected_threshold']),'inner_selected_policy_fold_variability')
    achievement=[]
    for k,g in heldout[heldout.policy.isin(POLICIES[1:3])].groupby(keys+['policy']):
        achievement.append(dict(zip(keys+['policy'],k))|dict(n_outer_folds=len(g),
            n_outer_folds_meeting_target=int(g.outer_constraint_achieved.sum()),
            proportion_outer_folds_meeting_target=float(g.outer_constraint_achieved.mean())))
    write(pd.DataFrame(achievement),'outer_constraint_achievement')
    exclusions=[]
    eligible_streams={k+'::'+m for k,m in eligible}
    for row in stream_inventory.itertuples():
        if row.stream not in eligible_streams:
            exclusions.append(dict(stream=row.stream,reason=('Inner SVC internal Platt probabilities not saved; margins cannot reconstruct probability thresholds.'
                if row.pipeline=='svc_rbf__random_under_sampler' else 'No corresponding Step35 saved inner score stream for this configuration.')))
    write(pd.DataFrame(exclusions),'selected_policy_scope_exclusions')
    # Figures use only the six Step35 configuration panels, tables cover all streams.
    sys.dont_write_bytecode=True
    os.environ['MPLCONFIGDIR']=str(OUT/'_matplotlib_config')
    sys.path.append(str(S34/'_plot_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    handles=[Line2D([0],[0],color=COLORS[m],label=m) for m in METHODS]
    def save(fig,name,caption):
        fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
        figures.append(dict(file=name+'.png',caption=caption))
    for number,x,y,title in [
        (1,'specificity','sensitivity','Sensitivity versus specificity across the fixed threshold grid'),
        (2,'sensitivity','precision','Precision versus recall across the fixed threshold grid'),
        (3,'fp','fn','False-positive versus false-negative burden across thresholds')]:
        fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
        for ax,key in zip(axes.flat,selected):
            for method in METHODS:
                g=summary[(summary.pipeline==key)&(summary.method==method)].sort_values('threshold')
                if y=='precision':
                    defined=pooled[(pooled.pipeline==key)&(pooled.method==method)].groupby('threshold').precision_defined.all()
                    g=g[g.threshold.map(defined)]
                ax.plot(g[x+'_mean'],g[y+'_mean'],color=COLORS[method],linewidth=1.3)
                ref=g[g.threshold==.5]
                if len(ref): ax.scatter(ref[x+'_mean'],ref[y+'_mean'],color=COLORS[method],marker='s',s=25,zorder=4)
            ax.set_title(SHORT[key]);ax.grid(alpha=.15)
            ax.set_xlabel({'specificity':'Specificity','sensitivity':'Recall / sensitivity','fp':'Mean FP per 4,088-row repeat'}[x])
            ax.set_ylabel({'sensitivity':'Sensitivity','precision':'Precision / PPV','fn':'Mean FN per 4,088-row repeat'}[y])
        fig.legend(handles=handles+[Line2D([0],[0],marker='s',linestyle='',color='black',label='Fixed 0.50')],loc='lower center',ncol=4,frameon=False)
        fig.suptitle(title);fig.tight_layout(rect=(0,.05,1,.95))
        save(fig,f'figure0{number}_{x}_{y}','Means over five separate pooled repeats; each repeat has 4,088 patients, 199 events and 3,889 nonevents. Squares mark 0.50. Curves are descriptive grid sweeps, not selected-threshold performance. Undefined PPV points are not drawn, but are retained and flagged in tables.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        for method in METHODS:
            g=summary[(summary.pipeline==key)&(summary.method==method)].sort_values('threshold')
            ax.plot(g.threshold,g.mcc_mean,color=COLORS[method],linestyle='-')
            ax.plot(g.threshold,g.f1_mean,color=COLORS[method],linestyle='--')
        ax.axvline(.5,color='#aaaaaa',linestyle=':',linewidth=.8);ax.set_title(SHORT[key]);ax.set_xlabel('Probability threshold');ax.set_ylabel('Mean score');ax.grid(alpha=.15)
    fig.legend(handles=handles+[Line2D([0],[0],color='black',label='MCC',linestyle='-'),Line2D([0],[0],color='black',label='F1',linestyle='--')],loc='lower center',ncol=5,frameon=False)
    fig.suptitle('Threshold versus F1 and MCC: descriptive curves, not clinical optimization')
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,'figure04_threshold_f1_mcc','Unchanged probability scores; threshold changes classifications only. Score maxima on these evaluated OOF data are apparent and not unbiased estimates of a selected policy.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        for method in METHODS:
            g=summary[(summary.pipeline==key)&(summary.method==method)&(summary.threshold>0)].sort_values('threshold')
            ax.plot(g.threshold,g.sensitivity_mean,color=COLORS[method])
            ax.fill_between(g.threshold,g.sensitivity_min,g.sensitivity_max,color=COLORS[method],alpha=.1)
        ax.set_xscale('log');ax.axvline(.5,color='#999999',linestyle=':',linewidth=.8)
        ax.set_title(SHORT[key]);ax.set_xlabel('Probability threshold (log axis)');ax.set_ylabel('Sensitivity');ax.grid(alpha=.15)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False);fig.suptitle('Original versus calibrated streams: numeric threshold shifts')
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,'figure05_original_calibrated_thresholds','Lines are pooled-repeat means; shading is the five-repeat min–max range, not CI. Threshold zero (all positive) is retained in tables but omitted from the log axis.')
    styles={POLICIES[1]:'-',POLICIES[2]:'--',POLICIES[3]:':'}
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        for method in METHODS:
            for policy,style in styles.items():
                g=policy_repeats[(policy_repeats.pipeline==key)&(policy_repeats.method==method)&(policy_repeats.policy==policy)].sort_values('repeat')
                if len(g):ax.plot(g.repeat,g.threshold_mean,color=COLORS[method],linestyle=style,marker='o',markersize=3)
        ax.set_title(SHORT[key]);ax.set_xticks(range(1,6));ax.set_xlabel('CV repeat, seed sequence 42');ax.set_ylabel('Mean of five inner-selected thresholds');ax.grid(alpha=.15)
    policy_handles=[Line2D([0],[0],color='black',linestyle=s,label=p) for p,s in styles.items()]
    fig.legend(handles=handles+policy_handles,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Threshold stability: inner-selected outer-fold policies')
    fig.tight_layout(rect=(0,.09,1,.95));save(fig,'figure06_threshold_repeat_stability','Each repeat point averages five distinct outer-fold thresholds selected on inner data. Means are not a single deployed threshold. Per-fold choices and ranges are preserved. Original SVM probability policy is unavailable; no replacement values are inserted.')
    markers=dict(zip(POLICIES,['s','o','^','D']))
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        for method in METHODS:
            for policy,marker in markers.items():
                g=policy_summary[(policy_summary.pipeline==key)&(policy_summary.method==method)&(policy_summary.policy==policy)]
                if len(g):ax.scatter(g.fp_mean,g.fn_mean,marker=marker,color=COLORS[method],s=45)
        ax.set_title(SHORT[key]);ax.set_xlabel('Mean FP per 4,088-row repeat');ax.set_ylabel('Mean FN per 4,088-row repeat');ax.grid(alpha=.15)
    fig.legend(handles=handles+[Line2D([0],[0],marker=marker,linestyle='',color='black',label=p) for p,marker in markers.items()],loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Inner-selected policies: observed outer-validation FP/FN burden')
    fig.tight_layout(rect=(0,.1,1,.95));save(fig,'figure07_heldout_policy_fp_fn','Policies were selected without outer targets; evaluation is paired by outer fold. Hypothetical inner constraints need not be achieved on validation. Configuration selection from earlier development CV remains exploratory.')
    # Report exact numerical findings and scope distinctions.
    focus='logistic_regression__class_weight_balanced'
    focus_grid=summary[(summary.pipeline==focus)&(summary.method=='original')].set_index('threshold')
    base=focus_grid.loc[.5];low=focus_grid.loc[.2];high=focus_grid.loc[.8]
    focus_policy=policy_summary[(policy_summary.pipeline==focus)&(policy_summary.policy=='sensitivity_ge_0.80')]
    reporting=policy_summary.copy()
    columns=['pipeline','method','policy','threshold_mean_mean','sensitivity_mean','specificity_mean',
             'precision_mean','f1_mean','mcc_mean','fp_mean','fn_mean']
    apparent_columns=['pipeline','method','policy','threshold_mean','threshold_std','sensitivity_mean',
                      'specificity_mean','precision_mean','mcc_mean','fp_mean','fn_mean']
    sections=[
        '# StrokeSure Step 36: classification thresholds and decision trade-offs',
        'Analysis date: 2026-11-10 (Asia/Calcutta). Saved Step 33–35 development/CV artifacts only. No model/calibrator fitting, raw-data loading or locked-test access was performed.',
        '**RQ7: How does changing the classification threshold affect sensitivity, specificity, precision, false positives and false negatives?**',
        '**H5: Changing the classification threshold will produce meaningful sensitivity-specificity and false-positive/false-negative trade-offs.**',
        '## Data, probability streams and predefined grid',
        f'The analysis includes 25 configurations and 37 unique probability streams: 25 original Step 33 streams and 12 additional Step 35 sigmoid/isotonic streams across six exploratory configurations. Six re-fitted Step 35 uncalibrated references duplicate Step 33 and are verified but not double-counted (maximum probability deviation {duplicate_max:.3g}). Positive-class field is y_proba; patient identity is row_index, with explicit repeat/fold IDs. SVC originals already include internal Platt probability estimation; original means no additional Step 35 calibration.',
        'The 191-point grid was frozen before calculating outcomes: 0.000–0.100 in increments of 0.001, then 0.110–1.000 in increments of 0.010. All streams use the same grid. Predictions use p ≥ threshold, matching the original evaluation rule. The 0.50 reference is retained. At threshold zero all observations are positive. At threshold one, exact-one isotonic probabilities may still predict positives; an all-negative endpoint is not forced.',
        'Each pooled repeat has 4,088 observations, 199 events and 3,889 nonevents. All five repeats revisit the same cohort. Reported mean FP/FN counts are per 4,088-row repeat, not totals over independent patients. Individual-fold tables retain 817/818 rows and 39/40 events. Mean/SD/min/max across five repeats and across 25 folds are separate, descriptive and not interchangeable. No naive independent-fold confidence interval or significance test was used.',
        'There are 35,335 pooled-repeat grid evaluations and 176,675 individual-fold grid evaluations (212,010 evaluation-grid rows). An additional 81,175 inner-development grid rows support threshold-policy provenance. These are metric-row counts, not patients, independent samples or model fits. Apparent policies provide 740 configuration/method/repeat/policy records. Inner-selected evaluation provides 1,700 policy/outer-fold records and 340 pooled-repeat policy records.',
        'Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); PPV = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN); MCC uses the standard four-cell formula; balanced accuracy = (sensitivity+specificity)/2. FPR=1−specificity, FNR=1−sensitivity, predicted-positive fraction=(TP+FP)/N. No-positive-prediction PPV and zero-denominator MCC follow sklearn’s zero convention but are explicitly flagged. Undefined PPV is not drawn as an observed zero-PPV estimate in PR plots.',
        'ROC-AUC and AP (average_precision_score, not trapezoidal PR-AUC) are computed once per unchanged stream/repeat and saved separately. Moving a threshold changes decisions, not the ranking of fixed scores. Calibration changes scores and fold-specific maps or isotonic ties can change pooled ranking, as documented in Step 35; that is a different operation.',
        '## Numerical trade-offs on original scores',
        f'For original class-weighted Logistic Regression, threshold 0.20 gives mean sensitivity {low.sensitivity_mean:.4f}, specificity {low.specificity_mean:.4f}, precision {low.precision_mean:.4f}, FP {low.fp_mean:.1f} and FN {low.fn_mean:.1f} per repeat. At 0.50 these are {base.sensitivity_mean:.4f}, {base.specificity_mean:.4f}, {base.precision_mean:.4f}, FP {base.fp_mean:.1f}, FN {base.fn_mean:.1f}. At 0.80 they are {high.sensitivity_mean:.4f}, {high.specificity_mean:.4f}, {high.precision_mean:.4f}, FP {high.fp_mean:.1f}, FN {high.fn_mean:.1f}. The lower threshold detects more events while increasing the false-positive burden; it is not automatically better.',
        'Uncorrected Logistic Regression and Random Forest have useful ROC-AUC yet zero recall at 0.50 because their saved positive-risk probabilities do not cross that decision cutoff for true events. A ranking metric does not require positive predictions at 0.50. Lowering a cutoff can retrieve events without improving the underlying ROC-AUC/AP, and introduces false positives.',
        '## Predefined hypothetical operating policies',
        '- Fixed 0.50: no outcome-based threshold choice.',
        '- Sensitivity-constrained: largest grid threshold with development sensitivity ≥0.80, limiting FP while meeting the stated grid constraint.',
        '- Specificity-constrained: smallest grid threshold with development specificity ≥0.90, maximizing available sensitivity along the grid subject to the constraint.',
        '- MCC-oriented: largest signed development MCC on the grid; ties within 1e-12 choose the highest threshold.',
        'The 0.80/0.90 targets are hypothetical research scenarios, not established clinical requirements. There are no externally justified misclassification costs, utility weights or deployment constraints, so no optimal clinical threshold is claimed. Grid resolution and tie rules affect selected cutoffs. Infeasible policies would be retained and flagged rather than silently omitted.',
        '## Apparent selection on evaluated OOF data: deliberately optimistic',
        'The apparent-policy tables choose each repeat’s threshold using that same evaluated OOF outcome set. Their resulting performance is optimistically selected and is unsuitable for an unbiased selected-threshold claim. These tables describe achievable trade-offs within the observed development data only. They must not be substituted for the inner-selected results below.',
        table(apparent_summary[apparent_summary.pipeline.isin(selected)],apparent_columns),
        '## Threshold selection using saved inner development data',
        'No additional modelling was necessary. Step 35 retained one inner OOF score per outer-training row and each outer fold’s fitted calibration maps. For LR original probabilities, expit(decision_function) reconstructs the probability; RF original scores are already predict_proba. For sigmoid/isotonic streams, the corresponding outer fold map transforms the saved inner scores. These transformations are verified on the saved outer records or map parameters.',
        'Each threshold is chosen solely from the 3,270/3,271 inner development score rows (159/160 events). Choices are frozen before the corresponding outer outcomes are used for evaluation. The calibration map was also fitted only on these outer-training rows. Threshold selection and calibration may share training targets; apparent inner criteria can overfit, but the outer validation observations remain disjoint from both stages. Exact row roles and fingerprints are checked against Step 35’s saved split plan. All method comparisons use the same untouched outer rows.',
        'Inner OOF scores came from base models fitted on smaller inner-training subsets, whereas outer scores came from the full outer-training refit. That cross-fit/refit score-distribution mismatch can affect a transferred threshold and is assessed by its actual outer-validation performance, not assumed away.',
        'There are 17 supported inner-selection streams: five original LR/RF streams and all 12 calibrated streams. The raw SVM stream has only inner decision margins, not the inner SVC internal Platt probabilities. It cannot support a reconstructed probability-threshold selection and is explicitly excluded from selected-policy evaluation; its predefined-grid descriptive results remain available. Other original configurations without saved Step 35 inner scores likewise have no selected-policy evaluation. No unsupported margin-to-probability mapping, replacement threshold or unnecessary retraining was introduced.',
        'These evaluations avoid threshold-selection leakage relative to their outer validation folds. However, the six configurations were previously chosen using the same development cohort in Step 34/35. This is conditional exploratory policy evaluation, not fully unbiased evaluation of an entire model-selection procedure. It does not justify selecting a final model, calibrator or threshold on these outer results.',
        '## Inner-selected policies: outer-validation results',
        'Threshold mean below is an average of fold-specific selected thresholds, not a single threshold applied to all validation patients. Metrics/counts are calculated by pooling the five disjoint validation-fold confusion matrices within each repeat, then summarizing five repeats. Inner sensitivity/specificity constraints need not hold on outer data; no outer fold violating a target was discarded.',
        table(reporting,columns),
        '### Calibration changes the numerical operating threshold',
        table(focus_policy,columns),
        'The same hypothetical sensitivity target can require substantially different numerical cutoffs after calibration. This reflects probability-scale changes and does not itself imply improved ranking or better clinical decisions. The 0.50 cutoff on a recalibrated natural-prevalence risk scale is not interchangeable with 0.50 on class-weighted or resampled scores.',
        '## Repeat stability and constraint achievement',
        'Apparent threshold SD/ranges across repeats and inner-selected per-fold thresholds are both saved with separate scope labels. The repeat-stability figure shows means of five distinct inner-selected thresholds per repeat; there is no single pooled-repeat threshold in that policy. Per-fold and repeat ranges reflect split variability under seed sequence 42, not independent seed experiments. Variability includes finite event counts, bin/grid resolution, ties and changes in fitted calibration maps.',
        'Threshold SDs on calibrated and original score scales are not directly comparable evidence of improved robustness. A nonlinear probability transformation changes the units of the cutoff. Examine outer decision rates, counts, target achievement and their variability as well.',
        'Illustrative cutoff stability for the sensitivity ≥0.80 policy in class-weighted LR: threshold_mean_std is SD of the five repeat-level means, threshold_sd_mean is mean within-repeat SD of five fold-specific choices, and threshold_min_min/threshold_max_max span all 25 fold-specific choices. All configurations and policies are retained in the corresponding CSVs.',
        table(focus_policy,['method','threshold_mean_mean','threshold_mean_std','threshold_sd_mean','threshold_min_min','threshold_max_max']),
        f'For original class-weighted LR, the inner-MCC-oriented rule achieved outer mean MCC {policy_summary[(policy_summary.pipeline==focus)&(policy_summary.method=="original")&(policy_summary.policy=="maximum_grid_mcc")].iloc[0].mcc_mean:.4f}, compared with {policy_summary[(policy_summary.pipeline==focus)&(policy_summary.method=="original")&(policy_summary.policy=="fixed_0.50")].iloc[0].mcc_mean:.4f} at the predefined 0.50 cutoff. Optimizing an inner criterion does not guarantee an outer improvement; this result is retained without adjusting the policy.',
        table(pd.DataFrame(achievement),['stream','policy','n_outer_folds','n_outer_folds_meeting_target','proportion_outer_folds_meeting_target']),
        '## Interpretation of RQ7 and H5',
        'Observed grid curves demonstrate sensitivity/specificity and FP/FN trade-offs as decision thresholds change, compatible with H5 in a descriptive sense. The measured burdens and paired calibration comparisons answer RQ7 for this internal development cohort. No formal significance test or prespecified clinical materiality test was performed; H5 is not statistically established. Raising sensitivity can produce many more false alarms in a low-prevalence cohort, so high recall alone is insufficient to justify a policy.',
        '## Handover: which results avoid threshold-selection bias?',
        '1. Every frozen grid cutoff (including 0.50) was predefined, so its individual performance estimate does not reuse outcomes to choose that cutoff. Choosing a preferred cutoff after viewing the curves would introduce selection optimism.',
        '2. The 17-stream inner-selected policy evaluations choose thresholds without corresponding outer targets. They are valid held-out evaluations of the stated conditional threshold-selection rules. Their fold-specific cutoffs, full provenance and outer target failures are retained.',
        '3. The apparent best/constrained thresholds chosen from evaluated repeat OOF outcomes are explicitly optimistic and must not be reported as unbiased policy performance.',
        '4. No inner-selected original-SVM probability policy is available; its missing inner Platt probabilities are a documented reconstruction blocker. No threshold has been chosen for clinical deployment or final held-out testing.',
        '5. Future decisions require agreed clinical costs/utility, a prespecified development-selection strategy and independent final evaluation. Keep the test partition locked until final model/calibrator/threshold decisions are fixed. No SHAP or final model evaluation was performed.',
        '## Figures',
    ]
    for fig in figures:sections += [f'![{fig["caption"]}]({fig["file"]})',fig['caption']]
    sections += ['## Verification, inventory and reproduction',
        'All earlier artifact inventories and original source hashes were checked before analysis. Independent Boolean confusion counts verify every pooled-repeat and fold/grid row, with sklearn score spot checks, Step34 fixed-.50 reconciliation, inner development/evaluation disjointness, map-role verification, independent policy tie-rule replay and selected-threshold confusion replay. No threshold record or failed outer constraint was filtered to improve results.',
        'Full outputs are indexed in `artifact_inventory.csv`. Key tables separate per-repeat versus fold grid metrics, original/calibrated paired differences, apparent selected policies, actual inner-selected outer-fold/repeat policies, constraint achievement, threshold provenance and scope exclusions. `threshold_independent_roc_auc_ap_by_repeat.csv` contains ranking metrics once per fixed stream/repeat. `threshold_design.json` records the frozen grid, constraints, tie rules and scope. All scripts and data outputs are local to Step 36.',
        'Reproduce without fitting: `.\\.venv\\Scripts\\python.exe results/step36_threshold_analysis/analyze_step36.py`. Plot libraries are read from the isolated Step 34 plotting dependencies; Step 36 has its own plotting configuration directory. Original Step 33–35 files are never overwritten. No Git commit or push was made.',
    ]
    check('all original artifacts unchanged',snapshot==original_files())
    report=OUT/'scientific_threshold_report.md'
    report.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    dump(figures,'figure_captions.json')
    verdict=dict(overall_passed=all(c['passed'] for c in CHECKS),n_checks=len(CHECKS),
                 n_passed=sum(c['passed'] for c in CHECKS),n_failed=sum(not c['passed'] for c in CHECKS),checks=CHECKS)
    dump(verdict,'verification.json')
    with report.open('a',encoding='utf-8') as handle:handle.write(f'\n\nFinal numerical/provenance checks: **{verdict["n_passed"]}/{verdict["n_checks"]} PASS**, {verdict["n_failed"]} failed. All original artifact hashes remain unchanged.\n')
    dump(dict(original_artifact_hashes=snapshot,source_run=RUN,configurations=25,unique_streams=37,
        original_streams=25,calibrated_streams=12,threshold_count=len(GRID),
        evaluation_grid_rows=len(pooled)+len(folds),pooled_repeat_grid_rows=len(pooled),individual_fold_grid_rows=len(folds),
        inner_development_grid_rows=sum(len(g) for g in inner_grids),apparent_policy_records=len(apparent),
        heldout_policy_streams=len(eligible),heldout_policy_fold_records=len(heldout),heldout_policy_repeat_records=len(policy_repeats),
        new_estimator_fits=0,new_calibrator_fits=0,
        analysis_code_hashes={p.name:sha(p) for p in OUT.glob('*.py')},
        numpy=np.__version__,pandas=pd.__version__,scikit_learn=sys.modules['sklearn'].__version__,matplotlib=matplotlib.__version__,
        verification={k:v for k,v in verdict.items() if k!='checks'}),'analysis_manifest.json')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and '_matplotlib_config' not in p.parts and '__pycache__' not in p.parts and p.name not in ['artifact_inventory.csv','execution.log']:
            inventory.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(inventory),'artifact_inventory')
    print(json.dumps({k:v for k,v in verdict.items() if k!='checks'},indent=2),flush=True)
    print('PRIMARY ORIGINAL GRID EXAMPLES',flush=True)
    print(focus_grid.loc[[.05,.1,.2,.5,.8],['sensitivity_mean','specificity_mean','precision_mean','fp_mean','fn_mean']].to_string(),flush=True)
    print('INNER SELECTED SENSITIVITY POLICY',flush=True);print(focus_policy[columns].to_string(index=False),flush=True)
    print('ALL INNER POLICY SUMMARY',flush=True);print(policy_summary[columns].to_string(index=False),flush=True)

if __name__=='__main__': main()
