"""Step 37: paired exploratory corrected-CV and conditional patient bootstrap."""
from pathlib import Path
import sys,os,json,hashlib,time,math
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (roc_auc_score,average_precision_score,brier_score_loss,
    recall_score,confusion_matrix,matthews_corrcoef)
from statistical_methods import METRICS,corrected_cv,holm_adjust,WeightedPredictionScorer

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
RUN='imbalance_cv_step33_20261109T165943Z'
S33=ROOT/'results/imbalance_cv'/RUN
S34=ROOT/'results/step34_imbalance_analysis';S35=ROOT/'results/step35_calibration';S36=ROOT/'results/step36_threshold_analysis'
BOOTSTRAPS=5000;BATCH=100;BOOT_SEED=37042
def stream(key,method='original'):return key+'::'+method
CONTRASTS=[
 dict(id='C1',a=stream('logistic_regression__class_weight_balanced'),b=stream('logistic_regression__none'),label='LR weighting minus LR none'),
 dict(id='C2',a=stream('logistic_regression__random_under_sampler'),b=stream('logistic_regression__none'),label='LR RUS minus LR none'),
 dict(id='C3',a=stream('random_forest__none'),b=stream('logistic_regression__none'),label='RF none minus LR none'),
 dict(id='C4',a=stream('logistic_regression__adasyn'),b=stream('logistic_regression__smote'),label='LR CatADASYN minus LR SMOTENC'),
 dict(id='C5',a=stream('logistic_regression__class_weight_balanced','sigmoid'),b=stream('logistic_regression__class_weight_balanced'),label='LR weighting sigmoid minus reference'),
 dict(id='C6',a=stream('logistic_regression__class_weight_balanced','isotonic'),b=stream('logistic_regression__class_weight_balanced'),label='LR weighting isotonic minus reference'),
 dict(id='C7',a=stream('random_forest__none','sigmoid'),b=stream('random_forest__none'),label='RF none sigmoid minus reference'),
 dict(id='C8',a=stream('random_forest__none','isotonic'),b=stream('random_forest__none'),label='RF none isotonic minus reference')]
INTERACTIONS=[dict(id='I1',strategy='class_weight_balanced',label='Weighting effect in LR minus weighting effect in RF'),
              dict(id='I2',strategy='random_under_sampler',label='RUS effect in LR minus RUS effect in RF')]
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(obj,name):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def check(name,passed,**details):
    CHECKS.append(dict(check=name,passed=bool(passed),**details))
    if not passed:raise AssertionError((name,details))
def write(frame,name):frame.to_csv(OUT/(name+'.csv'),index=False,float_format='%.15g')
def snapshot():
    excluded={'_plot_dependencies','_matplotlib_config','__pycache__'}
    return {str(p.relative_to(ROOT)):sha(p) for folder in [S33,S34,S35,S36] for p in folder.rglob('*')
            if p.is_file() and not any(x in excluded for x in p.parts)}
def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        items=[]
        for k in columns:
            value=row[k]
            items.append('n/a' if pd.isna(value) else f'{value:.6g}' if isinstance(value,(float,np.floating)) else str(value))
        lines.append('| '+' | '.join(items)+' |')
    return '\n'.join(lines)
def scores(y,p,weights=None):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=weights).ravel()
    return np.array([average_precision_score(y,p,sample_weight=weights),roc_auc_score(y,p,sample_weight=weights),
        np.average((y-p)**2,weights=weights),tp/(tp+fn),tn/(tn+fp),matthews_corrcoef(y,pred,sample_weight=weights)])
def stats(frame,keys,cols):
    a=frame.groupby(keys)[cols].agg(['mean','std','min','max']);a.columns=['_'.join(k) for k in a.columns]
    return a.reset_index()

def main():
    start=time.perf_counter()
    # This is a prospective freeze for this analysis, not preregistration before earlier experiments.
    plan=dict(contrasts=CONTRASTS,interactions=INTERACTIONS,metrics=METRICS,
        interaction_metrics=METRICS[:3],planned_test_family_size=54,
        test_family='48 primary contrast/metric slots + 6 interaction slots, one exploratory Holm family; no confirmatory tests.',
        estimand_fold='Mean of 25 paired outer-validation fold score differences; Brier lower is better, all other primary metrics higher is better.',
        estimand_bootstrap='Mean of five within-repeat pooled metric differences conditional on saved fitted predictions; not a five-repeat independent-sample bootstrap.',
        corrected_cv='SE=sqrt((1/25 + mean(n_validation/n_original_training))*sample_variance_of_paired_differences), Student t df24; approximate not exact.',
        multiplicity='Two-sided corrected-CV p values, Holm across all54 planned slots. Undefined/degenerate tests padded with1 for adjustment but displayed unavailable. Marginal 95% CIs and Bonferroni simultaneous approximate 95% intervals are separate; no claim that marginal intervals are Holm-adjusted.',
        degeneracy='If all differences <=1e-12 absolute or paired SD <=1e-15, no informative t inference/CI; preserve observed values and report unavailable p.',
        bootstrap=dict(B=BOOTSTRAPS,seed=BOOT_SEED,unit='patient row_index',n_patients=4088,
            rule='Ordinary paired multinomial patient resampling; same counts apply across all five repeat predictions and compared streams; preserve score ties; percentile 2.5/97.5 intervals conditional on fitted scores; no bootstrap p-values.',
            limitation='Shared training induces cross-patient prediction dependence not captured by fixed-prediction resampling; no full training/selection uncertainty or guaranteed coverage.'),
        cutoffs='Existing fixed0.5 only for minority metrics; Step36 threshold-selected policies summarized descriptively, not added to test family.',
        clinical_minimum_effect=None,confirmatory=False,prior_results_known=True,
        seed_scope='Model/CV seed42 unchanged. 37042 is a Monte Carlo resampling seed, not a new model/CV seed experiment.',
        model_fits=0,calibration_fits=0,
        methods_sources=[
            dict(title='Bouckaert & Frank (2004)',url='https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf',use='corrected repeated k-fold approximation, sections3.2–3.3'),
            dict(title='Nadeau & Bengio (2003)',url='https://doi.org/10.1023/A:1024068626366',use='training/test dependence motivation'),
            dict(title='Bengio & Grandvalet (2004)',url='https://www.jmlr.org/papers/v5/grandvalet04a.html',use='no universal unbiased K-fold CV variance estimator'),
            dict(title='Holm (1979)',url='https://www.jstor.org/stable/4615733',use='step-down multiplicity adjustment'),
            dict(title='Efron (1979)',url='https://doi.org/10.1214/aos/1176344552',use='empirical bootstrap foundation; no claim of full-CV validity for fixed-prediction intervals')])
    dump(plan,'statistical_analysis_plan.json')
    inputs=snapshot()
    for folder in [S34,S35,S36]:
        check('earlier verdict '+folder.name,json.loads((folder/'verification.json').read_text())['overall_passed'])
        for row in pd.read_csv(folder/'artifact_inventory.csv').itertuples():check('prior output hash '+row.file,sha(folder/row.file)==row.sha256)
    m33=json.loads((S33/f'{RUN}_manifest.json').read_text())
    for k,entry in m33['files'].items():
        if entry['sha256']:check('Step33 hash '+k,sha(ROOT/entry['path'])==entry['sha256'])
    raw=pd.read_csv(S33/f'{RUN}_oof_predictions.csv',float_precision='round_trip');raw['method']='original'
    nested=pd.read_csv(S35/'nested/outer_predictions.csv',float_precision='round_trip')
    fields=['pipeline','method','repeat','fold','row_index','y_true','y_proba']
    data=pd.concat([raw[fields],nested[nested.method!='uncalibrated_reference'][fields]],ignore_index=True)
    data['stream']=data.pipeline+'::'+data.method
    foldplan=pd.read_csv(S33/f'{RUN}_fold_plan.csv')
    ratio=float((foldplan.n_validation/foldplan.n_train).mean())
    check('25 shared splits and original observations',len(foldplan)==25 and
          set(foldplan.repeat)==set(range(1,6)) and (foldplan.n_train+foldplan.n_validation).eq(4088).all())
    check('25 configurations/37 streams',data.pipeline.nunique()==25 and data.stream.nunique()==37)
    check('no duplicate patient within stream/repeat',not data.duplicated(['stream','repeat','row_index']).any())
    ref=raw[(raw.pipeline=='dummy_prior__none')&(raw.repeat==1)].set_index('row_index').y_true.sort_index()
    ids=ref.index.to_numpy();y=ref.to_numpy(dtype=int)
    patient_folds=raw[raw.pipeline=='dummy_prior__none'].set_index(['repeat','row_index']).fold.sort_index()
    write(pd.DataFrame(dict(row_index=ids,y_true=y)),'bootstrap_patient_identifiers')
    fold_rows=[];repeat_rows=[];scorers={};cache={}
    needed={c[side] for c in CONTRASTS for side in ['a','b']}
    for s in sorted(data.stream.unique()):
        g=data[data.stream==s];key,method=s.split('::')
        for repeat in range(1,6):
            part=g[g.repeat==repeat].set_index('row_index').sort_index()
            check('paired patient identities/labels and shared split',part.index.equals(ref.index) and
                  np.array_equal(part.y_true,y) and np.array_equal(part.fold,patient_folds.loc[repeat].to_numpy()))
            p=part.y_proba.to_numpy()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            result=scores(y,p)
            repeat_rows.append(dict(stream=s,pipeline=key,method=method,repeat=repeat,**dict(zip(METRICS,result))))
            if s in needed:
                scorer=WeightedPredictionScorer(y,p);scorers[(s,repeat)]=scorer
                result_w,_=scorer.score(np.ones((1,len(y))))
                check('weighted rank/tie algorithm matches sklearn point metrics',np.allclose(result_w[0],result,atol=1e-12,rtol=0))
                cache[(s,repeat)]=(y,p)
            for fold in range(1,6):
                mask=part.fold.to_numpy()==fold
                result=scores(y[mask],p[mask])
                fold_rows.append(dict(stream=s,pipeline=key,method=method,repeat=repeat,fold=fold,
                                     n_validation=int(mask.sum()),**dict(zip(METRICS,result))))
    fold=pd.DataFrame(fold_rows);repeats=pd.DataFrame(repeat_rows)
    write(fold,'all_stream_fold_metrics');write(repeats,'all_stream_pooled_repeat_metrics')
    # Reconcile the existing saved Step34 fold scores, preventing metric drift.
    saved=pd.read_csv(S34/'fold_metrics_with_mcc.csv',float_precision='round_trip')
    actual=fold[fold.method=='original'].merge(saved,on=['pipeline','repeat','fold'],suffixes=('_new','_saved'))
    for metric,old in [('average_precision','average_precision'),('roc_auc','roc_auc'),('brier_score','brier_score'),
                       ('sensitivity','recall_sensitivity'),('specificity','specificity'),('mcc','mcc')]:
        new_column=metric+'_new' if metric==old else metric
        saved_column=old+'_saved' if metric==old else old
        check('Step34 metric reproduction '+metric,np.max(np.abs(actual[new_column]-actual[saved_column]))<1e-12)
    write(stats(fold,['stream','pipeline','method'],METRICS),'fold_variability_summary')
    write(stats(repeats,['stream','pipeline','method'],METRICS),'repeat_variability_summary')
    paired=[];paired_repeats=[];tests=[]
    fidx=fold.set_index(['stream','repeat','fold']);ridx=repeats.set_index(['stream','repeat'])
    for c in CONTRASTS:
        a=fidx.loc[c['a']].sort_index();b=fidx.loc[c['b']].sort_index()
        check('exact paired fold coverage '+c['id'],len(a)==25 and a.index.equals(b.index))
        for metric in METRICS:
            d=a[metric]-b[metric]
            tests.append(dict(contrast_id=c['id'],label=c['label'],kind='paired_configuration_contrast',metric=metric,
                              a=c['a'],b=c['b'],**corrected_cv(d,ratio,54)))
            for (r,f),value in d.items():paired.append(dict(contrast_id=c['id'],metric=metric,repeat=r,fold=f,difference=float(value),a=float(a.loc[(r,f),metric]),b=float(b.loc[(r,f),metric])))
            rd=ridx.loc[c['a'],metric]-ridx.loc[c['b'],metric]
            for r,value in rd.items():paired_repeats.append(dict(contrast_id=c['id'],metric=metric,repeat=int(r),difference=float(value)))
    interaction_rows=[]
    for interaction in INTERACTIONS:
        strategy=interaction['strategy']
        for metric in METRICS[:3]:
            d=(fidx.loc[stream('logistic_regression__'+strategy),metric]-fidx.loc[stream('logistic_regression__none'),metric])-(
               fidx.loc[stream('random_forest__'+strategy),metric]-fidx.loc[stream('random_forest__none'),metric])
            tests.append(dict(contrast_id=interaction['id'],label=interaction['label'],kind='paired_difference_in_differences',
                              metric=metric,a='LR strategy effect',b='RF strategy effect',**corrected_cv(d,ratio,54)))
            for (r,f),value in d.items():interaction_rows.append(dict(contrast_id=interaction['id'],metric=metric,repeat=r,fold=f,difference=float(value)))
    tests=pd.DataFrame(tests)
    check('54 planned hypothesis slots',len(tests)==54)
    ppad=tests.p_corrected.fillna(1).to_numpy()
    adjusted=holm_adjust(ppad)
    tests['p_holm_54']=np.where(tests.p_corrected.notna(),adjusted,np.nan)
    tests['holm_exploratory_flag_005']=tests.p_holm_54.lt(.05)
    tests['intervals_are_approximate']=True
    check('Holm never decreases p',np.all(adjusted+1e-15>=ppad))
    # Independent set-based Holm formula, including unavailable slots padded with1.
    for i,pvalue in enumerate(ppad):
        ordered=np.sort(ppad)
        expected=min(1,max((len(ppad)-j)*p for j,p in enumerate(ordered) if p<=pvalue))
        check('independent Holm step-down check',abs(expected-adjusted[i])<1e-12)
    write(tests,'corrected_cv_effects_uncertainty_and_multiplicity')
    write(pd.DataFrame(paired),'paired_fold_differences')
    paired_repeats=pd.DataFrame(paired_repeats)
    write(paired_repeats,'paired_pooled_repeat_differences')
    write(pd.DataFrame(interaction_rows),'paired_interaction_fold_differences')
    write(stats(paired_repeats,['contrast_id','metric'],['difference']),'paired_repeat_direction_and_variability')
    consistency=[]
    for (cid,metric),g in paired_repeats.groupby(['contrast_id','metric']):
        consistency.append(dict(contrast_id=cid,metric=metric,n_repeats=5,
            positive_repeat_count=int((g.difference>1e-12).sum()),negative_repeat_count=int((g.difference < -1e-12).sum()),
            min_repeat_difference=float(g.difference.min()),max_repeat_difference=float(g.difference.max()),
            mean_pooled_repeat_difference=float(g.difference.mean())))
    write(pd.DataFrame(consistency),'paired_repeat_sign_consistency')
    print('CORRECTED CV COMPLETE',len(tests),'slots;',int(tests.holm_exploratory_flag_005.sum()),'Holm flags',flush=True)
    # Ordinary patient bootstrap: same multinomial count vector for every stream/repeat.
    ordered_streams=sorted(needed);sindex={s:i for i,s in enumerate(ordered_streams)}
    bootstrap=np.empty((BOOTSTRAPS,len(ordered_streams),len(METRICS)))
    all_weights=np.empty((BOOTSTRAPS,len(ids)),dtype=np.uint16)
    rng=np.random.default_rng(BOOT_SEED)
    degeneracy={s:0 for s in ordered_streams}
    for begin in range(0,BOOTSTRAPS,BATCH):
        end=min(BOOTSTRAPS,begin+BATCH)
        counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=end-begin).astype(np.uint16)
        all_weights[begin:end]=counts
        check('patient bootstrap draw size/class support',np.all(counts.sum(axis=1)==4088) and
              np.all(counts@y>0) and np.all(counts@(1-y)>0))
        weights=counts.astype(float)
        for s in ordered_streams:
            values=np.zeros((end-begin,len(METRICS)))
            for repeat in range(1,6):
                scored,zero=scorers[(s,repeat)].score(weights)
                values+=scored/5;degeneracy[s]+=zero
                if begin==0:
                    for index in [0,1,2]:
                        yy,pp=cache[(s,repeat)]
                        independent=scores(yy,pp,weights[index])
                        check('bootstrap weighted metrics independently replayed',np.allclose(scored[index],independent,atol=1e-12,rtol=0))
            bootstrap[begin:end,sindex[s]]=values
        if end%500==0:print('PATIENT BOOTSTRAP',end,'/',BOOTSTRAPS,flush=True)
    np.savez_compressed(OUT/'patient_bootstrap_counts.npz',counts=all_weights,row_index=ids)
    bootrows=[];drawrows=[]
    for c in CONTRASTS:
        effect=bootstrap[:,sindex[c['a']]]-bootstrap[:,sindex[c['b']]]
        pa=repeats[repeats.stream==c['a']][METRICS].mean().to_numpy()
        pb=repeats[repeats.stream==c['b']][METRICS].mean().to_numpy()
        for j,metric in enumerate(METRICS):
            low,high=np.quantile(effect[:,j],[.025,.975])
            bootrows.append(dict(contrast_id=c['id'],label=c['label'],metric=metric,
                estimand='mean_within_repeat_pooled_metric_difference_conditional_on_saved_predictions',
                point_difference=float(pa[j]-pb[j]),bootstrap_sd=float(effect[:,j].std(ddof=1)),
                conditional_percentile_ci95_low=float(low),conditional_percentile_ci95_high=float(high),
                bootstrap_draws=BOOTSTRAPS,patient_clusters=4088,seed=BOOT_SEED,
                multiplicity_adjusted=False,full_retraining_uncertainty=False))
            for number,value in enumerate(effect[:,j],1):drawrows.append(dict(contrast_id=c['id'],metric=metric,draw=number,difference=float(value)))
    boot=pd.DataFrame(bootrows)
    write(boot,'conditional_patient_bootstrap_intervals')
    write(pd.DataFrame(drawrows),'conditional_patient_bootstrap_draws')
    write(pd.DataFrame([dict(stream=s,zero_mcc_denominator_stream_repeat_draws=n,
                            total_stream_repeat_draws=5*BOOTSTRAPS) for s,n in degeneracy.items()]),'bootstrap_metric_degeneracy')
    # Rank stability: same four learned models within each imbalance strategy.
    rankrows=[]
    for level,frame in [('individual_fold',fold[fold.method=='original']),('pooled_repeat',repeats[repeats.method=='original'])]:
        grouping=['repeat','fold'] if level=='individual_fold' else ['repeat']
        subset=frame[~frame.pipeline.str.startswith('dummy_prior')].copy()
        subset['model']=subset.pipeline.str.split('__').str[0];subset['strategy']=subset.pipeline.str.split('__').str[1]
        for group,rows in subset.groupby(grouping+['strategy']):
            repeat=group[0];f=group[1] if level=='individual_fold' else None;strategy=group[-1]
            check('four paired models in rank set',len(rows)==4)
            for metric in ['average_precision','roc_auc','brier_score','mcc']:
                ranks=rows[metric].rank(method='average',ascending=metric=='brier_score')
                best=rows[metric].min() if metric=='brier_score' else rows[metric].max()
                for idx,row in rows.iterrows():
                    rankrows.append(dict(level=level,repeat=int(repeat),fold=f,strategy=strategy,metric=metric,
                        model=row.model,rank=float(ranks.loc[idx]),value=float(row[metric]),
                        tied_best=bool(abs(row[metric]-best)<=1e-12)))
    ranks=pd.DataFrame(rankrows)
    write(ranks,'model_ranks_by_fold_repeat_and_strategy')
    rank_summary=ranks.groupby(['level','strategy','metric','model']).agg(n_correlated_units=('rank','size'),
        mean_rank=('rank','mean'),sd_rank=('rank','std'),min_rank=('rank','min'),max_rank=('rank','max'),
        tied_best_count=('tied_best','sum'),tied_best_fraction=('tied_best','mean')).reset_index()
    write(rank_summary,'rank_stability_summary')
    rank_pairs=[]
    repeat_ranks=ranks[ranks.level=='pooled_repeat']
    for (strategy,metric),g in repeat_ranks.groupby(['strategy','metric']):
        pivot=g.pivot(index='repeat',columns='model',values='rank')
        for a,b in combinations(range(1,6),2):
            estimable=bool(pivot.loc[a].std()>0 and pivot.loc[b].std()>0)
            corr=float(np.corrcoef(pivot.loc[a],pivot.loc[b])[0,1]) if estimable else None
            rank_pairs.append(dict(strategy=strategy,metric=metric,repeat_a=a,repeat_b=b,
                                   rank_correlation_descriptive=corr,
                                   status='descriptive' if estimable else 'undefined_constant_rank_vector'))
    write(pd.DataFrame(rank_pairs),'descriptive_repeat_rank_correlations')
    # Step36 results are inspected as conditional policy evidence, not added tests.
    policies=pd.read_csv(S36/'inner_selected_policy_pooled_repeat_results.csv',float_precision='round_trip')
    policy_subset=policies[(policies.pipeline=='logistic_regression__class_weight_balanced')&
                          policies.method.isin(['original','sigmoid'])]
    write(stats(policy_subset,['pipeline','method','policy'],['sensitivity','specificity','mcc','fp','fn','threshold_mean']),
          'step36_policy_stability_descriptive_only')
    # Publication figures, with clearly different estimands and interval types.
    sys.dont_write_bytecode=True;os.environ['MPLCONFIGDIR']=str(OUT/'_matplotlib_config')
    sys.path.append(str(S34/'_plot_dependencies'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,caption):
        fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
        figures.append(dict(file=name+'.png',caption=caption))
    fig,axes=plt.subplots(1,3,figsize=(14,6),sharey=True)
    for ax,metric in zip(axes,METRICS[:3]):
        rows=tests[(tests.kind=='paired_configuration_contrast')&(tests.metric==metric)].set_index('contrast_id')
        for i,c in enumerate(CONTRASTS):
            row=rows.loc[c['id']];x=row.mean_difference
            if pd.notna(row.ci95_low):
                ax.plot([row.bonferroni_family_ci_low,row.bonferroni_family_ci_high],[i,i],color='#bbbbbb',linewidth=2)
                ax.plot([row.ci95_low,row.ci95_high],[i,i],color='#0072B2',linewidth=4)
            ax.scatter(x,i,color='#D55E00' if row.holm_exploratory_flag_005 else '#222222',s=30,zorder=3)
        ax.axvline(0,color='#777777',linestyle='--');ax.set_title(metric.replace('_',' '));ax.set_xlabel('Mean paired fold difference A − B');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(8),[c['id']+': '+c['label'] for c in CONTRASTS]);axes[0].invert_yaxis()
    fig.suptitle('Exploratory corrected repeated-CV effects and approximate uncertainty')
    fig.text(.31,.015,'Blue: marginal 95% interval; gray: Bonferroni family interval (54 slots). Orange: Holm p<0.05 under approximation.\nNo interval for numerically equal/zero-variance differences. Brier decrease favors A; other scores increase favors A.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94));save(fig,'figure01_corrected_cv_effects','Paired mean fold-score effects; corrected SE uses actual original validation/training ratios. Approximate intervals do not guarantee coverage under repeated-CV dependence.')
    fig,axes=plt.subplots(1,3,figsize=(14,6),sharey=True)
    for ax,metric in zip(axes,METRICS[:3]):
        rows=boot[boot.metric==metric].set_index('contrast_id')
        for i,c in enumerate(CONTRASTS):
            row=rows.loc[c['id']]
            ax.plot([row.conditional_percentile_ci95_low,row.conditional_percentile_ci95_high],[i,i],color='#009E73',linewidth=3)
            ax.scatter(row.point_difference,i,color='#222222',s=25)
        ax.axvline(0,color='#777777',linestyle='--');ax.set_title(metric.replace('_',' '));ax.set_xlabel('Mean pooled-repeat difference A − B');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(8),[c['id']+': '+c['label'] for c in CONTRASTS]);axes[0].invert_yaxis()
    fig.suptitle('Patient-paired bootstrap uncertainty conditional on saved predictions')
    fig.text(.32,.015,'5,000 ordinary patient resamples; all five repeat predictions retained together. Unadjusted percentile intervals.\nDifferent estimand from fold-score corrected-CV; excludes refitting and prior selection uncertainty.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94));save(fig,'figure02_conditional_patient_bootstrap','These conditional intervals preserve patient pairing across streams/repeats but do not account for shared-training coupling between different patients. They are not full model-development confidence intervals.')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for ax,c in zip(axes.flat,CONTRASTS[:4]+CONTRASTS[4:6]):
        for metric,color in [('average_precision','#0072B2'),('roc_auc','#D55E00')]:
            g=paired_repeats[(paired_repeats.contrast_id==c['id'])&(paired_repeats.metric==metric)].sort_values('repeat')
            ax.plot(g.repeat,g.difference,marker='o',color=color,label=metric)
        ax.axhline(0,color='#777777',linestyle='--');ax.set_title(c['id']+': '+c['label'],fontsize=8)
        ax.set_xlabel('CV repeat, existing seed42 sequence');ax.set_ylabel('Pooled-repeat difference');ax.set_xticks(range(1,6));ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Observed direction consistency across dependent repeats')
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'figure03_repeat_effect_stability','Each point uses 4,088 patients once within a repeat. Repeat directions are descriptive counts, not independent binomial trials or a seed-robustness experiment.')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=True)
    models=['logistic_regression','decision_tree','random_forest','svc_rbf'];strategies=['none','class_weight_balanced','random_over_sampler','random_under_sampler','smote','adasyn']
    for ax,strategy in zip(axes.flat,strategies):
        g=repeat_ranks[(repeat_ranks.strategy==strategy)&(repeat_ranks.metric=='average_precision')]
        for model in models:
            part=g[g.model==model].sort_values('repeat');ax.plot(part.repeat,part['rank'],marker='o',label=model)
        ax.set_title(strategy.replace('_',' '));ax.set_xticks(range(1,6));ax.set_yticks([1,2,3,4]);ax.set_ylim(4.2,.8)
        ax.set_xlabel('CV repeat');ax.set_ylabel('Within-strategy AP rank');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Model rank stability depends on imbalance strategy')
    fig.tight_layout(rect=(0,.06,1,.94));save(fig,'figure04_ap_rank_stability','Ranks compare four learned families within a shared strategy. Ranks are descriptive; four models, strategies and repeated folds are not independent benchmark datasets.')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,METRICS[:3]):
        g=tests[(tests.kind=='paired_difference_in_differences')&(tests.metric==metric)]
        for i,(_,row) in enumerate(g.iterrows()):
            ax.plot([row.ci95_low,row.ci95_high],[i,i],color='#0072B2',linewidth=3)
            ax.scatter(row.mean_difference,i,color='#222222')
        ax.set_yticks([0,1],['Weighting','RUS']);ax.axvline(0,color='#777777',linestyle='--')
        ax.set_title(metric.replace('_',' '));ax.set_xlabel('(LR strategy−none) − (RF strategy−none)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Exploratory model-by-strategy difference-in-differences')
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'figure05_model_strategy_interactions','Marginal corrected-CV approximate 95% intervals; all six interaction slots share the same 54-slot Holm family. Not a factorial independent-observation ANOVA.')
    # Report: cite primary methods and separate estimands, assumptions and scope.
    keytests=tests[tests.kind=='paired_configuration_contrast']
    sections=[
        '# StrokeSure Step 37: statistical comparisons and robustness',
        'Analysis date: 2026-11-10 (Asia/Calcutta). Saved Steps 33–36 only. No training, calibration fitting, new CV seeds, raw-dataset loading, locked-test access, SHAP, final evaluation or Git changes.',
        '## Scope and evidential status',
        'This report supplies limited, assumption-dependent exploratory inference and descriptive robustness evidence. Earlier results and configuration choices were already examined; the Step 37 plan is fixed before its computations but is not pre-experiment preregistration. There is no confirmatory test family and no claim of clinical superiority. Multiplicity adjustment cannot remove earlier data-dependent selection or repair an invalid variance model.',
        '## Exact paired observation structure',
        'All original 25 configurations share 25 outer folds indexed by (repeat, fold), five folds per repeat and five repeats from random_state42. Every repeat predicts the same 4,088 training patients once (199 events, 3,889 nonevents). Within a fold, the same validation patient identities and outcomes are paired across configurations. Training sets overlap within and across repeats; patient predictions recur across repeats. Step35 calibration maps fit only inner outer-training scores/labels and are evaluated on these same outer validation identities. The six Step35 unadjusted references duplicate Step33 and are not counted twice.',
        'Step36 additionally selected policies on saved inner scores and evaluated on disjoint outer rows, but that policy evidence is kept descriptive here rather than enlarging the inferential family after seeing threshold results. Raw SVM inner Platt probabilities remain unavailable for its original selected-threshold policy. No apparent OOF-selected threshold results enter the statistical tests.',
        '## Prespecified-for-Step37 contrasts and endpoints',
        table(pd.DataFrame(CONTRASTS),['id','label','a','b']),
        'All effects are A minus B in absolute score units. Higher AP, ROC-AUC, sensitivity, specificity and MCC favor A; lower Brier favors A. Multiplying differences by100 gives percentage points, not relative percent change. Minority metrics use the original fixed0.50 cutoff. A zero MCC denominator uses the sklearn zero convention; no-positive-prediction behavior must not be mistaken for clinically useful detection.',
        'The eight contrasts use six metrics: AP, ROC-AUC, Brier, sensitivity, specificity and MCC. Two additional LR-versus-RF difference-in-differences compare weighting and undersampling effects against none for AP, ROC-AUC and Brier. Thus there are 48+6=54 planned slots, including unavailable/degenerate slots. No unrestricted pairwise search or threshold optimization was added.',
        '## Dependence-aware methods and limitations',
        f'### Approximate corrected repeated-CV inference\n\nFor paired differences d over 25 folds, SE = sqrt((1/25 + q) s_d²), with q = mean(n_validation/n_original_training) = {ratio:.9f}, and Student-t reference df24. We use original patient counts before resampling, not synthetic/oversampled counts or inner-fit sample sizes. The multiplier is {math.sqrt((1/25+ratio)/(1/25)):.4f} relative to the naive SE; 25 folds are not treated as independent datasets. This follows the corrected repeated-k-fold construction in [Bouckaert & Frank, sections3.2–3.3](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf), motivated by [Nadeau & Bengio](https://doi.org/10.1023/A:1024068626366).',
        'This is an approximate covariance correction, not a guarantee of exact test size or coverage. It assumes exchangeable data, stable comparable learning procedures, a reasonable variance/correlation approximation and approximately suitable paired-score distribution. Evidence is most straightforward for additive Brier losses; AP/AUC/MCC are nonlinear and low event counts can produce skewed/discrete fold differences. Their t-based p-values/intervals are sensitivity evidence under stronger assumptions. [Bengio & Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html) establish the absence of a universal unbiased variance estimator for K-fold CV, so this correction is not an exact solution.',
        'Numerically equal or zero-variance paired differences do not receive forced p-values or zero-width inferential intervals. Monotone sigmoid maps can preserve within-fold AUC/AP exactly while pooled-repeat ranks change across independently mapped folds. Those are different estimands, not a contradiction. Source and bootstrap checks preserve probability ties.',
        '### Multiplicity\n\nAll two-sided corrected-CV p-values share one exploratory Holm family of54 planned slots, alpha0.05, using [Holm’s step-down procedure](https://www.jstor.org/stable/4615733). Unavailable tests are conservatively padded with1 during adjustment and displayed unavailable. Marginal 95% intervals are not multiplicity-adjusted; a separate Bonferroni simultaneous approximate 95% family interval is supplied. Holm flags and Bonferroni intervals need not give identical decisions. Error-control language is conditional on valid underlying p-values; the correction does not make approximate tests exact.',
        '### Patient-paired conditional bootstrap\n\n5,000 ordinary multinomial resamples draw 4,088 patient IDs with replacement, using Monte Carlo seed37042. Each draw carries all five repeat predictions together and uses the same weights across streams. We compute each repeat’s metric and then average five repeat metrics; we do not average probabilities into an ensemble or treat 20,440 repeated rows as independent patients. Percentile 2.5/97.5 intervals are conditional on saved fitted scores and are unadjusted descriptive uncertainty intervals, not additional hypothesis tests. The empirical resampling idea follows [Efron](https://doi.org/10.1214/aos/1176344552).',
        'Patient pairing handles repeated measurements of the same patient and paired models, but fixed-prediction resampling does not reproduce model refitting or the dependence between different patients caused by shared training. It also excludes prior configuration/calibration-method selection uncertainty. These bootstrap intervals must not be sold as full training-process confidence intervals or used to claim significance merely because they exclude zero. They estimate mean pooled-repeat differences, whereas corrected-CV intervals estimate mean fold-score differences; AP in particular differs between these aggregations. No bootstrap p-values or independently resampled repeat tests were computed.',
        '## Paired effects and corrected-CV uncertainty',
        table(keytests[['contrast_id','metric','mean_difference','ci95_low','ci95_high','p_corrected','p_holm_54','test_status']],
              ['contrast_id','metric','mean_difference','ci95_low','ci95_high','p_corrected','p_holm_54','test_status']),
        f'Of 54 planned slots, {int(tests.p_corrected.notna().sum())} have informative approximate tests, {int(tests.p_corrected.isna().sum())} are flagged unavailable, and {int(tests.holm_exploratory_flag_005.sum())} have Holm-adjusted p<0.05 under the stated approximation. These are exploratory flags, not preregistered confirmations. All observed effects and unavailable cases remain in the CSV.',
        '## Conditional patient-bootstrap effects and uncertainty',
        table(boot,['contrast_id','metric','point_difference','bootstrap_sd','conditional_percentile_ci95_low','conditional_percentile_ci95_high']),
        'Bootstrap draw weights, raw contrast draws and zero-denominator MCC diagnostics are retained. Bootstrap intervals describe conditional patient-mix uncertainty and must not be conflated with corrected-CV fold-score intervals or interpreted as multiplicity-adjusted evidence.',
        '## Model-by-strategy interactions',
        table(tests[tests.kind=='paired_difference_in_differences'],['contrast_id','metric','mean_difference','ci95_low','ci95_high','p_holm_54']),
        'The interaction estimand is (strategy−none in LR) minus (strategy−none in RF), paired on the same folds. Raw Brier interaction signs reverse the benefit direction: a positive value means a larger Brier penalty for LR. Six interaction slots are already included in the54-slot family. This is a limited pairwise interaction analysis, not a global factorial claim across all models/strategies.',
        '## Rank stability and observed repeat consistency',
        'Within each imbalance strategy, the four learned model families are ranked for AP, ROC-AUC, Brier and MCC separately in each fold and pooled repeat. Ties use average rank; tied-best frequencies can sum above one. Counts over25 folds or5 repeats are descriptive, not independent binomial trials. No Friedman/Nemenyi test treats folds, repeats or strategies as independent benchmark datasets.',
        table(rank_summary[(rank_summary.level=='pooled_repeat')&(rank_summary.metric=='average_precision')],
              ['strategy','model','mean_rank','min_rank','max_rank','tied_best_count','n_correlated_units']),
        'Direction counts/ranges for each paired repeat contrast are supplied separately. Stable signs across five dependent repeats do not constitute a five-study meta-analysis, independent seed replication or a significance test. Rank changes can reflect small differences and do not establish practical materiality.',
        '## Research questions and working hypotheses',
        '**RQ5:** How stable are model-performance results across repeated validation folds and random seeds?\n\nFold/repeat effect distributions, ranks, score ranges and Step36 policy variability address observed partition stability under seed42. Independent model/CV seed sensitivity remains untested. Bootstrap seed37042 is a numerical resampling control, not an additional model seed.',
        '**RQ8:** Are observed performance differences between models and experimental configurations statistically consistent?\n\nThe paired tables distinguish observed direction consistency, approximate corrected-CV/multiplicity evidence and conditional fixed-score patient uncertainty. Conclusions are limited to the explicit contrasts, metrics, assumptions and this dataset; failure to flag a difference is not equivalence, and flagged differences need not be clinically important.',
        '**H1:** Different ML models will show statistically meaningful differences across at least some discrimination, minority-class, calibration, or robustness metrics.\n\nThis analysis supplies limited exploratory corrected comparisons, not a confirmatory global model-superiority claim. Brier is probability accuracy, not calibration alone; no universal best model is selected.',
        '**H2:** Imbalance-handling strategy will materially change minority-class performance and may produce trade-offs between recall, precision and specificity.\n\nFixed-threshold sensitivity/specificity/MCC effects and earlier FP/FN trade-offs quantify those changes. Approximate flags do not establish clinical materiality without costs or a prespecified meaningful-effect margin.',
        '**H4:** Performance estimates from a single split will show greater apparent variability than estimates summarized across repeated stratified validation.\n\nNot directly tested. Fold scores have different evaluation sample sizes/estimands from pooled-repeat scores; their SDs are not a valid standalone single-split-versus-repeated-estimate comparison. No new split experiment was justified merely to manufacture support.',
        '**H7:** Observed model differences will vary with imbalance strategy, so model ranking based on one configuration will not necessarily persist across all configurations.\n\nDescriptive strategy-specific ranks and the two explicit paired difference-in-differences examine this. Only those interaction metrics with adjusted exploratory evidence may be discussed as assumption-dependent statistical indications. Descriptive rank reversals alone do not prove an interaction.',
        '## Remaining limits and next steps',
        'All configurations were studied previously, particularly the calibration subset, so selection optimism remains. No analysis here changes that fact. There is one public dataset, 199 events, one model/CV seed and no untouched external cohort. Neither clinical deployment readiness nor broad across-dataset/seed robustness follows. Corrected-CV assumptions are especially uncertain for nonlinear rare-event metrics. Conditional bootstrap omits refitting and shared-training dependence. No exact distribution-free confidence statement is claimed.',
        'For stronger future evidence, predefine meaningful-effect margins and a confirmatory contrast family, obtain an independently designed multi-seed or external-validation protocol, and use a properly designed single-split-versus-repeated-estimate comparison for H4. Additional fitting would be needed for those targets, but none was necessary or performed for this saved-artifact analysis. Keep the locked test partition untouched until final development choices are fixed. No threshold, model or calibrator was selected by Step37.',
        '## Figures',
    ]
    for fig in figures:sections += [f'![{fig["caption"]}]({fig["file"]})',fig['caption']]
    sections += ['## Reproducibility and files',
        'The fixed plan is `statistical_analysis_plan.json`; raw paired effects, corrected-CV tables, multiplicity details, patient-bootstrap weights/draws/intervals, ranks and variability tables are separate CSV/NPZ outputs. `verification.json` records independent checks and `artifact_inventory.csv` hashes outputs. Original Steps33–36 files are checked before and after analysis. Source scripts import scorers only, not model/data loaders. Reproduce with the existing project Python: `.\\.venv\\Scripts\\python.exe results/step37_statistical_robustness/analyze_step37.py`.',
    ]
    check('all previous artifacts unchanged',inputs==snapshot())
    check('bootstrap effects finite',np.isfinite(bootstrap).all())
    verdict=dict(overall_passed=all(c['passed'] for c in CHECKS),n_checks=len(CHECKS),n_passed=sum(c['passed'] for c in CHECKS),n_failed=sum(not c['passed'] for c in CHECKS),checks=CHECKS)
    dump(verdict,'verification.json');dump(figures,'figure_captions.json')
    report=OUT/'scientific_statistical_robustness_report.md'
    report.write_text('\n\n'.join(sections)+f'\n\nFinal verification: **{verdict["n_passed"]}/{verdict["n_checks"]} PASS**, zero original-artifact alterations.\n',encoding='utf-8')
    dump(dict(original_artifact_hashes=inputs,source_run=RUN,contrasts=8,interaction_contrasts=2,planned_test_slots=54,
        informative_tests=int(tests.p_corrected.notna().sum()),holm_exploratory_flags=int(tests.holm_exploratory_flag_005.sum()),
        patient_bootstrap_draws=BOOTSTRAPS,bootstrap_seed=BOOT_SEED,patients=4088,positive_patients=199,
        fold_ratio=ratio,se_inflation_vs_naive=math.sqrt((1/25+ratio)/(1/25)),
        model_fits=0,calibrator_fits=0,seconds=time.perf_counter()-start,
        code_hashes={p.name:sha(p) for p in OUT.glob('*.py')},numpy=np.__version__,pandas=pd.__version__,
        scipy=sys.modules['scipy'].__version__,scikit_learn=sys.modules['sklearn'].__version__,matplotlib=matplotlib.__version__,
        verification={k:v for k,v in verdict.items() if k!='checks'}),'analysis_manifest.json')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ['_matplotlib_config','__pycache__']) and p.name not in ['artifact_inventory.csv','execution.log','delivery_verification.json']:
            inventory.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(inventory),'artifact_inventory')
    print(json.dumps({k:v for k,v in verdict.items() if k!='checks'},indent=2),flush=True)
    print(tests[['contrast_id','metric','mean_difference','ci95_low','ci95_high','p_holm_54']].to_string(index=False),flush=True)

if __name__=='__main__':main()
