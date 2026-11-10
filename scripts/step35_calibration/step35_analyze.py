"""Verify nested outputs and render a descriptive Step 35 scientific report."""
from pathlib import Path
import sys,os,json,hashlib,math
import numpy as np
import pandas as pd
from scipy.special import expit
from calibration_metrics import probability_metrics,reliability_bins,LOSS_EPS,LOGIT_EPS

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
NEST=OUT/'nested'
S34=ROOT/'results/step34_imbalance_analysis'
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(OUT/'_matplotlib_config')
sys.path.append(str(S34/'_plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS=['roc_auc','average_precision','brier_score','log_loss','ece_quantile_10','ece_uniform_10',
    'ece_quantile_5','ece_quantile_20','mce_quantile_10','mean_probability','mean_probability_bias',
    'calibration_in_large_intercept','joint_intercept','calibration_slope']
METHODS=['uncalibrated_reference','sigmoid','isotonic']
COLORS=dict(zip(METHODS,['#666666','#0072B2','#D55E00']))
SHORT={'logistic_regression__none':'LR / none','logistic_regression__class_weight_balanced':'LR / weight',
    'logistic_regression__random_under_sampler':'LR / RUS','logistic_regression__adasyn':'LR / CatADASYN',
    'random_forest__none':'RF / none','svc_rbf__random_under_sampler':'SVM / RUS'}
CHECKS=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(obj,p): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def fp(ids): return hashlib.sha256(np.asarray(sorted(map(int,ids)),dtype=np.int64).tobytes()).hexdigest()[:16]
def check(name,condition,**details):
    CHECKS.append(dict(check=name,passed=bool(condition),**details))
    if not condition: raise AssertionError((name,details))
def write(frame,name): frame.to_csv(OUT/(name+'.csv'),index=False,float_format='%.15g')
def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        values=[]
        for k in columns:
            value=r[k]
            values.append(f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)
def stats(frame,keys):
    cols=[m for m in METRICS if m in frame.columns]
    result=frame.groupby(keys)[cols].agg(['mean','std','min','max','count'])
    result.columns=['_'.join(k) for k in result.columns]
    return result.reset_index()

def main():
    design=json.loads((OUT/'design_and_provenance.json').read_text())
    execution=json.loads((NEST/'execution_verification.json').read_text())
    observations=json.loads((NEST/'execution_observations.json').read_text())
    check('nested executor passed',execution['overall_passed'])
    selected=design['selected']
    fits=pd.read_csv(NEST/'base_fit_records.csv',float_precision='round_trip')
    predictions=pd.read_csv(NEST/'outer_predictions.csv',float_precision='round_trip')
    inner=pd.read_csv(NEST/'inner_oof_calibration_scores.csv',float_precision='round_trip')
    audits=pd.read_csv(NEST/'sampling_audits.csv')
    warnings=pd.read_csv(NEST/'warnings.csv')
    errors=pd.read_csv(NEST/'errors.csv')
    maps=json.loads((NEST/'calibration_maps.json').read_text())
    plans=json.loads((NEST/'nested_split_plan.json').read_text())
    plan={(r['repeat'],r['fold']):r for r in plans}
    check('900 successful base fits',len(fits)==900 and fits.status.eq('ok').all() and
          len(fits[fits.phase=='inner'])==750 and len(fits[fits.phase=='outer'])==150)
    check('300 calibration maps',len(maps)==300 and {m['method'] for m in maps}=={'sigmoid','isotonic'})
    check('367920 outer prediction rows',len(predictions)==367920 and set(predictions.pipeline)==set(selected)
          and set(predictions.method)==set(METHODS))
    check('no duplicate outer predictions',not predictions.duplicated(['pipeline','method','repeat','row_index']).any())
    check('finite bounded predictions',np.isfinite(predictions.y_proba).all() and predictions.y_proba.between(0,1).all())
    check('450 sampling audits all pass',len(audits)==450 and audits.audit_status.eq('PASS').all()
          and audits.n_fractional_one_hot_values.eq(0).all())
    check('zero fold or calibration errors',errors.empty)
    check('150 expected SVC warnings',len(warnings)==150 and warnings.category.eq('FutureWarning').all()
          and warnings.pipeline.eq('svc_rbf__random_under_sampler').all()
          and warnings.message.str.startswith('The `probability` parameter was deprecated in 1.9').all())
    check('expected SVC warning events unique',not warnings.duplicated(['pipeline','repeat','fold','phase','inner_fold']).any())
    check('all inner calibration score rows retained',len(inner)==sum(r['n_calibration'] for r in plans)*len(selected))
    for r in plans:
        train=set(r['outer_fit_row_ids']); valid=set(r['outer_evaluation_row_ids'])
        check('outer calibration/evaluation disjoint',not train&valid and len(train|valid)==4088)
        scored=[]
        for s in r['inners']:
            fit=set(s['fit_row_ids']); score=set(s['calibration_score_row_ids'])
            check('inner scored rows never fitted or outer evaluated',not fit&score and not fit&valid
                  and not score&valid and fit|score==train)
            scored.extend(score)
        check('inner heldout coverage exactly once',len(scored)==len(train) and len(set(scored))==len(train))
    for row in fits.itertuples():
        r=plan[(row.repeat,row.fold)]
        if row.phase=='inner':
            split=r['inners'][int(row.inner_fold)-1]
            train,valid=split['fit_row_ids'],split['calibration_score_row_ids']
        else: train,valid=r['outer_fit_row_ids'],r['outer_evaluation_row_ids']
        check('actual fit/score fingerprints match independent role plan',row.train_fingerprint==fp(train)
              and row.score_fingerprint==fp(valid) and row.n_fit==len(train) and row.n_score==len(valid))
    original=pd.read_csv(ROOT/f'results/imbalance_cv/{design["source_run"]}/{design["source_run"]}_oof_predictions.csv',float_precision='round_trip')
    reference=original[(original.pipeline=='dummy_prior__none')&(original.repeat==1)].set_index('row_index').y_true.sort_index()
    max_raw=0.0; max_map=0.0
    for key in selected:
        for (repeat,fold),r in plan.items():
            inside=inner[(inner.pipeline==key)&(inner.repeat==repeat)&(inner.fold==fold)]
            check('inner calibration rows/labels match plan',len(inside)==r['n_calibration'] and inside.row_index.is_unique
                  and set(inside.row_index)==set(r['outer_fit_row_ids'])
                  and np.array_equal(inside.y_true.to_numpy(),reference.loc[inside.row_index].to_numpy()))
            for s in r['inners']:
                check('actual inner score provider was held out',set(inside[inside.inner_fold==s['inner_fold']].row_index)==set(s['calibration_score_row_ids']))
            part=predictions[(predictions.pipeline==key)&(predictions.repeat==repeat)&(predictions.fold==fold)]
            for method in METHODS:
                group=part[part.method==method]
                check('paired outer row membership/labels',len(group)==len(r['outer_evaluation_row_ids'])
                    and set(group.row_index)==set(r['outer_evaluation_row_ids'])
                    and np.array_equal(group.y_true.to_numpy(),reference.loc[group.row_index].to_numpy()))
                if method=='uncalibrated_reference':
                    source=original[(original.pipeline==key)&(original.repeat==repeat)&(original.fold==fold)].set_index('row_index')
                    delta=float(np.max(np.abs(group.y_proba.to_numpy()-source.loc[group.row_index,'y_proba'].to_numpy())))
                    max_raw=max(max_raw,delta)
                else:
                    map_rows=[m for m in maps if m['pipeline']==key and m['repeat']==repeat and m['fold']==fold and m['method']==method]
                    check('one fitted map per configuration/fold/method',len(map_rows)==1)
                    m=map_rows[0]; p=m['parameters']
                    expected=(expit(-(p['a']*group.base_score.to_numpy()+p['b'])) if method=='sigmoid'
                              else np.interp(group.base_score.to_numpy(),p['x_thresholds'],p['y_thresholds']))
                    max_map=max(max_map,float(np.max(np.abs(expected-group.y_proba.to_numpy()))))
                    check('map fits natural-prevalence disjoint calibration rows',m['calibration_fingerprint']==fp(r['outer_fit_row_ids'])
                        and m['evaluation_fingerprint']==fp(r['outer_evaluation_row_ids'])
                        and m['n_calibration']==r['n_calibration'] and m['n_calibration_positive']==r['n_calibration_positive'])
    check('new uncalibrated references reproduce Step33',max_raw<=1e-12,max_abs_deviation=max_raw)
    check('serialized maps reproduce all calibrated probabilities',max_map<=1e-12,max_abs_deviation=max_map)
    metrics=[]; bins=[]; folds=[]
    for (key,method,repeat),g in predictions.groupby(['pipeline','method','repeat']):
        check('4088 predictions/199 positives for each repeat/method',len(g)==4088 and int(g.y_true.sum())==199)
        metrics.append(dict(pipeline=key,method=method,repeat=int(repeat),**probability_metrics(g.y_true,g.y_proba)))
        for n,strategy in [(5,'quantile'),(10,'quantile'),(20,'quantile'),(10,'uniform')]:
            for b in reliability_bins(g.y_true,g.y_proba,n,strategy):
                bins.append(dict(pipeline=key,method=method,repeat=int(repeat),**b))
        for fold,gf in g.groupby('fold'):
            folds.append(dict(pipeline=key,method=method,repeat=int(repeat),fold=int(fold),
                              **probability_metrics(gf.y_true,gf.y_proba,coefficients=False)))
    metrics=pd.DataFrame(metrics); bins=pd.DataFrame(bins); folds=pd.DataFrame(folds)
    summary=stats(metrics,['pipeline','method'])
    write(metrics,'nested_probability_metrics_by_repeat')
    write(summary,'nested_probability_summary')
    write(folds,'nested_probability_metrics_by_outer_fold')
    write(stats(folds,['pipeline','method']),'nested_fold_mean_sd')
    write(bins,'nested_reliability_bins')
    deltas=[]
    for key in selected:
        raw=metrics[(metrics.pipeline==key)&(metrics.method=='uncalibrated_reference')].set_index('repeat')
        for method in ['sigmoid','isotonic']:
            part=metrics[(metrics.pipeline==key)&(metrics.method==method)].set_index('repeat')
            for repeat,row in part.iterrows():
                deltas.append(dict(pipeline=key,method=method,repeat=int(repeat),
                    **{'delta_'+m:float(row[m]-raw.loc[repeat,m]) for m in METRICS
                       if m in row and pd.notna(row[m]) and pd.notna(raw.loc[repeat,m])}))
    deltas=pd.DataFrame(deltas)
    write(deltas,'paired_calibration_effects_by_repeat')
    ds=deltas.groupby(['pipeline','method'])[[c for c in deltas if c.startswith('delta_')]].agg(['mean','std','min','max'])
    ds.columns=['_'.join(c) for c in ds.columns]
    write(ds.reset_index(),'paired_calibration_effects_summary')
    # Actual class counts for every newly fitted resampling pipeline.
    sampling=[]
    for row in audits.to_dict('records'):
        a=json.loads(row['class_counts_before']); b=json.loads(row['class_counts_after'])
        sampling.append({k:row[k] for k in ['pipeline','repeat','fold','phase','inner_fold']}|
            dict(n0_before=a['0'],n1_before=a['1'],n0_after=b['0'],n1_after=b['1'],achieved_ratio=b['1']/b['0']))
    write(pd.DataFrame(sampling),'nested_actual_sampling_ratios')
    unc=pd.read_csv(OUT/'uncalibrated_metrics_by_repeat.csv',float_precision='round_trip')
    us=pd.read_csv(OUT/'uncalibrated_reliability_summary.csv',float_precision='round_trip')
    ub=pd.read_csv(OUT/'uncalibrated_reliability_bins.csv',float_precision='round_trip')
    check('all original Step33/Step34 artifacts unchanged',all(sha(ROOT/k)==v for k,v in design['original_artifact_hashes'].items()))
    diagnostic=pd.concat([unc.assign(origin='Step33_descriptive',method='Step33_reference'),
                          metrics.assign(origin='nested_outer_evaluation')],ignore_index=True)
    write(diagnostic[['origin','pipeline','method','repeat','calibration_in_large_intercept','joint_intercept',
                     'calibration_slope','joint_fit_status','logit_sd','low_logit_spread','logit_clip_count']],
          'calibration_intercept_slope_diagnostics')
    # Render publication-ready static figures. Repeat ranges, never naive CIs.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,caption):
        fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
        figures.append(dict(file=name+'.png',caption=caption))
    repeat_colors=plt.get_cmap('viridis')(np.linspace(.1,.9,5))
    repeat_handles=[Line2D([0],[0],color=c,label=f'Repeat {r}') for r,c in zip(range(1,6),repeat_colors)]
    def reliability(ax,b,methods=None):
        ax.plot([0,1],[0,1],linestyle='--',color='#888888',linewidth=.8)
        if methods is None:
            for repeat,c in zip(range(1,6),repeat_colors):
                g=b[(b.repeat==repeat)&(b.binning=='quantile')&(b.requested_bins==10)&(b.n>0)]
                ax.plot(g.mean_probability,g.observed_fraction,marker='o',markersize=2,linewidth=.8,color=c)
        else:
            for method in methods:
                subset=b[(b.method==method)&(b.binning=='quantile')&(b.requested_bins==10)&(b.n>0)]
                for _,g in subset.groupby('repeat'):
                    ax.plot(g.mean_probability,g.observed_fraction,marker='o',markersize=2,linewidth=.8,
                            color=COLORS[method],alpha=.5)
        ax.grid(alpha=.15);ax.set_xlim(0,1);ax.set_ylim(0,1)
    allkeys=us.pipeline.tolist()
    fig,axes=plt.subplots(5,5,figsize=(15,14),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,allkeys):
        reliability(ax,ub[ub.pipeline==key]); ax.set_title(key.replace('logistic_regression','LR').replace('random_forest','RF').replace('decision_tree','DT').replace('svc_rbf','SVM').replace('random_under_sampler','RUS').replace('random_over_sampler','ROS').replace('class_weight_balanced','weight').replace('adasyn','CatADASYN').replace('smote','SMOTENC'),fontsize=8)
    fig.suptitle('Step 33 reference probabilities: reliability across five repeats',fontsize=15)
    fig.supxlabel('Mean predicted probability (10 quantile bins; ties collapsed)',y=.045);fig.supylabel('Observed event fraction')
    fig.legend(handles=repeat_handles,loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(.02,.07,1,.96))
    save(fig,'figure01_all_reference_reliability','All 25 configurations. Five lines show separate repeated-CV reliability estimates; counts and event counts for every bin are in the bin CSV. No independent-fold CI is implied.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        reliability(ax,ub[ub.pipeline==key]);ax.set_title(SHORT[key]);ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed fraction')
    fig.suptitle('Exploratory configuration set: unadjusted-reference reliability')
    fig.legend(handles=repeat_handles,loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,'figure02_selected_reference_reliability','Separate repeat curves; the high-probability tail may contain few positives. Quantile-bin edges differ by repeat. No pooled 20,440-patient cohort is constructed.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True)
    edges=np.linspace(0,1,31);centers=(edges[:-1]+edges[1:])/2
    for ax,key in zip(axes.flat,selected):
        for label,c in [(0,'#0072B2'),(1,'#D55E00')]:
            hist=[]
            for repeat in range(1,6):
                v=original[(original.pipeline==key)&(original.repeat==repeat)&(original.y_true==label)].y_proba
                hist.append(np.histogram(v,edges)[0]/len(v))
            h=np.asarray(hist)
            ax.plot(centers,h.mean(axis=0),color=c,label='Stroke' if label else 'No stroke')
            ax.fill_between(centers,h.min(axis=0),h.max(axis=0),color=c,alpha=.15)
        ax.set_title(SHORT[key]);ax.set_xlabel('Predicted probability');ax.set_ylabel('Within-class fraction per bin');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Reference probability distributions by outcome: mean and repeat range')
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,'figure03_probability_distributions','Each class histogram is normalized within each repeat, then averaged. Shading is min–max across five dependent repeats, not a confidence interval. Each repeat has 199 stroke and 3,889 nonstroke observations.')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric,label in zip(axes,['ece_quantile_10','brier_score','log_loss'],['ECE (10 quantile bins)','Brier score','Log loss']):
        ax.scatter(us.roc_auc_mean,us[metric+'_mean'],s=25,color='#777777')
        for key in selected:
            row=us[us.pipeline==key].iloc[0]
            ax.scatter(row.roc_auc_mean,row[metric+'_mean'],s=40,color='#0072B2')
            if key in ['logistic_regression__none','logistic_regression__class_weight_balanced']:
                ax.annotate(SHORT[key],(row.roc_auc_mean,row[metric+'_mean']),xytext=(-70,6),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Mean ROC-AUC across pooled repeats');ax.set_ylabel(label+' (lower is better)');ax.grid(alpha=.15)
    fig.suptitle('Discrimination and reliability/probability-error measures are different')
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig,'figure04_discrimination_vs_reliability','All 25 reference configurations; selected six are blue. ECE is a bin-dependent reliability estimate. Brier and log loss are overall probabilistic accuracy measures, not calibration-only quantities.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
    for ax,key in zip(axes.flat,selected):
        reliability(ax,bins[bins.pipeline==key],METHODS);ax.set_title(SHORT[key]);ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed fraction')
    handles=[Line2D([0],[0],color=COLORS[m],label=m.replace('_reference','')) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Paired nested calibration: reliability on untouched outer validation rows')
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,'figure05_nested_reliability','Every line is one outer-CV repeat pooled within repeat. Sigmoid/isotonic were trained using inner OOF scores from outer training rows only. Development-data configuration selection remains exploratory.')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,metric,title in zip(axes,['brier_score','log_loss','ece_quantile_10'],['Brier score','Log loss','ECE (10 quantile bins)']):
        for j,method in enumerate(METHODS):
            x=np.arange(6)+(j-1)*.18
            part=summary[summary.method==method].set_index('pipeline').loc[selected]
            ax.errorbar(x,part[metric+'_mean'],yerr=part[metric+'_std'],fmt='o',capsize=3,color=COLORS[method],markersize=4)
        ax.set_xticks(np.arange(6),[SHORT[k] for k in selected],rotation=45,ha='right')
        ax.set_title(title);ax.set_ylabel('Mean ± descriptive SD');ax.grid(axis='y',alpha=.15)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Paired outer-validation probability performance after calibration')
    fig.tight_layout(rect=(0,.07,1,.93))
    save(fig,'figure06_brier_logloss_ece','Mean ± sample SD of five pooled-repeat metrics; not confidence intervals. Endpoint probabilities are preserved; log loss uses epsilon=1e-15 only during scoring.')
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True)
    for ax,key in zip(axes.flat,selected):
        for method in METHODS:
            g=metrics[(metrics.pipeline==key)&(metrics.method==method)].sort_values('repeat')
            ax.plot(g.repeat,g.log_loss,marker='o',markersize=3,label=method,color=COLORS[method])
        ax.set_title(SHORT[key]);ax.set_xticks(range(1,6));ax.set_xlabel('CV repeat, seed sequence 42');ax.set_ylabel('Log loss');ax.grid(alpha=.15)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Calibration-method log-loss stability across repeated outer validation')
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,'figure07_repeated_validation_stability','No independently varied random seeds were tested. Lines connect repeated split realizations of the same cohort; ranges/SDs are descriptive.')
    # Complete report with actual numerical findings, not generic calibration claims.
    weighted=us[us.pipeline=='logistic_regression__class_weight_balanced'].iloc[0]
    none=us[us.pipeline=='logistic_regression__none'].iloc[0]
    cols=['pipeline','method','brier_score_mean','brier_score_std','log_loss_mean','log_loss_std','ece_quantile_10_mean','calibration_in_large_intercept_mean','calibration_slope_mean','roc_auc_mean','average_precision_mean']
    sections=[
        '# StrokeSure Step 35: probability calibration and reliability',
        'Analysis date: 2026-11-10 (Asia/Calcutta). Original source: Step 33 `imbalance_cv_step33_20261109T165943Z`; descriptive selection evidence: Step 34. All new outputs are separate. This is exploratory development-data research, not final held-out evaluation.',
        '## RQ4 and H3',
        '**RQ4: Does high discriminative performance necessarily correspond to well-calibrated predicted probabilities?**',
        '**H3: A model/configuration with strong ROC-AUC will not necessarily have the best probability calibration.**',
        f'In these data, similar ROC-AUC coexists with substantially different probability reliability. Class-weighted Logistic Regression has ROC-AUC {weighted.roc_auc_mean:.4f}, mean predicted risk {weighted.mean_probability_mean:.4f}, quantile ECE {weighted.ece_quantile_10_mean:.4f}, calibration-in-large intercept {weighted.calibration_in_large_intercept_mean:.4f}, and calibration slope {weighted.calibration_slope_mean:.4f}. Unweighted Logistic Regression has ROC-AUC {none.roc_auc_mean:.4f}, mean risk {none.mean_probability_mean:.4f}, ECE {none.ece_quantile_10_mean:.4f}, calibration-in-large intercept {none.calibration_in_large_intercept_mean:.4f}, and slope {none.calibration_slope_mean:.4f}. Observed prevalence is 0.048679. These descriptive findings illustrate why discrimination cannot substitute for calibration assessment and are compatible with H3; no significance or hypothesis test was performed.',
        '## Source audit and definitions',
        'Step 33 stores `pipeline, repeat, fold, row_index, y_true, y_proba`; y_proba is the positive-class probability. All 511,000 original predictions, 25 configurations and five repeats were audited. Brier is mean (y−p)^2, matching the existing evaluation implementation and Step 34 exactly. AP is average_precision_score, not trapezoidal PR-AUC. Brier and log loss measure overall probabilistic accuracy and must not be interpreted as calibration-only measures.',
        f'Log loss uses probabilities clipped to [{LOSS_EPS}, 1−{LOSS_EPS}] for scoring only. Raw predictions, including exact 0/1 isotonic values, remain unchanged in the saved CSV. Exact endpoint counts and clipped-score counts are reported per repeat. Diagnostic logits alone use [{LOGIT_EPS}, 1−{LOGIT_EPS}]; this affects intercept/slope interpretation for extreme probabilities.',
        'Primary ECE is sum_b (n_b/N)|mean(p_b)−mean(y_b)| from ten within-repeat quantile bins. Tied boundaries collapse; actual bin counts are saved. Sensitivity tables also include five/twenty quantile bins and ten equal-width bins. MCE is maximum populated-bin absolute error and can be noisy. ECE/MCE are finite-sample, bin-dependent diagnostics, not universal proper scores or inferential tests.',
        'No repeats are stacked into an independent 20,440-person sample. Each repeat metric pools only its five validation folds (4,088 patients / 199 events). Five repeat values are summarized by arithmetic mean, sample SD (ddof=1), min/max and nonmissing count. Separate outer-fold means/SDs are supplied. Repeat/fold overlap invalidates naive independent-fold confidence intervals and tests; none are computed.',
        'Reliability diagrams show five repeat curves. Bin tables retain denominators and event counts, including empty equal-width bins and sparse-event bins. No binomial CI ignoring CV/model dependence is shown. Repeated-curve spread is descriptive split variability, not a confidence band. Low positive counts in individual risk bins remain a substantive limitation.',
        'Calibration-in-large intercept is the MLE log-odds offset with logit(p) slope fixed at one (ideal 0). Joint intercept/slope are separate unpenalized logistic diagnostic fits (ideal intercept 0, slope 1). They are fitted to evaluated outcomes only to describe reliability, never applied to adjust probabilities or used as estimates of predictive calibration gain. Nonestimable fits are flagged; low logit spread (<0.05) is flagged even when a numerical estimate exists.',
        'The Dummy reference has almost constant probabilities within folds. Its pooled diagnostic slope can be negative/unstable because fold priors differ slightly; this is not meaningful evidence of an intrinsically inverted clinical risk relationship. Do not interpret its slope as a reliable model ranking.',
        '## All 25 original-reference configurations',
        table(us,['pipeline','roc_auc_mean','average_precision_mean','brier_score_mean','log_loss_mean','ece_quantile_10_mean','mean_probability_mean','calibration_in_large_intercept_mean','calibration_slope_mean']),
        '## Exploratory configuration set and justification',
        table(pd.read_csv(OUT/'exploratory_configuration_set.csv'),['pipeline','exploratory_reason']),
        'The set includes natural-prevalence LR/RF references, weighting, undersampling, custom synthetic sampling and a nonlinear margin model. It was not chosen by a single metric. Configurations were chosen using earlier CV evidence on these same development patients; the later nested comparisons remove calibration-fitting leakage but do not remove configuration-selection optimism. No final winner is declared. No SMOTENC configuration was added solely to fill a grid; the six configurations are an exploratory examination set, not an exhaustive calibration comparison.',
        'The SVC reference already uses probability=True and internal Platt probability estimation. Here uncalibrated_reference means no *additional Step 35* calibration. External sigmoid/isotonic maps use its decision margin, as in the pinned sklearn calibration response rule. LR also uses its decision_function; RF uses positive-class predict_proba. All original classifier/preprocessing/sampling hyperparameters were preserved.',
        '## Leakage-safe fitted comparison',
        'The original shared 5-fold × 5-repeat outer plan is reused. For each configuration and outer fold, five inner stratified folds (shuffle=True, random_state=42) generate one held-out score for every outer-training patient. All preprocessing, imputation, scaling and sampling fit only on each inner training fold. The sigmoid/isotonic map sees the 3,270/3,271 natural-prevalence inner OOF scores and their 159/160 positive outcomes, never resampled calibration labels or outer validation targets. The base estimator is then refitted on the entire outer-training fold, and all three methods are evaluated on the same untouched 817/818 outer validation rows (39/40 events).',
        'Base fits are shared across the two maps: 750 inner base fits plus 150 outer refits = 900 successful base fits. There are 150 sigmoid maps and 150 isotonic maps = 300 fitted maps, 1,200 predictive fit calls in total. These are six configurations, not a rerun of the original 625-fit grid. Diagnostic intercept/slope estimations are counted separately below. All maps, split identities, inner scores, outer probabilities, fit records, warnings/errors and sampling audits are preserved.',
        'Sigmoid uses the pinned sklearn Platt implementation with target smoothing; its source hash and a/b parameters are recorded. Isotonic uses increasing=True, y_min=0, y_max=1 and out_of_bounds=clip. With more than 3,000 calibration scores isotonic is feasible as an exploratory comparator, but 159/160 events do not guarantee stable tail estimation; stepwise probabilities, ties and exact endpoints can cause poor log loss. Its effective information is not 3,270 independent events. We therefore assess it on held-out outer predictions and do not assume it improves reliability.',
        'Cross-fitted calibration scores come from base models trained on roughly 80% of outer-training data; the final base refit uses all outer-training rows. This standard cross-fit/refit design can have score-distribution mismatch. Fold-specific maps can change pooled ranking metrics even when each map is monotone. Within-fold isotonic ties can also change AUC/AP; probability gains must be assessed jointly with discrimination.',
        '## Paired outer-validation results',
        table(summary,cols),
        'All values are means of five pooled-repeat scores; SD/range and paired changes are supplied in CSVs. Delta is calibrated minus its paired uncalibrated reference. Negative Brier/log-loss/ECE deltas favor that diagnostic, but no single composite objective or clinical preference was optimized.',
    ]
    for key in selected:
        part=summary[summary.pipeline==key].set_index('method')
        raw=part.loc['uncalibrated_reference'];sig=part.loc['sigmoid'];iso=part.loc['isotonic']
        endpoint=metrics[(metrics.pipeline==key)&(metrics.method=='isotonic')]
        sections += [f'### {SHORT[key]}',
            f'Brier: reference {raw.brier_score_mean:.4f}, sigmoid {sig.brier_score_mean:.4f} (Δ {sig.brier_score_mean-raw.brier_score_mean:+.4f}), isotonic {iso.brier_score_mean:.4f} (Δ {iso.brier_score_mean-raw.brier_score_mean:+.4f}). Log loss: {raw.log_loss_mean:.4f}, {sig.log_loss_mean:.4f}, {iso.log_loss_mean:.4f}, respectively. Ten-quantile ECE: {raw.ece_quantile_10_mean:.4f}, {sig.ece_quantile_10_mean:.4f}, {iso.ece_quantile_10_mean:.4f}. Isotonic exact-zero predictions range {int(endpoint.probability_zero_count.min())}–{int(endpoint.probability_zero_count.max())} per repeat and exact-one predictions {int(endpoint.probability_one_count.min())}–{int(endpoint.probability_one_count.max())}. These observed endpoint effects are retained, not discarded.']
    rfpart=summary[summary.pipeline=='random_forest__none'].set_index('method')
    lrpart=summary[summary.pipeline=='logistic_regression__none'].set_index('method')
    sections += [
        '### Calibration gains are configuration- and metric-dependent',
        f'For the uncorrected RF reference, sigmoid worsens Brier from {rfpart.loc["uncalibrated_reference","brier_score_mean"]:.5f} to {rfpart.loc["sigmoid","brier_score_mean"]:.5f} and log loss from {rfpart.loc["uncalibrated_reference","log_loss_mean"]:.5f} to {rfpart.loc["sigmoid","log_loss_mean"]:.5f}. The natural-prevalence reference was already comparatively reliable; post-hoc calibration is not automatically beneficial.',
        f'For unweighted LR, isotonic reduces ten-quantile ECE from {lrpart.loc["uncalibrated_reference","ece_quantile_10_mean"]:.5f} to {lrpart.loc["isotonic","ece_quantile_10_mean"]:.5f}, yet log loss worsens from {lrpart.loc["uncalibrated_reference","log_loss_mean"]:.5f} to {lrpart.loc["isotonic","log_loss_mean"]:.5f}. Coarse binning, ties and tail predictions can make binned ECE look favorable while probabilistic accuracy deteriorates. The results do not justify a universal calibration-method winner.',
        '## Stability and sample-size limitations',
        'Repeated-validation tables contain all 90 method/configuration/repeat reliability records and 450 outer-fold score records. Bin sensitivity (5/10/20 quantile and 10 equal-width) and per-repeat coefficient estimates are saved. Differences between fold SD and pooled-repeat SD reflect different validation sizes/estimands and dependence; they are not a variance-reduction test. No independently varied random seeds or external validation were evaluated.',
        f'Diagnostic coefficient requests: {len(diagnostic)} joint intercept/slope fits and {len(diagnostic)} calibration-in-large estimates (125 original-reference repeats plus 90 new outer-evaluation repeats). Estimability statuses are retained rather than silently dropping unsuccessful cases: {diagnostic.joint_fit_status.value_counts().to_dict()}. Diagnostic coefficient counts are not additional predictive-model fits. Per-metric summary count columns disclose missing values.',
        '## RQ4/H3 interpretation for the paper',
        'Across these internal CV configurations, strong discrimination did not ensure probability reliability. Class-weighted Logistic Regression retained ROC-AUC close to the unweighted reference while substantially overpredicting natural-prevalence event risk. Held-out nested comparisons quantify whether post-hoc calibration improves probability scores without fitting to evaluation outcomes. These findings are descriptive and compatible with H3. They do not establish statistical significance, a uniquely best calibration method, clinical transportability or final model selection. Brier/log-loss improvements alone do not prove perfect calibration; curves, bin counts, intercept/slope and repeat variability remain necessary.',
        '## Verification and provenance',
        f'Executor counters: {execution["counters"]}. Sampling audits: {execution["sampling_audit_count"]}/450 passing. Expected SVC deprecation warnings: {execution["expected_warning_count"]}; unexpected warnings: {execution["unexpected_warning_count"]}; fit/calibration errors: {execution["error_count"]}. Independent replay verifies every split role, one inner OOF score per calibration row, paired evaluation identities, reference reproduction and serialized map predictions. Maximum raw-reference deviation from Step 33 is {max_raw:.3g}; maximum map-replay deviation is {max_map:.3g}.',
        f'Nested experiment execution time: {observations["seconds"]:.2f} seconds. Sampled RSS observations: {observations["memory"]}. These cover nested execution, with a 20 ms probe that can miss brief spikes; they are not a guaranteed future RAM requirement.',
        'No locked held-out data fields were accessed. No Step 33 or Step 34 files were changed, and no source module or model/sampler hyperparameter was altered. Additional fitted results live only here. No classification threshold was selected, and no SHAP, final-model evaluation, statistical hypothesis test, Git commit or push was performed.',
        '## Figures',
    ]
    for item in figures: sections += [f'![{item["caption"]}]({item["file"]})',item['caption']]
    sections += ['## Recommendations for the subsequent threshold stage',
        '1. Agree an exploratory model/calibration examination set using paired probability results, discrimination and stability together; do not choose solely by minimum ECE or Brier.',
        '2. Specify clinical utility/costs and a threshold-selection protocol before optimization. Use genuinely held-out inner training/calibration data to choose thresholds and reserve outer evaluation for performance assessment.',
        '3. Keep the final test partition locked through development decisions. Do not choose a threshold using the already evaluated outer OOF outcomes and report performance on those same outcomes as unbiased.',
        '4. Retain unadjusted references, natural prevalence, sparse-bin counts and uncertainty caveats. Investigate isotonic endpoint behavior and bin sensitivity before relying on tail risks.',
        '5. Any future formal comparisons must respect paired, correlated repeats and multiplicity. These outputs supply descriptive evidence, not inferential p-values or deployment readiness.',
        '## Artifact guide and reproduction',
        'Original-reference reliability: `uncalibrated_metrics_by_repeat.csv`, `uncalibrated_reliability_summary.csv`, `uncalibrated_reliability_bins.csv`. Coefficients: `calibration_intercept_slope_diagnostics.csv`. New experiments: `nested/` contains predictions, all inner calibration scores, exact split row identities, 900 base fit records, 300 serialized calibration maps, 450 sampling audits, warnings/errors, frozen manifests and execution observations. Comparisons: `nested_probability_summary.csv`, repeat/fold metric tables and paired-effect tables. `artifact_inventory.csv` supplies output hashes; `verification.json` records independent checks; `figure_captions.json` describes the seven 300-dpi PNGs.',
        'To reproduce numerical analysis/figures without fitting, run `python results/step35_calibration/step35_analyze.py` in the project virtual environment. The nested-fitting script intentionally refuses an existing nested directory; it must not be rerun over these results. Plot dependencies are read from the isolated Step 34 plotting directory; no package or experiment source was changed during Step 35.',
    ]
    report_path=OUT/'scientific_calibration_report.md'
    report_path.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    dump(figures,OUT/'figure_captions.json')
    check('source artifacts still unchanged after plotting',all(sha(ROOT/k)==v for k,v in design['original_artifact_hashes'].items()))
    verdict=dict(overall_passed=all(r['passed'] for r in CHECKS),n_checks=len(CHECKS),
                 n_passed=sum(r['passed'] for r in CHECKS),n_failed=sum(not r['passed'] for r in CHECKS),checks=CHECKS)
    dump(verdict,OUT/'verification.json')
    with report_path.open('a',encoding='utf-8') as handle:
        handle.write(f'\n\nFinal independent verification: **{verdict["n_passed"]}/{verdict["n_checks"]} PASS**, {verdict["n_failed"]} failed checks. All original Step 33/34 artifact hashes remain unchanged.\n')
    dump(dict(source_run=design['source_run'],configurations_descriptive=25,configurations_fitted=6,
        counts=execution['counters'],diagnostic_coefficient_requests=len(diagnostic),
        source_artifact_hashes=design['original_artifact_hashes'],
        analysis_source_hashes={p.name:sha(p) for p in OUT.glob('*.py')},
        methods=METHODS,score_clips=dict(log_loss=LOSS_EPS,diagnostic_logit=LOGIT_EPS),
        numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
        scikit_learn=sys.modules['sklearn'].__version__,scipy=sys.modules['scipy'].__version__,
        verification={k:v for k,v in verdict.items() if k!='checks'},figures=figures),OUT/'analysis_manifest.json')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and '_matplotlib_config' not in p.parts and '__pycache__' not in p.parts and p.name!='artifact_inventory.csv' and not p.name.endswith('_execution.log'):
            inventory.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(inventory),'artifact_inventory')
    print(json.dumps({k:v for k,v in verdict.items() if k!='checks'},indent=2),flush=True)
    print(summary[cols].to_string(index=False),flush=True)

if __name__=='__main__': main()
