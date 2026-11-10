"""Focused training-only SHAP, fixed representative models; no evaluation/tuning."""
from pathlib import Path
import sys,os,json,time,hashlib,warnings,itertools,traceback
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'results/step34_imbalance_analysis/_plot_dependencies'))
os.environ['MPLCONFIGDIR']=str(OUT/'_matplotlib_config')
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from exact_shap import shapley_from_coalitions,enumerate_coalitions
from scipy.special import expit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.imbalance_pipelines import build_leakage_safe_pipeline
from src.config import RAW_DATA_PATH,FEATURE_COLUMNS,NUMERIC_FEATURES,BINARY_FEATURES,CATEGORICAL_FEATURES

RUN='imbalance_cv_step33_20261109T165943Z'
S33=ROOT/'results/imbalance_cv'/RUN
KEYS=['logistic_regression__none','logistic_regression__class_weight_balanced','random_forest__none']
LABELS=['LR uncorrected','LR class-weighted','RF uncorrected']
CHECKS=[];WARNINGS=[];FIGURES=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(x,name):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def check(name,ok,**info):
    CHECKS.append(dict(check=name,passed=bool(ok),**info))
    if not ok:raise AssertionError((name,info))
def write(df,name):df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
def snapshot():
    dirs=[S33]+[ROOT/'results'/n for n in ['step34_imbalance_analysis','step35_calibration','step36_threshold_analysis','step37_statistical_robustness']]+[ROOT/'src']
    return {str(p.relative_to(ROOT)):sha(p) for d in dirs for p in sorted(d.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['_plot_dependencies','_dependencies','_matplotlib_config','__pycache__'])}
def savefig(name,caption):
    plt.savefig(OUT/name,dpi=300,bbox_inches='tight');plt.close()
    FIGURES.append(dict(file=name,caption=caption))
def table(df):
    def cell(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    start=time.perf_counter();before=snapshot()
    check('fresh scientific output',not (OUT/'analysis_manifest.json').exists())
    # Validate all existing integrity references without changing their files.
    previous=json.loads((ROOT/'results/step37_statistical_robustness/analysis_manifest.json').read_text())
    for p,h in previous['original_artifact_hashes'].items():check('prior artifact hash '+p,sha(ROOT/p)==h)
    for stage,path in [('33',S33/(RUN+'_verification.json')),('34',ROOT/'results/step34_imbalance_analysis/verification.json'),('35',ROOT/'results/step35_calibration/verification.json'),('36',ROOT/'results/step36_threshold_analysis/verification.json'),('37',ROOT/'results/step37_statistical_robustness/verification.json')]:
        v=json.loads(path.read_text());check('verified Step '+stage,v.get('overall_passed',v.get('passed',False)))
    plan=dict(models=KEYS,selection='exploratory representatives, not winners; same preferred set as user',model_seed=42,
        fits=3,training_rows=4088,explanation_n=256,background_n=32,
        sample_seed=42,alternate_background_seed=43,importance_bootstrap_seed=38042,importance_bootstrap_draws=1000,
        scales={'logistic_regression':'positive-class decision_function log-odds','random_forest':'positive-class uncalibrated probability'},
        local_selection='common patients nearest RF sample probability quantiles .10,.50,.90; no outcome filtering; ties by row ID',
        reference='natural-prevalence simple random samples from training IDs excluding explanation IDs; no label balancing',
        explainers='exact analytic linear SHAP and exact model-agnostic finite-background enumeration of all 1024 coalitions; no SHAP library execution',
        aggregation='original-feature players; whole categorical block masked jointly. LR analytic transformed SHAP sums are exact grouped-game SHAP by additivity',
        adaptation='SHAP/Numba DLLs blocked by Windows application control, including escalated attempt; initial script and logs retained, no fits occurred before adaptation')
    dump(plan,'analysis_plan.json')
    # Reuse verified Step33 membership. Skip held-out CSV rows before parsing;
    # no full-data load, resplit, held-out labels or held-out features are used.
    oof=pd.read_csv(S33/(RUN+'_oof_predictions.csv'),usecols=['pipeline','repeat','row_index','y_true'])
    members=oof[(oof.pipeline==KEYS[0])&(oof['repeat']==1)].sort_values('row_index')
    ids=members.row_index.to_numpy(dtype=int);allowed=set(ids.tolist())
    check('exact unique training membership',len(ids)==4088 and len(allowed)==4088)
    Xy=pd.read_csv(RAW_DATA_PATH,skiprows=lambda line:line>0 and (line-1) not in allowed,usecols=FEATURE_COLUMNS+['stroke'])
    Xy.index=ids;X=Xy[FEATURE_COLUMNS];y=Xy.stroke
    check('training labels match verified OOF',np.array_equal(y.to_numpy(),members.y_true.to_numpy()) and int(y.sum())==199)
    rng=np.random.default_rng(42);explain_ids=np.sort(rng.choice(ids,256,replace=False))
    rest=np.array(sorted(allowed-set(explain_ids)));bg=np.sort(rng.choice(rest,32,replace=False))
    alt=np.sort(np.random.default_rng(43).choice(rest,32,replace=False))
    check('reproducible explanation selection',np.array_equal(explain_ids,np.sort(np.random.default_rng(42).choice(ids,256,replace=False))))
    for role,rows in [('explanation',explain_ids),('background_primary',bg),('background_alternate',alt)]:
        check('training-only '+role,set(rows)<=allowed and len(np.unique(rows))==len(rows))
    check('backgrounds exclude explanation rows',not set(explain_ids)&(set(bg)|set(alt)))
    selections=pd.concat([pd.DataFrame(dict(role=role,row_index=rows,y_true=y.loc[rows].to_numpy())) for role,rows in [('explanation',explain_ids),('background_primary',bg),('background_alternate',alt)]])
    write(selections,'selected_observations');write(pd.DataFrame(dict(row_index=ids,y_true=y)),'training_membership')
    check('no ID or target predictors','id' not in X and 'stroke' not in X and len(X.columns)==10)
    check('no synthetic data',all(k.endswith('__none') or k.endswith('__class_weight_balanced') for k in KEYS))
    model_results={};mapping=[];importance=[];predictions=[];additivity=[];params={};timings=[]
    for key,label in zip(KEYS,LABELS):
        print('FITTING',key,flush=True);tick=time.perf_counter()
        model=build_leakage_safe_pipeline(*key.split('__'),random_state=42,sampling_strategy=.5,k_neighbors=5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X,y)
        WARNINGS.extend(dict(model=key,phase='fit',category=w.category.__name__,message=str(w.message)) for w in caught)
        clf=model.named_steps['classifier'];prep=model.named_steps['preprocessor']
        params[key]=clf.get_params();names=list(prep.get_feature_names_out());z=prep.transform(X.loc[explain_ids]);b=prep.transform(X.loc[bg]);a=prep.transform(X.loc[alt])
        is_lr=key.startswith('logistic');scale='log_odds' if is_lr else 'probability'
        check('preprocessor row count '+key,int(prep.named_transformers_['num'].named_steps['imputer'].n_features_in_)==3)
        check('classifier class order '+key,np.array_equal(clf.classes_,[0,1]))
        check('transformed feature count '+key,z.shape==(256,16 if is_lr else 21) and len(names)==z.shape[1])
        # Map exact preprocessor prefixes; categorical variable names may contain underscores.
        groups=[]
        for j,name in enumerate(names):
            matches=[f for f in FEATURE_COLUMNS if name in ['num__'+f,'bin__'+f] or (f in CATEGORICAL_FEATURES and name.startswith('cat__'+f+'_'))]
            check('unique feature mapping '+key+' '+name,len(matches)==1)
            groups.append(matches[0]);mapping.append(dict(model=key,transformed_index=j,transformed_name=name,clinical_group=matches[0],output_scale=scale))
        check('all clinical groups represented '+key,set(groups)==set(FEATURE_COLUMNS))
        for f in CATEGORICAL_FEATURES:
            cols=[i for i,g in enumerate(groups) if g==f]
            check('valid categorical one-hot '+key+' '+f,np.isin(z[:,cols],[0,1]).all() and np.all(z[:,cols].sum(axis=1)<=1))
        prob=model.predict_proba(X.loc[explain_ids])[:,1]
        target=model.decision_function(X.loc[explain_ids]) if is_lr else prob
        check('pipeline/classifier probability alignment '+key,np.allclose(prob,clf.predict_proba(z)[:,1],atol=1e-14,rtol=0))
        variants={}
        for variant,background in [('primary',b),('alternate',a)]:
            print('EXPLAINING',key,variant,flush=True)
            column_groups=np.array([FEATURE_COLUMNS.index(g) for g in groups])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if is_lr:
                    phi=(z-background.mean(axis=0))*clf.coef_[0]
                    base=np.full(256,float(clf.decision_function(background).mean()))
                    grouped=np.column_stack([phi[:,np.array(groups)==f].sum(axis=1) for f in FEATURE_COLUMNS])
                    # Independent exhaustive original-feature game on three real rows.
                    small=enumerate_coalitions(clf.decision_function,z[:3],background,column_groups,10,batch_size=64)
                    check('analytic LR equals exhaustive grouped SHAP '+key+' '+variant,np.allclose(shapley_from_coalitions(small,10),grouped[:3],atol=1e-12,rtol=0))
                else:
                    def progress(n,total):
                        if n%128==0 or n==total:print('COALITIONS',variant,n,'/',total,flush=True)
                    values=enumerate_coalitions(lambda q:clf.predict_proba(q)[:,1],z,background,column_groups,10,progress=progress)
                    np.savez_compressed(OUT/(key+'__'+variant+'__coalition_values.npz'),values=values,row_index=explain_ids)
                    grouped=shapley_from_coalitions(values,10);phi=grouped;base=values[:,0]
                    check('RF endpoint probabilities '+variant,np.allclose(values[:,-1],prob,atol=1e-14,rtol=0))
                    # Verify masked encoded inputs match actual raw original-feature swaps.
                    donor_ids=bg if variant=='primary' else alt
                    for mask in [0,1,341,682,1023]:
                        hybrid=X.loc[[explain_ids[0]]].copy()
                        for j,f in enumerate(FEATURE_COLUMNS):
                            if not mask&(1<<j):hybrid.loc[:,f]=X.loc[donor_ids[0],f]
                        expected=np.where(((mask>>column_groups)&1).astype(bool),z[0],background[0])
                        check('whole-group masking preserves preprocessing '+variant+' '+str(mask),np.allclose(prep.transform(hybrid)[0],expected,atol=1e-12,rtol=0))
            WARNINGS.extend(dict(model=key,phase=variant,category=w.category.__name__,message=str(w.message)) for w in caught)
            check('finite SHAP and base '+key+' '+variant,np.isfinite(phi).all() and np.isfinite(base).all() and grouped.shape==(256,10) and base.shape==(256,))
            reconstruction=base+phi.sum(axis=1);residual=np.abs(reconstruction-target)
            check('SHAP additivity '+key+' '+variant,float(residual.max())<1e-6,max_absolute_residual=float(residual.max()))
            if is_lr:check('log-odds probability reconstruction '+key+' '+variant,np.allclose(expit(reconstruction),prob,atol=1e-10,rtol=0))
            check('signed group aggregation '+key+' '+variant,np.allclose(grouped.sum(axis=1),phi.sum(axis=1),atol=1e-12,rtol=0))
            variants[variant]=dict(phi=phi,base=base,grouped=grouped)
            if is_lr:
                df=pd.DataFrame(phi,columns=names);df.insert(0,'row_index',explain_ids);write(df,key+'__'+variant+'__transformed_shap')
            df=pd.DataFrame(grouped,columns=FEATURE_COLUMNS);df.insert(0,'row_index',explain_ids);write(df,key+'__'+variant+'__grouped_shap')
            additivity.append(dict(model=key,background=variant,scale=scale,n=256,max_absolute_residual=float(residual.max()),mean_absolute_residual=float(residual.mean())))
            meanabs=np.abs(grouped).mean(axis=0);ranks=pd.Series(meanabs).rank(ascending=False,method='average').to_numpy()
            for j,f in enumerate(FEATURE_COLUMNS):importance.append(dict(model=key,background=variant,feature=f,scale=scale,mean_abs_grouped_shap=float(meanabs[j]),mean_signed_grouped_shap=float(grouped[:,j].mean()),normalized_importance=float(meanabs[j]/meanabs.sum()),rank=float(ranks[j]),mean_sum_abs_dummy_shap=float(np.abs(phi[:,np.array(groups)==f]).sum(axis=1).mean()) if is_lr else None))
            write(pd.DataFrame(dict(row_index=explain_ids,base_value=base,model_output=target,reconstructed_output=reconstruction,residual=reconstruction-target,positive_probability=prob)),key+'__'+variant+'__additivity_rows')
        # Re-evaluate primary explainer on a few rows, no refit; deterministic reproducibility.
        if is_lr:
            replay=(z[:8]-b.mean(axis=0))*clf.coef_[0];reference=variants['primary']['phi'][:8]
        else:
            v=enumerate_coalitions(lambda q:clf.predict_proba(q)[:,1],z[:1],b,column_groups,10,batch_size=64)
            replay=shapley_from_coalitions(v,10);reference=variants['primary']['grouped'][:1]
        check('SHAP replay '+key,np.allclose(replay,reference,atol=1e-12,rtol=0))
        predictions.append(pd.DataFrame(dict(model=key,row_index=explain_ids,y_true=y.loc[explain_ids].to_numpy(),positive_probability=prob,model_output=target,scale=scale)))
        model_results[key]=dict(**variants,z=z,names=names,groups=groups,prob=prob,target=target,label=label,scale=scale)
        joblib.dump(model,OUT/(key+'__training_fit.joblib'))
        timings.append(dict(model=key,seconds=time.perf_counter()-tick));print('DONE',key,timings[-1]['seconds'],flush=True)
    importance=pd.DataFrame(importance);write(importance,'global_grouped_importance');write(pd.DataFrame(mapping),'feature_mapping');write(pd.concat(predictions),'explained_training_predictions');write(pd.DataFrame(additivity),'additivity_summary');write(pd.DataFrame(timings),'model_timings');dump(params,'estimator_parameters.json')
    # Conditional explanation-row resampling: sensitivity of global ranks, no model fits.
    bootstrap_rng=np.random.default_rng(38042);draws=bootstrap_rng.integers(0,256,size=(1000,256));stability=[];bg_sensitivity=[]
    for key,r in model_results.items():
        absolute=np.abs(r['primary']['grouped']);estimates=absolute[draws].mean(axis=1)
        rankdraws=pd.DataFrame(estimates).rank(axis=1,ascending=False,method='average').to_numpy()
        lo,hi=np.quantile(estimates,[.025,.975],axis=0)
        for j,f in enumerate(FEATURE_COLUMNS):stability.append(dict(model=key,feature=f,scale=r['scale'],mean_abs_shap=float(absolute[:,j].mean()),conditional_row_resampling_p025=float(lo[j]),conditional_row_resampling_p975=float(hi[j]),mean_rank=float(rankdraws[:,j].mean()),min_rank=float(rankdraws[:,j].min()),max_rank=float(rankdraws[:,j].max()),top3_fraction=float((rankdraws[:,j]<=3).mean()),n_draws=1000))
        p=importance[(importance.model==key)&(importance.background=='primary')].set_index('feature').loc[FEATURE_COLUMNS]
        a=importance[(importance.model==key)&(importance.background=='alternate')].set_index('feature').loc[FEATURE_COLUMNS]
        bg_sensitivity.append(dict(model=key,spearman_rank_correlation=float(spearmanr(p['rank'],a['rank']).statistic),top3_overlap=len(set(p.nsmallest(3,'rank').index)&set(a.nsmallest(3,'rank').index)),max_normalized_importance_change=float(abs(p.normalized_importance-a.normalized_importance).max())))
    write(pd.DataFrame(stability),'conditional_importance_rank_stability');write(pd.DataFrame(bg_sensitivity),'background_sensitivity')
    cross=[]
    primary=importance[importance.background=='primary']
    for a,b in itertools.combinations(KEYS,2):
        pa=primary[primary.model==a].set_index('feature').loc[FEATURE_COLUMNS];pb=primary[primary.model==b].set_index('feature').loc[FEATURE_COLUMNS]
        cross.append(dict(model_a=a,model_b=b,spearman_rank_correlation=float(spearmanr(pa['rank'],pb['rank']).statistic),top3_overlap=len(set(pa.nsmallest(3,'rank').index)&set(pb.nsmallest(3,'rank').index))))
    write(pd.DataFrame(cross),'model_rank_comparisons')
    # Local exemplars fixed by RF sample probability quantiles, common to all models.
    rf=model_results[KEYS[2]]['prob'];local=[];localrows=[]
    for quantile in [.1,.5,.9]:
        value=float(np.quantile(rf,quantile));j=int(np.argmin(abs(rf-value)));row=int(explain_ids[j]);local.append((quantile,j,row))
        for key,r in model_results.items():
            for k,f in enumerate(FEATURE_COLUMNS):localrows.append(dict(model=key,rf_quantile=quantile,row_index=row,y_true=int(y.loc[row]),positive_probability=float(r['prob'][j]),base_value=float(r['primary']['base'][j]),model_output=float(r['target'][j]),scale=r['scale'],feature=f,observed_value=str(X.loc[row,f]),signed_shap=float(r['primary']['grouped'][j,k])))
    check('distinct local quantile exemplars',len({r for _,_,r in local})==3)
    write(pd.DataFrame(localrows),'local_explanations');write(X.loc[[r for _,_,r in local]].reset_index(names='row_index'),'local_observed_features')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    # Absolute global importance, separate scales; bootstrap bars conditional only.
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    stab=pd.DataFrame(stability)
    for ax,key in zip(axes,KEYS):
        s=stab[stab.model==key].sort_values('mean_abs_shap');ax.barh(s.feature,s.mean_abs_shap,color='#276c8e')
        ax.errorbar(s.mean_abs_shap,s.feature,xerr=np.vstack([np.maximum(0,s.mean_abs_shap-s.conditional_row_resampling_p025),np.maximum(0,s.conditional_row_resampling_p975-s.mean_abs_shap)]),fmt='none',ecolor='black',capsize=2)
        ax.set(title=model_results[key]['label'],xlabel='Mean |signed group sum| ('+model_results[key]['scale']+')')
    fig.suptitle('Training-sample global importance; conditional explanation-row resampling intervals');fig.tight_layout()
    savefig('figure01_global_importance.png','Mean absolute signed group-summed SHAP over the same 256 training observations; whiskers are conditional 2.5/97.5 row-resampling percentiles, excluding refitting/background uncertainty. LR and RF have different units.')
    # Original-feature signed dot summaries; categorical values have no artificial order.
    for number,(key,r) in enumerate(model_results.items(),2):
        fig,ax=plt.subplots(figsize=(10,7));order=np.argsort(np.abs(r['primary']['grouped']).mean(axis=0));jitter=np.random.default_rng(42).uniform(-.22,.22,(256,10))
        for k,j in enumerate(order):
            f=FEATURE_COLUMNS[j]
            if f in NUMERIC_FEATURES+BINARY_FEATURES:
                values=X.loc[explain_ids,f].to_numpy(dtype=float);values=np.where(np.isnan(values),np.nanmedian(values),values);span=np.ptp(values);color=(values-values.min())/span if span else np.zeros(256)
                ax.scatter(r['primary']['grouped'][:,j],k+jitter[:,j],c=color,cmap='coolwarm',vmin=0,vmax=1,s=12,alpha=.6)
            else:ax.scatter(r['primary']['grouped'][:,j],k+jitter[:,j],s=12,alpha=.5,color='gray')
        ax.set_yticks(range(10),np.array(FEATURE_COLUMNS)[order]);ax.axvline(0,color='gray',ls='--');ax.set(title=r['label']+' — original-feature SHAP',xlabel='Signed contribution ('+r['scale']+')')
        fig.text(.1,.01,'Continuous/binary: blue low, red high (BMI missing coloured at sample median); categorical: gray, unordered.',fontsize=9);fig.tight_layout(rect=[0,.035,1,1])
        savefig(f'figure0{number}_summary_{key}.png','Original-feature SHAP signed summary for '+r['label']+'. Each dot is one fitted-training observation. Categorical colours are unordered gray; colour-imputed BMI is for display only and is not preprocessing.')
    # Grouped signed summary with deterministic jitter and no misleading categorical ordering.
    fig,axes=plt.subplots(1,3,figsize=(15,6));jitter=np.random.default_rng(42).uniform(-.23,.23,size=(256,10))
    for ax,key in zip(axes,KEYS):
        r=model_results[key];order=np.argsort(np.abs(r['primary']['grouped']).mean(axis=0))
        for k,j in enumerate(order):ax.scatter(r['primary']['grouped'][:,j],k+jitter[:,j],s=5,alpha=.35,color='#276c8e',rasterized=True)
        ax.set_yticks(range(10),np.array(FEATURE_COLUMNS)[order]);ax.axvline(0,color='gray',ls='--');ax.set(title=r['label'],xlabel='Signed grouped SHAP ('+r['scale']+')')
    fig.suptitle('Original-feature signed contributions; no categorical-value colour ordering');fig.tight_layout()
    savefig('figure05_grouped_signed_summary.png','Original-feature signed SHAP contributions. LR uses the additive group equivalence; RF computes the original-feature coalition game directly.')
    fig,axes=plt.subplots(3,3,figsize=(17,14))
    for col,key in enumerate(KEYS):
        r=model_results[key]
        for row,(quantile,j,patient) in enumerate(local):
            ax=axes[row,col];v=r['primary']['grouped'][j];order=np.argsort(abs(v));ax.barh(np.array(FEATURE_COLUMNS)[order],v[order],color=np.where(v[order]>=0,'#be4b45','#276c8e'));ax.axvline(0,color='gray',lw=.8)
            ax.set(title=f'{r["label"]}\nrow {patient}; RF q{quantile:.1f}; p={r["prob"][j]:.4f}',xlabel=f'Signed {r["scale"]} contribution')
            ax.text(.02,-.20,f'base {r["primary"]["base"][j]:.4f} + sum {v.sum():.4f} = {r["target"][j]:.4f}',transform=ax.transAxes,fontsize=9)
    fig.suptitle('Common real training patients near RF probability deciles; not outcome-selected');fig.tight_layout(rect=[0,0,1,.97],h_pad=4)
    savefig('figure06_local_explanations.png','Three common training observations near RF sample probability quantiles 0.10, 0.50 and 0.90; bars show signed grouped contributions. Baseline plus contributions reconstructs model output. They illustrate prediction strata, not population-representative clinical phenotypes.')
    pivot=primary.pivot(index='feature',columns='model',values='normalized_importance').loc[FEATURE_COLUMNS,KEYS]
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(pivot,aspect='auto',cmap='Blues',vmin=0,vmax=float(pivot.max().max()))
    ax.set_xticks(range(3),LABELS);ax.set_yticks(range(10),FEATURE_COLUMNS)
    for j in range(10):
        for k in range(3):ax.text(k,j,f'{pivot.iloc[j,k]:.1%}',ha='center',va='center',color='white' if pivot.iloc[j,k]>.3 else 'black')
    fig.colorbar(im,ax=ax,label='Within-model normalized mean absolute grouped contribution');ax.set_title('Relative importance shares; raw magnitudes are not cross-scale comparable');fig.tight_layout()
    savefig('figure07_model_comparison.png','Each column sums to one. Normalization permits descriptive comparison of relative importance allocations; it does not make log-odds and probability explanations the same estimand.')
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    for ax,key in zip(axes,KEYS):
        ss=primary[primary.model==key].sort_values('rank');aa=importance[(importance.model==key)&(importance.background=='alternate')].set_index('feature').loc[ss.feature]
        ax.plot(ss['rank'],range(10),'o-',label='Primary background');ax.plot(aa['rank'],range(10),'s--',label='Alternate background');ax.set_yticks(range(10),ss.feature);ax.invert_yaxis();ax.set(title=model_results[key]['label'],xlabel='Global importance rank',xticks=range(1,11));ax.legend(fontsize=8)
    fig.suptitle('Sensitivity to two natural-prevalence 32-row training backgrounds');fig.tight_layout()
    savefig('figure08_background_rank_sensitivity.png','Same fitted models and same 256 explained observations, two 32-row backgrounds. This assesses reference-sample sensitivity, not training-seed, CV-fold or across-dataset stability.')
    check('all existing artifacts and sources unchanged',before==snapshot())
    check('no unexpected warnings',len(WARNINGS)==0,warnings=WARNINGS)
    dump(WARNINGS,'warning_records.json');dump([],'error_records.json');dump(FIGURES,'figure_captions.json')
    result=dict(overall_passed=all(c['passed'] for c in CHECKS),n_checks=len(CHECKS),n_passed=sum(c['passed'] for c in CHECKS),n_failed=sum(not c['passed'] for c in CHECKS),checks=CHECKS)
    dump(result,'verification.json')
    top=primary.sort_values(['model','rank']).groupby('model').head(5)
    sections=['# StrokeSure Step 38: focused SHAP explainability',
        '## Scope and exploratory model choice',
        'Three exact established configurations were refitted on all 4,088 development-training observations (199 events), seed42: uncorrected LR, balanced-class-weight LR and uncorrected RF. No resampling, calibrator or threshold change was applied. LR C=1, lbfgs, max_iter1000; RF200 trees, min_samples_leaf5, max_features sqrt, otherwise original builder parameters. Full parameters and fitted pipelines are saved. These are explanatory representatives, not selected winners. Steps33–37 show why weighting merits examination: increased fixed-cutoff sensitivity accompanies poorer Brier without established discrimination gains; RF supplies a nonlinear contrast. No new predictive performance estimate is produced.',
        '## Training-only data, samples and local cases',
        'Verified Step33 OOF IDs define training membership. The raw CSV loader skips all nontraining row positions before parsing; only the 4,088 permitted feature/label rows enter memory. Outcomes are checked against saved OOF labels. No held-out partition object, held-out labels or held-out feature records are loaded. Source-file bytes are hashed solely for integrity. One simple random sample of256 training observations is explained for every model. Primary and alternate backgrounds each contain32 natural-prevalence training observations excluded from the explanation sample; sampling is without replacement within each sample, not event-balanced. All IDs/outcomes are in selected_observations.csv. Models were fitted on these same patients: these explanations are in-sample model-behavior descriptions and do not establish out-of-sample explanatory stability.',
        f'The explanation sample contains {int(y.loc[explain_ids].sum())} events ({float(y.loc[explain_ids].mean()):.3%}); primary background {int(y.loc[bg].sum())}/32 and alternate {int(y.loc[alt].sum())}/32. Backgrounds overlap in {len(set(bg)&set(alt))} patients. Common local examples are real rows '+', '.join(str(row) for _,_,row in local)+'. They are nearest to RF sample predicted-probability quantiles .10/.50/.90, with ties resolved by sorted row ID, no label filtering. They illustrate low/middle/high relative score strata rather than clinically representative cases or error exemplars. A high relative decile need not exceed probability .5.',
        '## Explainers, output scales and categorical interpretation',
        'The SHAP library was installed in an isolated directory, but Windows application control blocked its Numba DLLs even in an escalated attempt. Both failed import logs and the initial implementation are preserved. No models had been fitted. The final analysis does not import or execute SHAP/Numba; it implements the standard Shapley formula directly in exact_shap.py. LR uses exact coefficient × (transformed value−background mean), explaining decision_function log-odds; sums across each original feature are exact for the original-feature linear game. RF uses exact enumeration of all 1,024 coalitions of ten original-feature players and empirical background averaging of positive-class probability, not TreeSHAP or a fitted surrogate. This is exact for the specified finite-background marginal game, not exact conditional-distribution or population SHAP. See [Lundberg & Lee](https://arxiv.org/abs/1705.07874) and the [SHAP exact-enumeration documentation](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Exact.html). Positive contributions increase the specified model output relative to its own background mean; they are not calibrated absolute patient risks.',
        'The original preprocessing is retained: median imputation, continuous standardization and drop-first one-hot for LR (16 columns); median imputation, unscaled continuous features and full one-hot for RF (21 columns). Hypertension/heart_disease pass through as binary. Unknown smoking status remains explicit. feature_mapping.csv maps every transformed column to exactly one of10 original features. Omitted LR reference levels have no extra transformed column. Their effects are represented relative to the learned encoding and background; no missing reference attribution is fabricated.',
        'Each categorical variable is one indivisible original-feature player: its full encoded block is swapped jointly from the explained patient or a real background donor. All absent original features come from the same donor row. Thus every categorical block is valid, and dependencies among jointly absent features are retained empirically. Mixing present and absent original variables can still disrupt cross-feature dependencies and generate implausible clinical combinations. This is marginal/interventional rather than observational conditional SHAP and is not causal. For LR only, signed transformed-feature values are also saved and summed; linear additivity makes this identical to the exhaustive original-feature game, verified independently on three rows for each background. RF directly computes ten original-feature contributions and has no invented per-dummy SHAP allocation. Global importance is mean(abs(original-feature contribution)); LR sum-absolute-dummy importance is separately supplied to expose within-group cancellation. Encoding and correlated age/marriage/work variables constrain interpretation.',
        '## Global findings',table(top[['model','feature','mean_abs_grouped_shap','normalized_importance','rank','scale']]),
        'Signed mean contributions and absolute importance answer different questions; both are saved. Raw SHAP magnitudes across LR log-odds and RF probability must not be compared directly. Comparisons below use within-model normalized importance shares and ranks, which still depend on the output scale and explanatory game.',
        '## Model differences and sensitivity',table(pd.DataFrame(cross)),table(pd.DataFrame(bg_sensitivity)),
        'conditional_importance_rank_stability.csv contains1,000 paired resamples of the same256 explanation rows (Monte Carlo seed38042), conditional mean-absolute-importance percentiles and top-three frequencies. These assess explanation-sample composition sensitivity for fixed fitted models/backgrounds, not retraining uncertainty or statistical hypothesis tests. Two backgrounds reuse fixed models and the same explanation rows (primary sampling42, alternate43); alternate43 is not a model-seed experiment. No additional model fits were performed.',
        '## Local explanations',table(pd.DataFrame(localrows).drop_duplicates(['model','row_index'])[['model','rf_quantile','row_index','y_true','positive_probability','base_value','model_output','scale']]),
        'All ten signed original-feature contributions and observed feature values are supplied for every local/model pair. Feature values are training observations, never synthetic cases. Local attribution signs reflect each model/background/encoding; they must not be translated into causal interventions or individual clinical advice.',
        '## RQ6 and H6',
        '**RQ6:** Which features contribute most strongly to stroke-risk predictions according to SHAP-based analysis? The global table identifies the leading original-feature contributions for these three fitted models and this fixed training sample. Rankings are conditional on preprocessing, model output, reference population and aggregation definition.',
        '**H6:** Important predictors identified by SHAP will be model-dependent rather than identical across every classifier. Model-specific importance shares and rankings provide descriptive evidence about this working hypothesis. Shared leading predictors can coexist with differing lower ranks and contribution magnitudes. Neither identical top predictors nor rank differences alone constitute a formal hypothesis test; different output scales, encodings and explanatory games also contribute. No causal, clinical-readiness or model-superiority conclusion follows.',
        '## Verification and limitations',table(pd.DataFrame(additivity)),
        f'**{result["n_passed"]}/{result["n_checks"]} checks PASS**, zero failed checks, zero unexpected warnings, three model fits. Checks cover upstream hashes, training membership/labels, reproducible sampling, categorical validity, feature alignment, tensor shapes, finite values, prediction reconstruction, signed grouping, analytical LR equivalence, and explainer replay. Steps33–37 and source files match their starting hashes. One public dataset, one model seed, in-sample explanation, modest backgrounds and correlated/encoded features remain material limitations. Bootstrap intervals do not include model refitting. No held-out data, final selection, SHAP interaction search, threshold optimization or clinical inference was used.',
        '## Figures']
    for f in FIGURES:sections += [f'![{f["caption"]}]({f["file"]})',f['caption']]
    sections += ['## Reproducibility and next steps','Run `.\\.venv\\Scripts\\python.exe results/step38_shap_explainability/analyze_step38.py` only in a fresh output directory: this script refuses to overwrite a completed manifest. The blocked SHAP installation is retained in this output directory; plotting reuses existing isolated Step34 dependencies without modifying them. .venv was not modified. analysis_plan.json, estimator_parameters.json, joblib pipelines, per-row transformed/grouped SHAP, additivity rows, selections, feature_mapping.csv and analysis_manifest.json allow audit/reuse. For future stronger explanation claims, prespecify held-out or outer-fold explanations and group-aware masking/conditional sensitivity; they require a separate approved design. Keep the final test locked. No such extra experiment is necessary for this focused descriptive task.']
    (OUT/'scientific_shap_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    dump(dict(models=KEYS,model_fits=3,training_n=4088,training_events=199,explanation_n=256,explanation_events=int(y.loc[explain_ids].sum()),background_n=32,heldout_data_accessed=False,loading='skiprows by verified Step33 training IDs before parsing',raw_data_sha256=sha(RAW_DATA_PATH),original_artifact_hashes=before,code_hashes={p.name:sha(p) for p in OUT.glob('*.py')},versions={k:getattr(v,'__version__','unknown') for k,v in [('numpy',np),('pandas',pd),('sklearn',sklearn),('scipy',scipy),('matplotlib',matplotlib)]},shap_implementation='exact_shap.py; no SHAP/Numba library execution',seconds=time.perf_counter()-start,verification={k:v for k,v in result.items() if k!='checks'}),'analysis_manifest.json')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ['_dependencies','_matplotlib_config','__pycache__']) and p.name not in ['artifact_inventory.csv','execution.log','delivery_verification.json']:
            inventory.append(dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(inventory),'artifact_inventory')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True);print(top[['model','feature','mean_abs_grouped_shap','rank']].to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        dump([dict(traceback=traceback.format_exc())],'error_records.json');raise
