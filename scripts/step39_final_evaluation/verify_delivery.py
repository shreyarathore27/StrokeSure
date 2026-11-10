"""Independent post-test audit using saved predictions only, no model inference."""
from final_common import *
import datetime
from scipy.special import expit,logit
from scipy.stats import binomtest
sys.path.append(str(ROOT/'results/step34_imbalance_analysis/_plot_dependencies'))
from PIL import Image

checks=[]
def check(name,ok,**details):checks.append(dict(check=name,passed=bool(ok),**details))
freeze=validate_seal('selection_freeze.json');deployment=validate_seal('model_and_threshold_freeze.json')
started=json.loads((OUT/'test_access_started.json').read_text());completed=json.loads((OUT/'test_access_completed.json').read_text());pre=json.loads((OUT/'independent_preaccess_validation.json').read_text())
check('seals and frozen source identities',source_hashes()==freeze['source_hashes']==deployment['source_hashes'] and deployment['selection_freeze_sha256']==sha(OUT/'selection_freeze.json'))
times=[freeze['created_utc'],deployment['created_utc'],pre['checked_utc'],started['created_utc'],completed['completed_utc']]
check('freeze before any test access',all(datetime.datetime.fromisoformat(a)<datetime.datetime.fromisoformat(b) for a,b in zip(times,times[1:])),chronology=times)
check('exactly one locked test API access',completed['locked_api_get_calls']==1 and completed['base_probability_calls']==4 and completed['probability_streams']==5 and completed['operating_rows']==6)
for p,h in freeze['original_artifact_hashes'].items():check('prior unchanged '+p,sha(ROOT/p)==h)
for k,v in deployment['models'].items():check('fitted model unchanged '+k,sha(OUT/v['file'])==v['sha256'])
check('calibrator unchanged',sha(OUT/deployment['calibration']['file'])==deployment['calibration']['sha256'])
for v in ['pretest_verification.json','test_execution_verification.json']:check(v,json.loads((OUT/v).read_text())['overall_passed'])
check('no model/test warnings or errors',all(json.loads((OUT/p).read_text())==[] for p in ['training_warnings.json','test_warning_records.json','test_error_records.json']))
pred=pd.read_csv(OUT/'locked_test_predictions.csv',float_precision='round_trip');metrics=pd.read_csv(OUT/'locked_test_metrics.csv',float_precision='round_trip')
check('five exact frozen probability streams',set(pred.stream)=={c['id'] for c in freeze['configurations']} and len(pred)==5110)
grouped={s:g for s,g in pred.groupby('stream',sort=False)};first=next(iter(grouped.values()));ids=first.row_index.to_numpy();y=first.y_true.to_numpy()
check('test counts and separation',len(ids)==1022 and len(set(ids))==1022 and int(y.sum())==50 and not set(ids)&set(deployment['training_membership']))
for s,g in grouped.items():check('paired saved rows '+s,np.array_equal(g.row_index,ids) and np.array_equal(g.y_true,y) and np.isfinite(g.positive_probability).all() and g.positive_probability.between(0,1).all())
raw=grouped[BASES[2]+'::original'].positive_probability.to_numpy();sig=grouped[SIG].positive_probability.to_numpy();cal=deployment['calibration']
check('frozen sigmoid transformation',np.allclose(sig,expit(-(cal['a']*logit(raw)+cal['b'])),atol=1e-12,rtol=0))
check('strictly monotone ranking preserved',cal['a']<0 and roc_auc_score(y,raw)==roc_auc_score(y,sig) and average_precision_score(y,raw)==average_precision_score(y,sig))
for policy in deployment['policies']:
    row=metrics[metrics.id==policy['id']].iloc[0];p=grouped[policy['stream']].positive_probability.to_numpy();q=p>=policy['threshold']
    tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());fn=int(((y==1)&~q).sum());tp=int(((y==1)&q).sum())
    check('direct confusion counts '+policy['id'],[tn,fp,fn,tp]==[row.tn,row.fp,row.fn,row.tp] and row.threshold==policy['threshold'])
    check('independent ranking/probability metrics '+policy['id'],np.allclose([row.roc_auc,row.average_precision,row.brier_score,row.log_loss],[roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])],atol=1e-12,rtol=0))
    den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));mcc=(tp*tn-fp*fn)/den if den else 0
    check('independent decision metrics '+policy['id'],np.allclose([row.sensitivity,row.specificity,row.f1,row.mcc,row.balanced_accuracy],[tp/50,tn/972,2*tp/(2*tp+fp+fn),mcc,.5*(tp/50+tn/972)],atol=1e-12,rtol=0))
    check('undefined PPV retained '+policy['id'],(pd.isna(row.precision) and not row.precision_defined) if tp+fp==0 else abs(row.precision-tp/(tp+fp))<1e-12)
exact=pd.read_csv(OUT/'exact_binomial_intervals.csv',float_precision='round_trip')
for row in exact.itertuples():
    if row.n:
        c=binomtest(row.k,row.n).proportion_ci(method='exact');ok=np.allclose([row.exact95_low,row.exact95_high],[c.low,c.high],atol=1e-12,rtol=0)
    else:ok=pd.isna(row.exact95_low) and pd.isna(row.exact95_high)
    check('exact binomial replay '+row.id+row.metric,ok)
with np.load(OUT/'test_patient_bootstrap_counts.npz') as data:counts=data['counts'];check('bootstrap row identities',np.array_equal(data['row_index'],ids))
with np.load(OUT/'test_patient_bootstrap_metrics.npz') as data:values=data['values'];check('bootstrap metric/policy identities',list(data['metrics'])==BOOT_METRICS and list(data['policies'])==[p['id'] for p in deployment['policies']])
check('all5000 ordinary draws retained',counts.shape==(5000,1022) and np.all(counts.sum(axis=1)==1022) and np.all(counts@y>0) and np.all(counts@(1-y)>0))
replay=np.random.default_rng(39042).multinomial(1022,np.full(1022,1/1022),size=5000)
check('bootstrap seed exact replay',np.array_equal(counts,replay))
for draw in [0,1,4999]:
    idx=np.repeat(np.arange(1022),counts[draw])
    for j,policy in enumerate(deployment['policies']):
        p=grouped[policy['stream']].positive_probability.to_numpy();result=all_metrics(y[idx],p[idx],policy['threshold'],diagnostics=False)
        expected=np.array([np.nan if result[k] is None else result[k] for k in BOOT_METRICS])
        check('independent expanded-patient bootstrap '+str(draw)+policy['id'],np.allclose(values[draw,j],expected,atol=1e-12,rtol=0,equal_nan=True))
intervals=pd.read_csv(OUT/'test_bootstrap_intervals.csv',float_precision='round_trip')
for j,policy in enumerate(deployment['policies']):
    for k,metric in enumerate(BOOT_METRICS):
        row=intervals[(intervals.id==policy['id'])&(intervals.metric==metric)].iloc[0];v=values[:,j,k]
        if np.isfinite(v).all():ok=np.allclose([row.percentile95_low,row.percentile95_high],np.quantile(v,[.025,.975]),atol=1e-12,rtol=0)
        else:ok=pd.isna(row.percentile95_low) and pd.isna(row.percentile95_high) and row.n_defined==np.isfinite(v).sum()
        check('bootstrap interval replay '+policy['id']+metric,ok)
grid=pd.read_csv(OUT/'training_threshold_grid_by_repeat.csv',float_precision='round_trip');m=grid.groupby('threshold').mcc.mean();t=float(m.index[np.isclose(m,m.max(),rtol=0,atol=1e-12)][-1])
check('frozen training threshold and coverage',t==deployment['training_threshold']==.061 and len(grid)==5*191)
cv=pd.read_csv(OUT/'development_metrics_by_correlated_repeat.csv',float_precision='round_trip')
reference=pd.read_csv(ROOT/'results/step34_imbalance_analysis/comparison_pooled_repeat_mean_sd.csv',float_precision='round_trip')
for base in BASES:
    g=cv[cv.id==base+'::original::fixed_050'];r=reference[reference.pipeline==base].iloc[0]
    for metric in PRIMARY:check('development-reference reconciliation '+base+metric,abs(g[metric].mean()-r[metric+'_mean'])<1e-12)
check('development roles transparent',len(cv)==30 and set(cv[cv.id.str.endswith('training_mcc')].role)=={'apparent_after_threshold_selection'})
matrix=pd.read_csv(OUT/'research_question_hypothesis_evidence_matrix.csv')
check('all research questions and hypotheses',set(matrix['item'])=={f'{p}{i}' for p in ['RQ','H'] for i in range(1,9)})
check('official RQ1/H8 wording',matrix.set_index('item').loc['RQ1','official_definition']=='Can machine-learning models provide robust, well-calibrated, and interpretable stroke-risk predictions under severe class imbalance?' and matrix.set_index('item').loc['H8','official_definition']=='Explicit dataset-quality and preprocessing checks will identify characteristics that materially affect interpretation or reproducibility of results.')
figures=json.loads((OUT/'figure_captions.json').read_text());check('five publication figures',len(figures)==5)
for f in figures:
    with Image.open(OUT/f['file']) as im:
        im.load();a,b=im.convert('L').getextrema();check('PNG '+f['file'],im.format=='PNG' and min(im.size)>=1000 and a<b and all(abs(d-300)<1 for d in im.info.get('dpi',(0,0))),dimensions=list(im.size))
for row in pd.read_csv(OUT/'artifact_inventory.csv').itertuples():
    p=OUT/row.file;check('inventory '+row.file,p.exists() and p.stat().st_size==row.bytes and sha(p)==row.sha256)
result=dict(overall_passed=all(c['passed'] for c in checks),n_checks=len(checks),n_passed=sum(c['passed'] for c in checks),n_failed=sum(not c['passed'] for c in checks),checks=checks,scope='saved-prediction audit; no raw test loading or model predictions')
dump(result,'delivery_verification.json');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if not result['overall_passed']:print([c for c in checks if not c['passed']])
raise SystemExit(0 if result['overall_passed'] else 1)
