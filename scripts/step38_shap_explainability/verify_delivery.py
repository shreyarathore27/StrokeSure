"""Independent algebra, saved-pipeline, row, image and integrity verification."""
from pathlib import Path
import sys,json,hashlib,itertools,math
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'results/step34_imbalance_analysis/_plot_dependencies'))
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from exact_shap import shapley_from_coalitions

checks=[]
def check(name,ok,**info):checks.append(dict(check=name,passed=bool(ok),**info))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((OUT/'analysis_manifest.json').read_text())
for p,h in manifest['original_artifact_hashes'].items():check('prior unchanged '+p,sha(ROOT/p)==h)
for p,h in manifest['code_hashes'].items():check('source '+p,sha(OUT/p)==h)
for row in pd.read_csv(OUT/'artifact_inventory.csv').itertuples():
    p=OUT/row.file;check('artifact '+row.file,p.is_file() and p.stat().st_size==row.bytes and sha(p)==row.sha256)
check('exact scope',manifest['model_fits']==3 and manifest['explanation_n']==256 and manifest['background_n']==32 and not manifest['heldout_data_accessed'])
run='imbalance_cv_step33_20261109T165943Z'
upstream=json.loads((ROOT/'results/imbalance_cv'/run/(run+'_manifest.json')).read_text())
check('raw source hash matches verified Step33',manifest['raw_data_sha256']==upstream['data']['data_sha256'])
# Known asymmetric game: additive effects plus a two-way and three-way interaction.
v=np.array([[.2+sum(c for j,c in enumerate([1,2,3]) if s&(1<<j))+(6 if s&3==3 else 0)+(9 if s==7 else 0) for s in range(8)]])
check('known interaction allocation',np.allclose(shapley_from_coalitions(v,3),[[7,8,6]],atol=1e-12,rtol=0))
check('constant game',np.allclose(shapley_from_coalitions(np.ones((2,32)),5),0,atol=1e-15,rtol=0))
# Independent permutation definition over every permutation of a random five-player game.
v=np.random.default_rng(123).normal(size=(3,32));reference=np.zeros((3,5))
for order in itertools.permutations(range(5)):
    mask=0
    for j in order:
        new=mask|(1<<j);reference[:,j]+=v[:,new]-v[:,mask];mask=new
reference/=math.factorial(5)
check('exhaustive independent permutation identity',np.allclose(shapley_from_coalitions(v,5),reference,atol=1e-12,rtol=0))

selection=pd.read_csv(OUT/'selected_observations.csv');training=pd.read_csv(OUT/'training_membership.csv')
explain=selection[selection.role=='explanation'];ids=explain.row_index.to_numpy();allowed=set(training.row_index)
check('all selected IDs training-only',set(selection.row_index)<=allowed and len(allowed)==4088 and training.y_true.sum()==199)
replay=np.sort(np.random.default_rng(42).choice(training.row_index.to_numpy(),256,replace=False))
check('independent explanation sampling replay',np.array_equal(replay,ids))
rng=np.random.default_rng(42);rng.choice(training.row_index.to_numpy(),256,replace=False)
rest=np.array(sorted(allowed-set(ids)))
check('independent primary background replay',np.array_equal(np.sort(rng.choice(rest,32,replace=False)),selection[selection.role=='background_primary'].row_index))
check('independent alternate background replay',np.array_equal(np.sort(np.random.default_rng(43).choice(rest,32,replace=False)),selection[selection.role=='background_alternate'].row_index))
mapping=pd.read_csv(OUT/'feature_mapping.csv');importance=pd.read_csv(OUT/'global_grouped_importance.csv')
for model in manifest['models']:
    fitted=joblib.load(OUT/(model+'__training_fit.joblib'));names=fitted.named_steps['preprocessor'].get_feature_names_out()
    params=fitted.named_steps['classifier'].get_params()
    check('documented seed and weights '+model,params['random_state']==42 and params['class_weight']==('balanced' if model.endswith('class_weight_balanced') else None))
    if model.startswith('logistic'):
        check('documented LR settings '+model,params['C']==1 and params['solver']=='lbfgs' and params['max_iter']==1000 and fitted.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].n_samples_seen_==4088)
    else:check('documented RF settings',params['n_estimators']==200 and params['min_samples_leaf']==5 and params['max_features']=='sqrt' and params['n_jobs']==-1)
    mp=mapping[mapping.model==model]
    check('loaded fitted feature names '+model,list(names)==mp.transformed_name.tolist() and len(names)==(21 if model.startswith('random') else 16))
    for variant in ['primary','alternate']:
        grouped=pd.read_csv(OUT/(model+'__'+variant+'__grouped_shap.csv'))
        rows=pd.read_csv(OUT/(model+'__'+variant+'__additivity_rows.csv'))
        values=grouped.drop(columns='row_index').to_numpy();features=grouped.columns[1:]
        check('saved SHAP shape, order, finite '+model+variant,values.shape==(256,10) and np.isfinite(values).all() and np.array_equal(grouped.row_index,ids))
        check('saved grouped additivity '+model+variant,np.allclose(rows.base_value+values.sum(axis=1),rows.model_output,atol=1e-10,rtol=0))
        check('finite valid probabilities '+model+variant,np.isfinite(rows.positive_probability).all() and rows.positive_probability.between(0,1).all())
        reference=importance[(importance.model==model)&(importance.background==variant)].set_index('feature').loc[features]
        expected=np.abs(values).mean(axis=0)
        check('independent importance calculation '+model+variant,np.allclose(reference.mean_abs_grouped_shap,expected,atol=1e-12,rtol=0))
        check('normalized importance '+model+variant,np.allclose(reference.normalized_importance,expected/expected.sum(),atol=1e-12,rtol=0))
        if model.startswith('logistic'):
            transformed=pd.read_csv(OUT/(model+'__'+variant+'__transformed_shap.csv')).drop(columns='row_index')
            for f in features:
                cols=mp[mp.clinical_group==f].transformed_name
                check('independent grouping '+model+variant+f,np.allclose(transformed[cols].sum(axis=1),grouped[f],atol=1e-12,rtol=0))
        else:
            with np.load(OUT/(model+'__'+variant+'__coalition_values.npz')) as data:
                v=data['values'];check('all RF coalition values '+variant,v.shape==(256,1024) and np.isfinite(v).all() and np.all((v>=0)&(v<=1)))
                # Independent coefficient accumulation on saved arbitrary game, all patients.
                independent=np.zeros_like(values)
                for j in range(10):
                    for s in range(1024):
                        if s&(1<<j):continue
                        size=int(s).bit_count();weight=math.factorial(size)*math.factorial(9-size)/math.factorial(10)
                        independent[:,j]+=weight*(v[:,s|(1<<j)]-v[:,s])
                check('RF all-patient independent Shapley formula '+variant,np.allclose(independent,values,atol=1e-12,rtol=0))
local=pd.read_csv(OUT/'local_explanations.csv');observed=pd.read_csv(OUT/'local_observed_features.csv').set_index('row_index')
rfpred=pd.read_csv(OUT/'explained_training_predictions.csv',float_precision='round_trip');rfpred=rfpred[rfpred.model=='random_forest__none']
for q in [.1,.5,.9]:
    target=np.quantile(rfpred.positive_probability,q);index=int(np.argmin(abs(rfpred.positive_probability.to_numpy()-target)))
    row=int(rfpred.iloc[index].row_index);check('local quantile selection '+str(q),set(local[local.rf_quantile==q].row_index)=={row})
for model in manifest['models']:
    fitted=joblib.load(OUT/(model+'__training_fit.joblib'));p=fitted.predict_proba(observed)[:,1]
    saved=local[local.model==model].drop_duplicates('row_index').set_index('row_index').loc[observed.index]
    check('loaded pipeline reproduces local probabilities '+model,np.allclose(p,saved.positive_probability,atol=1e-12,rtol=0))
for figure in json.loads((OUT/'figure_captions.json').read_text()):
    with Image.open(OUT/figure['file']) as im:
        im.load();lo,hi=im.convert('L').getextrema();dpi=im.info.get('dpi',(0,0))
        check('PNG '+figure['file'],im.format=='PNG' and min(im.size)>=1000 and lo<hi and all(abs(x-300)<1 for x in dpi),dimensions=list(im.size))
check('eight publication PNGs',len(json.loads((OUT/'figure_captions.json').read_text()))==8)
check('zero warnings/errors',json.loads((OUT/'warning_records.json').read_text())==[] and json.loads((OUT/'error_records.json').read_text())==[])
result=dict(overall_passed=all(c['passed'] for c in checks),n_checks=len(checks),n_passed=sum(c['passed'] for c in checks),n_failed=sum(not c['passed'] for c in checks),checks=checks)
(OUT/'delivery_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if not result['overall_passed']:print([c for c in checks if not c['passed']])
raise SystemExit(0 if result['overall_passed'] else 1)
