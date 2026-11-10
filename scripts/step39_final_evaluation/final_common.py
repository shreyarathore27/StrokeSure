"""Frozen Step39 decision/evaluation utilities. No data are loaded on import."""
from pathlib import Path
import sys,json,hashlib,datetime,os
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'results/step35_calibration'));sys.path.insert(0,str(ROOT/'results/step36_threshold_analysis'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix,matthews_corrcoef
from calibration_metrics import probability_metrics,reliability_bins
from threshold_metrics import GRID,grid_metrics
RUN='imbalance_cv_step33_20261109T165943Z';S33=ROOT/'results/imbalance_cv'/RUN;S35=ROOT/'results/step35_calibration'
BASES=['dummy_prior__none','logistic_regression__none','logistic_regression__class_weight_balanced','random_forest__none']
SIG='logistic_regression__class_weight_balanced::sigmoid'
CONFIGS=[dict(id=k+'::original',base=k,calibration='none') for k in BASES]+[dict(id=SIG,base=BASES[2],calibration='sigmoid')]
PRIMARY=['average_precision','roc_auc','brier_score']
BOOT_METRICS=PRIMARY+['sensitivity','specificity','precision','f1','mcc','balanced_accuracy','log_loss']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(x,name):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def seal(x,name):
    p=OUT/name
    with p.open('x',encoding='utf-8') as h:json.dump(x,h,indent=2,allow_nan=False)
    digest=sha(p)
    with (OUT/(name+'.sha256')).open('x',encoding='ascii') as h:h.write(digest+'\n')
    os.chmod(p,0o444)
    return digest
def validate_seal(name):
    p=OUT/name;assert sha(p)==(OUT/(name+'.sha256')).read_text().strip()
    return json.loads(p.read_text())
def write(df,name):df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
def prior_snapshot():
    dirs=[S33]+[ROOT/'results'/n for n in ['step34_imbalance_analysis','step35_calibration','step36_threshold_analysis','step37_statistical_robustness','step38_shap_explainability']]+[ROOT/'src']
    return {str(p.relative_to(ROOT)):sha(p) for d in dirs for p in sorted(d.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['_plot_dependencies','_dependencies','_matplotlib_config','__pycache__'])}
def source_hashes():
    paths=list((ROOT/'src').glob('*.py'))+[OUT/n for n in ['final_common.py','prepare_final.py','evaluate_locked_test.py']]+[S35/'calibration_metrics.py',ROOT/'results/step36_threshold_analysis/threshold_metrics.py',ROOT/'results/step37_statistical_robustness/statistical_methods.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def all_metrics(y,p,threshold,diagnostics=True):
    y=np.asarray(y,int);p=np.asarray(p,float);assert set(np.unique(y))=={0,1} and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    predicted=p>=threshold;tn,fp,fn,tp=map(int,confusion_matrix(y,predicted,labels=[0,1]).ravel())
    result=probability_metrics(y,p,coefficients=diagnostics)
    result.update(threshold=float(threshold),tn=tn,fp=fp,fn=fn,tp=tp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        precision=tp/(tp+fp) if tp+fp else None,precision_defined=bool(tp+fp),f1=2*tp/(2*tp+fp+fn),mcc=float(matthews_corrcoef(y,predicted)),
        mcc_denominator_nonzero=bool((tp+fp)*(tn+fn)),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)))
    return result
