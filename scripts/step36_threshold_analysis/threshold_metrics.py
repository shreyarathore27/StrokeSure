"""Deterministic threshold/count calculations; contains no model fitting."""
import numpy as np

GRID=np.round(np.r_[0.,np.arange(1,101)/1000,np.arange(11,101)/100],3)
POLICIES=['fixed_0.50','sensitivity_ge_0.80','specificity_ge_0.90','maximum_grid_mcc']
METRICS=['sensitivity','specificity','precision','f1','mcc','balanced_accuracy',
         'fp','fn','fpr','fnr','predicted_positive_fraction','tp','tn']


def metrics_from_counts(tn,fp,fn,tp):
    tn,fp,fn,tp=[np.asarray(v,dtype=np.int64) for v in (tn,fp,fn,tp)]
    positive=tp+fn; negative=tn+fp; predicted=tp+fp; n=positive+negative
    assert np.all(positive>0) and np.all(negative>0)
    sensitivity=tp/positive; specificity=tn/negative
    precision=np.divide(tp,predicted,out=np.zeros_like(tp,dtype=float),where=predicted>0)
    f1=2*tp/(2*tp+fp+fn)
    denominator=np.sqrt(predicted.astype(float)*positive*negative*(tn+fn))
    mcc=np.divide(tp*tn-fp*fn,denominator,out=np.zeros_like(tp,dtype=float),where=denominator>0)
    return dict(n=n,n_positive=positive,n_negative=negative,tn=tn,fp=fp,fn=fn,tp=tp,
                sensitivity=sensitivity,specificity=specificity,precision=precision,f1=f1,mcc=mcc,
                balanced_accuracy=(sensitivity+specificity)/2,fpr=1-specificity,fnr=1-sensitivity,
                predicted_positive_fraction=predicted/n,precision_defined=predicted>0,
                mcc_denominator_nonzero=denominator>0)


def grid_metrics(y,p,thresholds=GRID):
    y,p=np.asarray(y,dtype=int),np.asarray(p,dtype=float)
    assert len(y)==len(p) and set(np.unique(y))=={0,1}
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    thresholds=np.asarray(thresholds,dtype=float)
    n1=int(y.sum());n0=len(y)-n1
    tp=n1-np.searchsorted(np.sort(p[y==1]),thresholds,side='left')
    fp=n0-np.searchsorted(np.sort(p[y==0]),thresholds,side='left')
    result=metrics_from_counts(n0-fp,fp,n1-tp,tp)
    result['threshold']=thresholds
    return result


def select_index(values,policy):
    """Policy sees only its supplied development metrics, never evaluation labels."""
    if policy=='fixed_0.50': return int(np.flatnonzero(values['threshold']==.5)[0])
    if policy=='sensitivity_ge_0.80':
        eligible=np.flatnonzero(values['sensitivity']>=.8)
        return int(eligible[-1]) if len(eligible) else None
    if policy=='specificity_ge_0.90':
        eligible=np.flatnonzero(values['specificity']>=.9)
        return int(eligible[0]) if len(eligible) else None
    if policy=='maximum_grid_mcc':
        best=np.max(values['mcc'])
        return int(np.flatnonzero(np.isclose(values['mcc'],best,rtol=0,atol=1e-12))[-1])
    raise ValueError(policy)
