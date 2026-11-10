"""Probability reliability metrics and descriptive diagnostic coefficient fits."""
import math
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

LOSS_EPS = 1e-15
LOGIT_EPS = 1e-6


def reliability_bins(y, p, n_bins=10, strategy='quantile'):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    assert len(y) == len(p) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    if strategy == 'uniform':
        edges = np.linspace(0, 1, n_bins+1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins+1)))
    n_actual = max(1, len(edges)-1)
    ids = np.searchsorted(edges[1:-1], p, side='right')
    records=[]
    for i in range(n_actual):
        mask=ids == i
        n=int(mask.sum())
        records.append(dict(bin=i, requested_bins=n_bins, actual_edge_bins=n_actual,
            binning=strategy, n=n, n_positive=int(y[mask].sum()),
            mean_probability=float(p[mask].mean()) if n else None,
            observed_fraction=float(y[mask].mean()) if n else None,
            lower_edge=float(edges[i]) if len(edges)>1 else float(p[0]),
            upper_edge=float(edges[i+1]) if len(edges)>1 else float(p[0])))
    assert sum(r['n'] for r in records) == len(y)
    return records


def calibration_coefficients(y, p):
    """Descriptive MLE on evaluated predictions, never applied as a calibrator."""
    y=np.asarray(y,dtype=float)
    z=logit(np.clip(p, LOGIT_EPS, 1-LOGIT_EPS))
    mu, sd=float(z.mean()), float(z.std())
    result=dict(calibration_in_large_intercept=None, joint_intercept=None,
                calibration_slope=None, joint_fit_status='not_estimable',
                logit_sd=sd, low_logit_spread=sd < 0.05,
                logit_clip_count=int(((np.asarray(p)<LOGIT_EPS)|(np.asarray(p)>1-LOGIT_EPS)).sum()))
    if not 0 < y.sum() < len(y):
        result['joint_fit_status']='one_class'
        return result
    result['calibration_in_large_intercept']=float(brentq(lambda a: float(expit(z+a).sum()-y.sum()), -80, 80))
    if sd < 1e-8:
        result['joint_fit_status']='constant_scores'
        return result
    if z[y==0].max() < z[y==1].min() or z[y==1].max() < z[y==0].min():
        result['joint_fit_status']='complete_separation'
        return result
    design=np.column_stack([np.ones(len(y)),(z-mu)/sd])
    def fun(beta):
        t=design@beta
        return float(np.mean(np.logaddexp(0,t)-y*t))
    def jac(beta):
        return design.T@(expit(design@beta)-y)/len(y)
    def hess(beta):
        q=expit(design@beta)
        return (design.T*(q*(1-q)))@design/len(y)
    fitted=minimize(fun,[logit(y.mean()),0],jac=jac,hess=hess,method='trust-exact',
                    options={'gtol':1e-9,'maxiter':200})
    gradient=float(np.max(np.abs(jac(fitted.x))))
    condition=float(np.linalg.cond(hess(fitted.x)))
    estimable=(np.isfinite(fitted.x).all() and (fitted.success or gradient<1e-7)
               and condition<1e12 and np.max(np.abs(fitted.x))<50)
    result.update(joint_gradient_max=gradient,joint_hessian_condition=condition,
                  joint_optimizer_success=bool(fitted.success))
    if estimable:
        result.update(joint_intercept=float(fitted.x[0]-fitted.x[1]*mu/sd),
                      calibration_slope=float(fitted.x[1]/sd),joint_fit_status='estimable')
    else:
        result['joint_fit_status']='nonconvergence_or_ill_conditioning'
    return result


def probability_metrics(y,p,coefficients=True):
    y,p=np.asarray(y,dtype=int),np.asarray(p,dtype=float)
    assert len(y)==len(p) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    clipped=np.clip(p,LOSS_EPS,1-LOSS_EPS)
    result=dict(n=len(y), n_positive=int(y.sum()), prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
        brier_score=float(brier_score_loss(y,p)),
        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        mean_probability=float(p.mean()), mean_probability_bias=float(p.mean()-y.mean()),
        observed_expected_ratio=float(y.mean()/p.mean()) if p.mean()>0 else None,
        probability_zero_count=int((p==0).sum()), probability_one_count=int((p==1).sum()),
        loss_clip_count=int(((p<LOSS_EPS)|(p>1-LOSS_EPS)).sum()),
        probability_q01=float(np.quantile(p,.01)),probability_q50=float(np.quantile(p,.5)),
        probability_q95=float(np.quantile(p,.95)),probability_q99=float(np.quantile(p,.99)))
    for n_bins,strategy in [(5,'quantile'),(10,'quantile'),(20,'quantile'),(10,'uniform')]:
        bins=reliability_bins(y,p,n_bins,strategy)
        gaps=[abs(r['mean_probability']-r['observed_fraction']) for r in bins if r['n']]
        weighted=sum(r['n']/len(y)*abs(r['mean_probability']-r['observed_fraction']) for r in bins if r['n'])
        result[f'ece_{strategy}_{n_bins}']=float(weighted)
        result[f'mce_{strategy}_{n_bins}']=float(max(gaps))
    if coefficients:
        result.update(calibration_coefficients(y,p))
    return result
