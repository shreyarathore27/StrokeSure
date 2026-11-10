"""Dependence-aware exploratory summaries; no estimator fitting."""
import numpy as np
from scipy.stats import t

METRICS=['average_precision','roc_auc','brier_score','sensitivity','specificity','mcc']

def corrected_cv(difference,ratio,family_size=54):
    d=np.asarray(difference,dtype=float)
    n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1));factor=1/n+ratio
    se=float(np.sqrt(factor)*sd)
    result=dict(n_paired_folds=n,mean_difference=mean,sd_paired_fold_difference=sd,
        validation_training_ratio=ratio,correction_factor=factor,corrected_se=se,
        naive_se_reference=sd/np.sqrt(n),df=n-1,
        mean_difference_percentage_points=100*mean,
        min_fold_difference=float(d.min()),max_fold_difference=float(d.max()),
        positive_fold_count=int((d>1e-12).sum()),negative_fold_count=int((d < -1e-12).sum()),
        near_zero_fold_count=int((np.abs(d)<=1e-12).sum()))
    if np.max(np.abs(d))<=1e-12 or sd<=1e-15:
        result.update(test_status='numerically_equal_or_zero_variance_no_informative_test',
                      t_statistic=None,p_corrected=None,ci95_low=None,ci95_high=None,
                      bonferroni_family_ci_low=None,bonferroni_family_ci_high=None)
    else:
        crit=float(t.ppf(.975,n-1));familycrit=float(t.ppf(1-.05/(2*family_size),n-1))
        statistic=mean/se
        result.update(test_status='approximate_corrected_repeated_cv',t_statistic=statistic,
            p_corrected=float(2*t.sf(abs(statistic),n-1)),
            ci95_low=mean-crit*se,ci95_high=mean+crit*se,
            bonferroni_family_ci_low=mean-familycrit*se,
            bonferroni_family_ci_high=mean+familycrit*se)
    return result

def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p,kind='stable');sorted_p=p[order]
    adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*sorted_p))
    output=np.empty(len(p));output[order]=adjusted
    return output

class WeightedPredictionScorer:
    """Bootstrap metric vectorization preserving exact probability ties."""
    def __init__(self,y,p):
        self.y=np.asarray(y,dtype=int);self.p=np.asarray(p,dtype=float)
        self.order=np.argsort(self.p,kind='stable')
        ordered=self.p[self.order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(ordered)!=0)+1]
        self.ordered_y=self.y[self.order]
        self.sq=(self.p-self.y)**2
        self.positive=self.p>=.5

    def score(self,weights):
        w=np.asarray(weights,dtype=float)
        positive=w@self.y;negative=w.sum(axis=1)-positive
        assert np.all(positive>0) and np.all(negative>0)
        order_w=w[:,self.order]
        gp=np.add.reduceat(order_w*self.ordered_y,self.starts,axis=1)
        gn=np.add.reduceat(order_w*(1-self.ordered_y),self.starts,axis=1)
        cumulative_negative=np.cumsum(gn,axis=1)
        auc=np.sum(gp*(cumulative_negative-.5*gn),axis=1)/(positive*negative)
        gpdesc=gp[:,::-1];gndesc=gn[:,::-1]
        cp=np.cumsum(gpdesc,axis=1);ct=np.cumsum(gpdesc+gndesc,axis=1)
        precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0)
        ap=np.sum((gpdesc/positive[:,None])*precision,axis=1)
        tp=np.sum(w*(self.positive&(self.y==1)),axis=1)
        fp=np.sum(w*(self.positive&(self.y==0)),axis=1)
        fn=positive-tp;tn=negative-fp
        den=np.sqrt((tp+fp)*positive*negative*(tn+fn))
        mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(tp),where=den>0)
        result=np.column_stack([ap,auc,np.sum(w*self.sq,axis=1)/w.sum(axis=1),tp/positive,tn/negative,mcc])
        return result,int((den==0).sum())
