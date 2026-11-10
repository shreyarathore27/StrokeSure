"""Finite-background, original-feature interventional Shapley enumeration.

For M players, phi_j = sum_{S not containing j}
  [1 / (M * C(M-1, |S|))] * [v(S union j)-v(S)].
No fitted surrogate, permutation approximation, or binary extension required.
"""
import math
import numpy as np

def shapley_from_coalitions(values,m):
    v=np.asarray(values,dtype=float)
    assert v.ndim==2 and v.shape[1]==2**m and np.isfinite(v).all()
    masks=np.arange(2**m,dtype=int)
    out=np.empty((len(v),m))
    for j in range(m):
        absent=masks[(masks&(1<<j))==0]
        weights=np.array([1/(m*math.comb(m-1,int(s).bit_count())) for s in absent])
        out[:,j]=(v[:,absent|(1<<j)]-v[:,absent])@weights
    return out

def enumerate_coalitions(predict,z,background,column_groups,m,progress=None,batch_size=8):
    """Swap whole original-feature blocks; omitted blocks share one donor row."""
    z=np.asarray(z);background=np.asarray(background);groups=np.asarray(column_groups,dtype=int)
    assert z.shape[1]==background.shape[1]==len(groups) and set(groups)==set(range(m))
    result=np.empty((len(z),2**m));full=(1<<m)-1
    result[:,0]=float(np.mean(predict(background)));result[:,full]=predict(z)
    masks=np.arange(1,full)
    for begin in range(0,len(masks),batch_size):
        current=masks[begin:begin+batch_size]
        present=((current[:,None]>>groups[None,:])&1).astype(bool)
        hybrid=np.where(present[:,None,None,:],z[None,:,None,:],background[None,None,:,:])
        prediction=np.asarray(predict(hybrid.reshape(-1,z.shape[1])))
        result[:,current]=prediction.reshape(len(current),len(z),len(background)).mean(axis=2).T
        if progress is not None:progress(min(begin+batch_size,len(masks)),len(masks))
    return result
