import pandas as pd, numpy as np, sys
from scipy.optimize import minimize
df=pd.read_pickle('feats.pkl'); d=df[(df.is_test==0)].copy()
nc=d[(d.rain_mm_h==0)&(d.d_rc>1)&(d.cross_ns==0)]
pace=(nc.ride_time_min/nc.d_rc).groupby(nc.vehicle,observed=True).median()
d['km']=d.ride_time_min/d.vehicle.map(pace).astype(float)
d=d[(d.ev_active==0)&(d.rain_mm_h<0.5)&(d.day_idx<49)]
KN=np.array([-7,-5,-3,-1,1,3,5,7.])
rx,ry,cx,cy,km=[d[c].values for c in ('rx','ry','cx','cy','km')]
def route(p,K,rx,ry,cx,cy):
    yk=p[:len(KN)]; bx=p[len(KN):len(KN)+K]
    sr=ry>np.interp(rx,KN,yk); sc=cy>np.interp(cx,KN,yk)
    by=np.interp(bx,KN,yk)
    via=np.min(np.hypot(rx[:,None]-bx,ry[:,None]-by)+np.hypot(cx[:,None]-bx,cy[:,None]-by),axis=1)
    return np.where(sr!=sc,via,np.hypot(cx-rx,cy-ry))
def loss(p,K):
    r=km-route(p,K,rx,ry,cx,cy); return np.mean(np.minimum(np.abs(r),3.0))
best=None
for K,b0 in [(3,[-4.5,3.2,5.2]),(4,[-6.5,-4.5,3.2,5.2]),(5,[-6.5,-4.5,-1.5,3.2,5.2]),(6,[-6.5,-4.5,-1.5,1.8,3.2,5.2])]:
    p0=np.array([-0.2,-0.2,0.1,0.45,1.1,1.4,1.2,1.2]+b0)
    r=minimize(loss,p0,args=(K,),method='Powell',options=dict(maxiter=20000,xtol=1e-3,ftol=1e-6))
    r=minimize(loss,r.x,args=(K,),method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-3,fatol=1e-7))
    print(K, round(loss(p0,K),4), round(r.fun,4), np.round(r.x,3))
    np.save(f'river_{K}.npy',r.x)
print('no river loss', np.mean(np.minimum(np.abs(km-np.hypot(cx-rx,cy-ry)),3)))
