from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
OUT=Path(__file__).resolve().parent/"gain_offset_isolation";OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(8);records=[];example=None
for rep in range(500):
 n,t=400,60;train=np.sort(rng.choice(t,40,replace=False));test=np.setdiff1d(np.arange(t),train)
 shape=np.exp(rng.uniform(np.log(.07),np.log(2),n))
 activity=rng.gamma(shape[:,None],1/shape[:,None],size=(n,t))
 for pf in ("all","train"):
  den=activity.mean(axis=1,keepdims=True) if pf=="all" else activity[:,train].mean(axis=1,keepdims=True)
  g=np.sqrt(activity/den)
  offsets=rng.integers(1,t,size=n);rolled=np.take_along_axis(g,(np.arange(t)[None,:]-offsets[:,None])%t,axis=1)
  for condition,x in (("Real",g),("Roll old gains",rolled)):
   y=x[:,test];mu=x[:,train].mean(axis=1,keepdims=True)
   score=1-np.sum((y-mu)**2)/np.sum((y-y.mean())**2)
   records.append(dict(rep=rep,pf=pf,condition=condition,r2=score))
  # Roll raw responses first, then re-estimate the training reference. This repeats
  # preprocessing under randomization rather than carrying over the old denominator.
  rolled_activity=np.take_along_axis(activity,(np.arange(t)[None,:]-offsets[:,None])%t,axis=1)
  d=rolled_activity.mean(axis=1,keepdims=True) if pf=="all" else rolled_activity[:,train].mean(axis=1,keepdims=True)
  x=np.sqrt(rolled_activity/d);y=x[:,test];mu=x[:,train].mean(axis=1,keepdims=True)
  records.append(dict(rep=rep,pf=pf,condition="Roll then refit reference",r2=1-np.sum((y-mu)**2)/np.sum((y-y.mean())**2)))
f=pd.DataFrame(records);f.to_csv(OUT/"independent_simulation.csv",index=False)
print(f.groupby(["pf","condition"]).r2.agg(["mean","std"]).to_string(),flush=True)
fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
conditions=["Real","Roll old gains","Roll then refit reference"]
for i,(pf,color) in enumerate((("all","#2566aa"),("train","#c25a18"))):
 means=[f[(f.pf==pf)&(f.condition==c)].r2.mean() for c in conditions]
 ax.bar(np.arange(3)+(i-.5)*.32,means,width=.32,color=color,label="PF="+pf)
ax.set_xticks(range(3),conditions);ax.axhline(0,c=".5",lw=.7)
ax.set_ylabel("Mean-only test R²");ax.set_title("Independent simulated neurons: no shared activity, no drift")
ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
fig.savefig(OUT/"independent_simulation.png",dpi=170)
