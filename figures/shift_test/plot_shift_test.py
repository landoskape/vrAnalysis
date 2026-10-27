"""Export compact stored shift-test scores from Myriad and plot avg_env results."""
from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REMOTE = r'''
import sqlite3, pickle, json, sys, subprocess, os
from pathlib import Path
import numpy as np
root=Path('~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2').expanduser()
query = "import sqlite3,json; from pathlib import Path; root=Path('~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2').expanduser(); c=sqlite3.connect('file:'+str(root/'results.db')+'?immutable=1',uri=True); print(json.dumps(c.execute(\"SELECT result_uid,session_id,analysis_summary FROM results WHERE analysis_type='gain_regression' AND schema_version='v4' AND result_stored=1 ORDER BY session_id,analysis_summary\").fetchall()))"
query_env=dict(os.environ)
query_env.pop('LD_LIBRARY_PATH',None)
rows=json.loads(subprocess.check_output(['/shared/ucl/apps/miniconda/24.3.0-0/bin/python3','-c',query],env=query_env,text=True))
for i,(uid,sid,summary) in enumerate(rows):
    with (root/'blobs'/(uid+'.pkl')).open('rb') as f: d=pickle.load(f)
    record=dict(uid=uid,session=sid,summary=summary)
    for key in ['r2_test','r2_test_null','mse_test','mse_test_null','env_slot_ids','n_trials_env']:
        record[key]=np.asarray(d[key],dtype=float).tolist()
    print(json.dumps(record),flush=True)
    if (i+1)%500==0: print('Exported',i+1,'/',len(rows),file=sys.stderr,flush=True)
print('Export complete:',len(rows),file=sys.stderr,flush=True)
'''
raw=OUT/'shift_scores.jsonl'
if not raw.exists():
    partial=OUT/'shift_scores.partial.jsonl'
    with partial.open('w',encoding='utf-8') as f:
        p=subprocess.run(['ssh','myriad','module load python/3.11.4 && ~/Scratch/envs/vrAnalysis/bin/python -'],input=REMOTE,text=True,stdout=f)
    if p.returncode: raise SystemExit(p.returncode)
    partial.replace(raw)

records=[]
for line in raw.read_text().splitlines():
    d=json.loads(line)
    import re
    params={k:re.search(pattern,d['summary']).group(1) for k,pattern in {
        'placefield_split':r'_pf=(.*?)_estimator=',
        'gain_estimator':r'_estimator=(.*?)_gain=',
        'gain_transform':r'_gain=(.*?)_seed=',
        'xshift':r'_shift=([+-]?\d+)_v4'}.items()}
    params['xshift']=int(params['xshift'])
    for slot,(env,real,null) in enumerate(zip(d['env_slot_ids'],d['r2_test'],d['r2_test_null'])):
        if not np.isfinite(env) or not np.isfinite(real) or not np.isfinite(null): continue
        records.append(dict(**params,session=d['session'],mouse=d['session'].split('.')[0],environment=int(env),r2=real,null_r2=null,n_trials=d['n_trials_env'][slot]))
df=pd.DataFrame(records)
keys=['placefield_split','gain_estimator','gain_transform']
# Match the same session/environment observations at all seven lags.
complete=df.groupby(keys+['session','environment']).xshift.nunique()
keep=complete[complete==7].reset_index()[keys+['session','environment']]
df=df.merge(keep,on=keys+['session','environment'],validate='many_to_one')
df.to_csv(OUT/'environment_scores.csv',index=False)
session=df.groupby(keys+['mouse','session','xshift'])[['r2','null_r2']].mean().reset_index()
mouse=session.groupby(keys+['mouse','xshift'])[['r2','null_r2']].mean().reset_index()
mouse['delta_r2']=mouse.r2-mouse.null_r2
mouse.to_csv(OUT/'mouse_avg_env_scores.csv',index=False)
summary=mouse.groupby(keys+['xshift']).agg(mean_r2=('r2','mean'),sem_r2=('r2','sem'),null_r2=('null_r2','mean'),null_sem=('null_r2','sem'),delta_r2=('delta_r2','mean'),delta_sem=('delta_r2','sem'),n_mice=('mouse','nunique')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
for split in ['train','all']:
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout='constrained')
    for row,est in enumerate(['gaussian','least_squares']):
        for col,transform in enumerate(['raw','sqrt']):
            ax=axes[row,col]
            sel=(mouse.placefield_split==split)&(mouse.gain_estimator==est)&(mouse.gain_transform==transform)
            sub=mouse.loc[sel]
            for _,v in sub.groupby('mouse'):
                ax.plot(v.xshift,v.r2,color='#287ca5',alpha=.16,lw=.8)
            s=summary[(summary.placefield_split==split)&(summary.gain_estimator==est)&(summary.gain_transform==transform)].sort_values('xshift')
            x=s.xshift.to_numpy()
            for y,se,color,label,style in [('mean_r2','sem_r2','#156d9e','Real','-'),('null_r2','null_sem','#d47b32','Trial-roll null','--')]:
                ax.plot(x,s[y],style,color=color,marker='o',ms=4,lw=2,label=label)
                ax.fill_between(x,s[y]-s[se],s[y]+s[se],color=color,alpha=.14,lw=0)
            ax.axvline(0,color='.65',ls=':',lw=1,zorder=0)
            ax.axhline(0,color='.75',lw=.7,zorder=0)
            n_sessions=session[(session.placefield_split==split)&(session.gain_estimator==est)&(session.gain_transform==transform)].session.nunique()
            n_mice=sub.mouse.nunique()
            ax.set_title(f'{est.replace("_"," ").title()} | {transform} gain\n{n_mice} mice, {n_sessions} sessions',fontsize=12)
            ax.set_xticks(range(-3,4)); ax.grid(axis='y',alpha=.12)
            if row==1: ax.set_xlabel('xshift (target trial − source trial)')
            if col==0: ax.set_ylabel('Performance: held-out test R²')
    axes[0,0].legend(frameon=False,fontsize=10)
    title='Training trials only' if split=='train' else 'All trials'
    fig.suptitle(f'Gain shift test · avg_env · place-field template: {title}',fontsize=16)
    fig.supxlabel('Environment mean per session → session mean per mouse → equal-weight mouse mean ± SEM\nFaint lines: individual mice. Same session/environment observations at all seven shifts.',fontsize=10)
    fig.savefig(OUT/f'shift_test_{split}.png'); fig.savefig(OUT/f'shift_test_{split}.pdf'); plt.close(fig)
print('RECORDS',len(raw.read_text().splitlines()),'SESSIONS',df.session.nunique(),'MICE',df.mouse.nunique(),'ENV OBSERVATIONS',len(df))
print(summary[summary.xshift==0].to_string(index=False))
print('FILES',OUT)
