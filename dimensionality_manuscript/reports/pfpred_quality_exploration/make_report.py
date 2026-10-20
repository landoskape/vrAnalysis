"""Scratch report from stored v9 results; run with the vrAnalysis interpreter."""
from pathlib import Path
import sqlite3, pickle, json, itertools, html, warnings, tomllib, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
with (ROOT/'paths.toml').open('rb') as f:
    STORE = Path(tomllib.load(f)['paths']['storage'])/'dimensionality-manuscript/cache/pipeline_v2'
con = sqlite3.connect(f'file:{(STORE/"results.db").as_posix()}?mode=ro&immutable=1', uri=True)
rows = con.execute("SELECT result_uid,session_id,analysis_key,analysis_summary,computed_at FROM results WHERE analysis_type='pfpred_quality' AND schema_version='v9' AND result_stored=1 ORDER BY session_id").fetchall()
assert len({r[2] for r in rows}) == 1, 'Select a single configuration before comparing sessions'
base = [('reliability_slot','Spatial reliability'),('fraction_active_slot','Fraction active'),('pf_peak','PF peak'),('r2_slot','Prediction R2'),('rms_slot','RMS error'),('norm_rms_slot','Normalized RMS'),('frac_var_pred_slot','PF variance fraction'),('var_total_slot','Activity variance'),('var_pred_slot','Prediction variance'),('var_residual_slot','Residual variance')]
amp = [(f'trial_{m}_{s}',f'{label}: {stat}') for m,label in [('rms','RMS'),('mean','Mean activity'),('pf_dot','PF projection'),('gain','Gaussian gain')] for s,stat in [('mean','mean'),('variance','variance'),('cv','CV')]]
metrics = base+amp
keys, labels = zip(*metrics)
colors=['black','blue','green']; slots=['1st','2nd','3rd']
records=[]; provenance=[]
for uid,sid,akey,summary,computed in rows:
    d=pickle.loads((STORE/'blobs'/f'{uid}.pkl').read_bytes())
    assert all(k in d for k in keys)
    mouse,date,session=sid.rsplit('.',2)
    provenance.append(dict(uid=uid,session=sid,mouse=mouse,analysis_key=akey,configuration=summary,computed_at=computed))
    for slot,env in enumerate(d['env_slot_ids']):
        if not np.isfinite(env) or not np.any(np.isfinite(d['reliability_slot'][slot])): continue
        x=np.column_stack([np.asarray(d[k][slot],float) for k in keys])
        q=np.asarray(d['quality_filtered_roi_mask_slot'][slot],bool)
        assert x.shape[0] == len(q)
        expected=(x[:,0]>.3)&(x[:,1]>.1)
        boundary=np.isclose(x[:,0],.3,rtol=0,atol=1e-7)|np.isclose(x[:,1],.1,rtol=0,atol=1e-7)
        assert np.all((q==expected)|boundary), sid
        records.append(dict(sid=sid,mouse=mouse,date=date,session=int(session),slot=slot,env=int(env),x=x,q=q))
mice=sorted({r['mouse'] for r in records})
for mouse in mice:
    ss=sorted({r['sid'] for r in records if r['mouse']==mouse})
    for slot in range(3):
        rr=sorted([r for r in records if r['mouse']==mouse and r['slot']==slot],key=lambda r:(r['date'],r['session']))
        for i,r in enumerate(rr): r['within']=i+1; r['overall']=ss.index(r['sid'])+1
pd.DataFrame(provenance).to_csv(OUT/'provenance.csv',index=False)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
sections=[]
def save(fig,name,title,caption):
    if os.environ.get('REPORT_REUSE_FIGURES') and (FIG/f'{name}.png').exists():
        plt.close(fig); sections.append((name,title,caption)); return
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white')
    plt.close(fig); sections.append((name,title,caption)); print(name,flush=True)

def selected(r,subset):
    return np.ones(len(r['q']),bool) if subset=='all' else r['q'] if subset=='quality' else (r['x'][:,1]>.1) if subset=='active' else ~r['q']

# One row per session/environment/subset/metric; metric-specific finite denominators.
summary_rows=[]
for r in records:
    for subset in ['all','quality','complement']:
        mask=selected(r,subset)
        for j,(key,label) in enumerate(metrics):
            v=r['x'][mask,j]; v=v[np.isfinite(v)]
            summary_rows.append(dict(session=r['sid'],mouse=r['mouse'],slot=r['slot'],env=r['env'],within=r['within'],overall=r['overall'],subset=subset,metric=key,n=len(v),median=np.median(v) if len(v) else np.nan,mean=np.mean(v) if len(v) else np.nan))
summary_df=pd.DataFrame(summary_rows); summary_df.to_csv(OUT/'session_summaries.csv',index=False)

# Primary familiarity pages, with data first reduced within each session and mouse.
for subset in ['all','quality','complement']:
    for stat in ['median','mean']:
        fig,axs=plt.subplots(4,3,figsize=(13,12),layout='constrained')
        for ax,(key,label) in zip(axs.flat,amp):
            for slot,col in enumerate(colors):
                z=summary_df.query('subset==@subset and metric==@key and slot==@slot')
                pivot=z.pivot(index='mouse',columns='within',values=stat).reindex(mice)
                ns=pivot.count(); avg=pivot.mean(); sem=pivot.std(ddof=1)/np.sqrt(ns)
                # A one-mouse tail has no between-mouse uncertainty and is omitted here.
                keep=ns>=3; xx=ns.index.to_numpy()[keep]; yy=avg.to_numpy()[keep]; ee=sem.to_numpy()[keep]
                ax.plot(xx,yy,color=col,label=slots[slot]); ax.fill_between(xx,yy-ee,yy+ee,color=col,alpha=.16)
            ax.set(title=label,xlabel='Session within environment',ylabel=f'ROI {stat}')
            if key=='trial_gain_mean' and stat=='median': ax.set_ylim(.95,1.05)
        axs.flat[0].legend(frameon=False)
        fig.suptitle(f'Trial-amplitude experience curves | {subset} cells | ROI {stat} -> mouse mean +/- SEM')
        save(fig,f'experience_{subset}_{stat}',f'Amplitude experience: {subset}, {stat}','Each session contributes its finite-only ROI '+stat+'. Curves are the equal-mouse mean +/- SEM at each within-environment session. Only points with at least 3 mice are shown. Cell membership can change across sessions; these are not tracked-cell trajectories.')

for mode in ['within','overall']:
    fig,axs=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for ax,(key,label) in zip(axs.flat,base[:7]+base[8:]):
        for slot,col in enumerate(colors):
            z=summary_df.query('subset=="all" and metric==@key and slot==@slot')
            p=z.pivot(index='mouse',columns=mode,values='median').reindex(mice)
            n=p.count(); keep=n>=3; x=n.index.to_numpy()[keep]; a=p.mean().to_numpy()[keep]; e=(p.std()/np.sqrt(n)).to_numpy()[keep]
            ax.plot(x,a,color=col,label=slots[slot]); ax.fill_between(x,a-e,a+e,color=col,alpha=.16)
        ax.set(title=label,xlabel=f'{mode.capitalize()} session #',ylabel='ROI median')
    axs.flat[0].legend(frameon=False);fig.suptitle(f'Prediction and spatial properties | all cells | {mode} alignment')
    save(fig,f'prediction_{mode}',f'Prediction properties: {mode} alignment','Equal-mouse mean +/- SEM of session ROI medians; minimum 3 mice. Overall alignment retains gaps where an environment was absent. Different metrics use their own finite ROI subsets.')

example=next(r for r in records if r['sid']=='ATL027.2023-07-27.701' and r['env']==3)
print('EXAMPLE',example['sid'],example['slot'],len(example['q']),flush=True)

def scatter(ax,r,i,j,subset='all'):
    x,y=r['x'][:,i],r['x'][:,j]; mask=selected(r,subset)&np.isfinite(x)&np.isfinite(y); x=x[mask];y=y[mask]
    if len(x)<3: ax.set_title('Insufficient finite cells');return
    rho=spearmanr(x,y).statistic
    # Percentile view avoids a handful of unstable gain/CV values flattening every panel.
    xl=np.quantile(x,[.005,.995]); yl=np.quantile(y,[.005,.995]); vis=(x>=xl[0])&(x<=xl[1])&(y>=yl[0])&(y<=yl[1])
    ax.scatter(x[vis],y[vis],s=2,alpha=.22,color='black',rasterized=True)
    edges=np.unique(np.quantile(x,np.linspace(0,1,13))); bx=[];by=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        k=(x>=lo)&(x<=hi)
        if k.sum()>=10:bx.append(np.median(x[k]));by.append(np.median(y[k]))
    ax.plot(bx,by,color='darkorange',lw=1.4)
    if xl[1]>xl[0]: ax.set_xlim(xl)
    if yl[1]>yl[0]: ax.set_ylim(yl)
    ax.set(xlabel=labels[i],ylabel=labels[j],title=f'rho={rho:.2f} | n={len(x):,} | outside={len(x)-vis.sum()}')

fig,axs=plt.subplots(4,3,figsize=(13,12),layout='constrained')
for ax,j in zip(axs.flat,range(10,22)): scatter(ax,example,0,j)
fig.suptitle(f'Spatial reliability vs all 12 amplitude summaries | {example["sid"]} | env {example["env"]}')
save(fig,'spatial_amplitude_examples','All twelve spatial/amplitude comparisons','All cells with pairwise finite values. Orange: equal-count-bin medians. Axes show the central 99% of each marginal; outside counts report omitted points. Spearman rho uses all finite pairs. CV measures variability, so higher CV means less amplitude consistency.')

fig,axs=plt.subplots(4,3,figsize=(13,12),layout='constrained')
for ax,j in zip(axs.flat,range(10,22)): scatter(ax,example,0,j,'active')
fig.suptitle('Spatial vs amplitude | '+example['sid']+' | fraction active > 0.1')
save(fig,'spatial_amplitude_active','Spatial/amplitude comparisons: active cells','As in the supplied scatter, restrict to fraction active > 0.1. Otherwise the same finite-pair statistics and percentile view. The original stored quality mask is used for quality-subset summaries; rounding at exact thresholds can differ from recreating the mask.')

# Full atlas: all 231 unique pairs across the 22 per-slot metrics.
pairs=list(itertools.combinations(range(len(metrics)),2))
for page,start in enumerate(range(0,len(pairs),20),1):
    fig,axs=plt.subplots(5,4,figsize=(16,18),layout='constrained')
    chunk=pairs[start:start+20]
    for ax,(i,j) in zip(axs.flat,chunk): scatter(ax,example,i,j)
    for ax in list(axs.flat)[len(chunk):]: ax.axis('off')
    fig.suptitle(f'All metric combinations | {example["sid"]}, env {example["env"]} (slot {example["slot"]+1}) | page {page}/12')
    save(fig,f'pair_atlas_{page:02}',f'Pairwise scatter atlas {page}/12','Every unique pair across the 22 slot metrics appears once across these 12 pages (231 panels). Same example session, finite-pair rho, orange binned medians, and central-99% view as above. Orange lines are descriptive, not fitted predictions.')

# Within-session correlations, aggregated first within each mouse; avoid pooling cells across mice.
corr_rows=[]
for r in records:
    for subset in ['all','quality']:
        x=r['x'][selected(r,subset)]; mat=pd.DataFrame(x).corr(method='spearman',min_periods=20).to_numpy()
        for i in range(len(metrics)):
            for j in range(i,len(metrics)):
                keep=np.isfinite(x[:,i])&np.isfinite(x[:,j]); n=keep.sum()
                if n>=20 and np.ptp(x[keep,i])>0 and np.ptp(x[keep,j])>0:
                    rho=mat[i,j]
                    corr_rows.append(dict(mouse=r['mouse'],session=r['sid'],slot=r['slot'],subset=subset,i=i,j=j,rho=rho,n=int(n)))
corr=pd.DataFrame(corr_rows);corr.to_csv(OUT/'within_session_correlations.csv',index=False)
for subset in ['all','quality']:
    fig,axs=plt.subplots(1,3,figsize=(24,9),layout='constrained')
    for slot,ax in enumerate(axs):
        z=corr.query('subset==@subset and slot==@slot').groupby(['mouse','i','j']).rho.median().groupby(['i','j']).agg(['median','count'])
        mat=np.full((22,22),np.nan)
        for (i,j),v in z.iterrows():
            if v['count']>=3:mat[i,j]=mat[j,i]=v['median']
        im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(22),labels,rotation=90,fontsize=7);ax.set_yticks(range(22),labels,fontsize=7);ax.set_title(f'{slots[slot]} environment',color=colors[slot])
    fig.colorbar(im,ax=axs,label='Median mouse Spearman rho',shrink=.65)
    fig.suptitle(f'Within-session metric relationships | {subset} cells | mouse-balanced summary')
    save(fig,f'correlations_{subset}',f'All-pairs cohort correlations: {subset}','Spearman rho computed separately in each session/environment with >=20 finite pairs; median across sessions per mouse, then median across mice. A pair needs >=3 mice. Quality selection restricts both reliability and fraction active, so correlations can change through range restriction. No p-values or independence claims.')

fig,axs=plt.subplots(4,3,figsize=(13,12),layout='constrained')
for ax,(key,label) in zip(axs.flat,amp):
    j=keys.index(key)
    for slot,col in enumerate(colors):
        curves=[]
        # Per-mouse empirical CDF on a common grid; each session has equal weight within mouse.
        values=np.concatenate([r['x'][:,j][np.isfinite(r['x'][:,j])] for r in records if r['slot']==slot]); bounds=np.quantile(values,[.005,.995]); grid=np.linspace(*bounds,200)
        for mouse in mice:
            cc=[]
            for r in records:
                if r['mouse']!=mouse or r['slot']!=slot:continue
                v=r['x'][:,j];v=np.sort(v[np.isfinite(v)])
                if len(v):cc.append(np.searchsorted(v,grid,side='right')/len(v))
            if cc:curves.append(np.mean(cc,axis=0))
        if curves:ax.plot(grid,np.mean(curves,axis=0),color=col,label=slots[slot])
    ax.set(title=label,ylabel='Cumulative fraction',xlabel='Metric value (central 99% view)')
axs.flat[0].legend(frameon=False);fig.suptitle('Amplitude distributions | equal sessions within mouse, equal mice within slot')
save(fig,'amplitude_distributions','Amplitude distributions by environment','Finite-only empirical CDFs, equal sessions within mouse and equal mice within slot. Each slot uses its own pooled central-99% x range. Tails remain in CDF denominators. These curves combine experience levels and do not isolate familiarity effects.')

fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for slot,col in enumerate(colors):
    rr=[r for r in records if r['slot']==slot]
    for ax,field,label in zip(axs,['n','quality_fraction','gain_valid'],['ROI count (session median across mice)','Quality-filtered fraction','Finite Gaussian-gain CV fraction']):
        data=[]
        for r in rr:
            val=len(r['q']) if field=='n' else np.mean(r['q']) if field=='quality_fraction' else np.mean(np.isfinite(r['x'][:,keys.index('trial_gain_cv')]))
            data.append({'mouse':r['mouse'],'within':r['within'],'v':val})
        p=pd.DataFrame(data).pivot(index='mouse',columns='within',values='v');ax.plot(p.columns,p.median(),color=col,label=slots[slot]);ax.set(title=label,xlabel='Session within environment')
axs[0].legend(frameon=False);fig.suptitle('Coverage and selection diagnostics')
save(fig,'coverage','Coverage and selection','Session summaries followed by median across mice, including one-mouse tails in this diagnostic only. Finite gain coverage helps identify changes in which cells contribute to gain summaries.')
fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
for slot,col in enumerate(colors):
    z=pd.DataFrame([dict(mouse=r['mouse'],within=r['within']) for r in records if r['slot']==slot]);n=z.groupby('within').mouse.nunique();ax.plot(n.index,n.values,'o-',color=col,label=slots[slot])
ax.axhline(3,color='.6',ls='--');ax.set(xlabel='Session within environment',ylabel='Number of mice',title='Support for experience curves');ax.legend(frameon=False)
save(fig,'mouse_support','Mouse support across experience','Main experience plots omit points below the dashed 3-mouse threshold. Numbers of ROIs, finite metric values, and session-level summaries are exported to CSV.')

# Concise numerical findings for inspection and report text.
findings=[]
for key in ['trial_rms_cv','trial_mean_cv','trial_pf_dot_cv','trial_gain_cv']:
    j=keys.index(key);z=corr.query('subset=="all" and i==0 and j==@j').groupby('mouse').rho.median()
    findings.append(f'Spatial reliability vs {dict(metrics)[key]}: median within-mouse rho {z.median():.2f} across {len(z)} mice (sessions and slots combined within mouse).')
manifest={'sessions':len(rows),'mice':len(mice),'session_environments':len(records),'roi_environment_observations':sum(len(r['q']) for r in records),'figures':len(sections),'configuration':rows[0][3],'example':{k:example[k] for k in ['sid','slot','env']},'findings':findings}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
intro=f'''<h1>Place-cell properties: exploratory report</h1><p>{len(rows)} stored v9 sessions, {len(mice)} mice, {len(records)} session/environment combinations. {len(sections)} figures. Configuration: <code>{html.escape(rows[0][3])}</code>.</p>
<p>This is a scratch analysis of cached results, with no new formal plotters. Black / blue / green mean first / second / third environment experienced by each mouse, rather than a common raw environment ID. Within-environment session numbers count recorded sessions where that slot was present. There is no cross-session ROI tracking.</p>
<p>Quality-filtered cells require spatial reliability &gt;0.3 AND fraction active &gt;0.1; complement is every other ROI. This differs from a reliability-only place-cell cutoff of 0.7. Amplitude CV is standard deviation / mean across trials, not a correlation reliability score. Higher positive CV means more trial-to-trial variability. Tiny or signed means can make CV unstable; inspect medians, means, tails and finite coverage together.</p>
<p>Trial RMS and mean activity summarize across position; PF projection uses the unit-L2 all-trial place field; Gaussian gain uses the fitted Gaussian field. Means/variances/CVs summarize across trials. These are descriptive amplitude measures from all-trial maps; their relationships to spatial metrics may share data and algebraic dependencies. Normalized RMS and R2 are mechanically related (R2 = 1 - normalized RMS squared, where defined); PF variance fraction is prediction variance / activity variance, not explained variance.</p>
<p>Only the 22 metrics available per environment slot enter the 231-pair atlas. Older best-environment-only activity-shape and correlation statistics are excluded from that atlas to keep ROI/environment matching consistent.</p>
<p>In this cache the median Gaussian gain is essentially 1; its median experience panel uses a fixed 0.95-1.05 scale to avoid magnifying numerical differences. Spearman correlations involving nearly constant gain means can rank tiny numerical fluctuations and should not be interpreted as biological effects. Cached v9 results span computation dates; this report reads them as stored, without recomputing them with the current source.</p><h2>Observed CV relationships</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in findings)}</ul>
<p>These are descriptive correlations; no hypothesis tests. Quality-subset comparisons are conditional on the selection criteria.</p>
<p>Downloads: <a href="session_summaries.csv">session summaries</a> | <a href="within_session_correlations.csv">pairwise correlations and counts</a> | <a href="provenance.csv">source UIDs and timestamps</a> | <a href="make_report.py">reproduction script</a>.</p>'''
nav='<h2>Figures</h2><ol>'+''.join(f'<li><a href="#{n}">{html.escape(t)}</a></li>' for n,t,c in sections)+'</ol>'
body=''.join(f'<section id="{n}"><h2>{html.escape(t)}</h2><p>{html.escape(c)}</p><a href="figures/{n}.png"><img loading="lazy" src="figures/{n}.png"></a></section>' for n,t,c in sections)
(OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Place-cell properties exploration</title><style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#222}p{max-width:1100px;line-height:1.5}img{width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:48px;padding-top:12px}a{color:#164eac}</style>'+intro+nav+body,encoding='utf-8')
print(json.dumps(manifest,indent=2))




