"""Activity-level follow-up to the v9 exploratory report; no raw-data recomputation."""
from pathlib import Path
import json,sqlite3,pickle,tomllib,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;FIG=OUT/'figures'
with (ROOT/'paths.toml').open('rb') as f:STORE=Path(tomllib.load(f)['paths']['storage'])/'dimensionality-manuscript/cache/pipeline_v2'
c=sqlite3.connect(f'file:{(STORE/"results.db").as_posix()}?mode=ro&immutable=1',uri=True)
rows=c.execute("SELECT result_uid,session_id,analysis_key FROM results WHERE analysis_type='pfpred_quality' AND schema_version='v9' AND result_stored=1 ORDER BY session_id").fetchall()
assert len({r[2] for r in rows})==1
colors=['black','blue','green'];slots=['1st','2nd','3rd'];subsets=['all','quality','complement'];subsetlabels=['All cells','Quality-filtered','Complement']
records=[];audit=[]
for uid,sid,akey in rows:
 d=pickle.loads((STORE/'blobs'/f'{uid}.pkl').read_bytes());mouse,date,sess=sid.rsplit('.',2)
 for slot,env in enumerate(d['env_slot_ids']):
  if not np.isfinite(env) or not np.any(np.isfinite(d['reliability_slot'][slot])):continue
  values={k:np.asarray(d[k][slot],float) for k in ['reliability_slot','fraction_active_slot','trial_rms_mean','trial_mean_mean','trial_pf_dot_mean','pf_peak','trial_rms_variance','var_total_slot','trial_rms_cv','trial_mean_cv','trial_pf_dot_cv','trial_gain_cv']}
  values['overall_rms']=np.sqrt(values['trial_rms_variance']+values['trial_rms_mean']**2)
  values['trial_rms_sd']=np.sqrt(values['trial_rms_variance'])
  q=np.asarray(d['quality_filtered_roi_mask_slot'][slot],bool)
  r=dict(sid=sid,mouse=mouse,date=date,sess=int(sess),slot=slot,env=int(env),values=values,q=q);records.append(r)
  fa=values['fraction_active_slot'];cv=values['trial_rms_cv'];keep=np.isfinite(fa)&np.isfinite(cv)&(fa>0)&(fa<=1)
  errors=np.abs(fa[keep]-1/(1+cv[keep]**2));audit.append(dict(session=sid,slot=slot,n=int(keep.sum()),median_abs_error=float(np.median(errors)),max_abs_error=float(np.max(errors))))
mice=sorted({r['mouse'] for r in records})
for mouse in mice:
 for slot in range(3):
  rr=sorted([r for r in records if r['mouse']==mouse and r['slot']==slot],key=lambda r:(r['date'],r['sess']))
  for index,r in enumerate(rr,1):r['within']=index
entries=[]
for r in records:
 for subset in subsets:
  mask=np.ones(len(r['q']),bool) if subset=='all' else r['q'] if subset=='quality' else ~r['q']
  for key,v in r['values'].items():
   v=v[mask&np.isfinite(v)]
   entries.append(dict(session=r['sid'],mouse=r['mouse'],slot=r['slot'],within=r['within'],subset=subset,metric=key,n=len(v),median=np.median(v) if len(v) else np.nan,mean=np.mean(v) if len(v) else np.nan))
summ=pd.DataFrame(entries);summ.to_csv(OUT/'activity_session_summaries.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'participation_identity_audit.csv',index=False)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
sections=[];curve_rows=[]
def save(fig,name,title,caption):
 fig.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white');plt.close(fig);sections.append((name,title,caption));print(name,flush=True)
def curve(ax,key,subset,stat):
 for slot,col in enumerate(colors):
  z=summ.query('metric==@key and subset==@subset and slot==@slot');p=z.pivot(index='mouse',columns='within',values=stat).reindex(mice)
  n=p.count();avg=p.mean();sem=p.std()/np.sqrt(n);keep=n>=3
  # Preserve missing x positions, so lines cannot bridge absent cohort support.
  yy=avg.where(keep);ee=sem.where(keep);x=p.columns.to_numpy()
  ax.plot(x,yy,color=col,label=slots[slot]);ax.fill_between(x,yy-ee,yy+ee,color=col,alpha=.16)
  for t in x:curve_rows.append(dict(metric=key,subset=subset,roi_statistic=stat,slot=slot,within=int(t),mice=int(n[t]),mean=avg[t],sem=sem[t]))
 ax.set_xlabel('Session within environment')
for stat in ['mean','median']:
 fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
 for ax,subset,label in zip(axs,subsets,subsetlabels):
  curve(ax,'reliability_slot',subset,stat)
  ax.set(title=label,ylabel='Spatial reliability (ROI '+stat+')')
 axs[0].legend(frameon=False)
 fig.suptitle(f'Spatial reliability with familiarity | per-session ROI {stat}, then mouse mean +/- SEM')
 save(fig,f'spatial_reliability_{stat}',f'Spatial reliability: ROI {stat}',f'Same environment-slot colors, observed-environment session alignment, and cell subsets as the activity figures. Each mouse contributes the {stat} of finite ROI reliability values in each session; curves are the equal-mouse mean +/- between-mouse SEM, with at least 3 mice per point. Axes use independent y ranges. Quality-filtered cells must exceed reliability 0.3 AND fraction active 0.1, so their reliability distribution is truncated by selection; complement fails either criterion. These are changing session populations, not tracked-cell trajectories.')
level=[('trial_mean_mean','Mean activity across position/trials'),('trial_rms_mean','Mean trial RMS'),('trial_pf_dot_mean','Mean PF projection'),('pf_peak','Peak of mean place field')]
for stat in ['median','mean']:
 fig,axs=plt.subplots(4,3,figsize=(14,11),layout='constrained')
 for i,(key,label) in enumerate(level):
  for j,subset in enumerate(subsets):
   ax=axs[i,j];curve(ax,key,subset,stat);ax.set_title(f'{label}\n{subsetlabels[j]}');ax.set_ylabel('ROI '+stat+' (standardized units)')
 axs[0,0].legend(frameon=False);fig.suptitle(f'Activity amplitude with familiarity | per-session ROI {stat}, then equal-mouse mean +/- SEM')
 save(fig,f'activity_levels_{stat}',f'Activity amplitude: ROI {stat}',f'Each panel shows an amplitude measure, rather than participation or CV. First reduce ROIs within each session using the {stat}, then average mice equally; shading is between-mouse SEM. At least 3 mice per point. RMS, mean activity and PF peak are in per-cell full-session standard-deviation units; projection also depends on the number of position bins. Quality filtering uses reliability >0.3 AND fraction active >0.1; complement is its exact stored complement. These populations can change over time.')
energy=[('fraction_active_slot','Trial participation (not amplitude)'),('overall_rms','RMS across trials and position'),('trial_rms_sd','SD of trial RMS'),('var_total_slot','Temporal activity variance within env')]
for stat in ['median','mean']:
 fig,axs=plt.subplots(4,3,figsize=(14,11),layout='constrained')
 for i,(key,label) in enumerate(energy):
  for j,subset in enumerate(subsets):
   ax=axs[i,j];curve(ax,key,subset,stat);ax.set_title(f'{label}\n{subsetlabels[j]}');ax.set_ylabel('ROI '+stat)
 axs[0,0].legend(frameon=False);fig.suptitle(f'Participation, activity energy and variability | ROI {stat} -> mouse mean +/- SEM')
 save(fig,f'activity_energy_{stat}',f'Activity energy and participation: ROI {stat}','Overall RMS is sqrt(mean(trial RMS)^2 + variance(trial RMS)) for each cell: the square root of mean squared trial RMS. It weights each valid trial equally and each finite position equally within trial; it is not a time-weighted firing rate. SD of trial RMS is sqrt(variance), an absolute variability measure rather than CV. Temporal variance within environment is normalized by full-session cell variance, so it measures relative allocation across environments. Participation is included as a reference. Same mouse weighting and minimum support as the amplitude figure.')
example=next(r for r in records if r['sid']=='ATL027.2023-07-27.701' and r['env']==3)
fig,axs=plt.subplots(1,4,figsize=(16,4.1),layout='constrained');identity_stats=[]
for ax,key,label in zip(axs,['trial_rms_cv','trial_mean_cv','trial_pf_dot_cv','trial_gain_cv'],['RMS CV','Mean activity CV','PF projection CV','Gaussian gain CV']):
 fa=example['values']['fraction_active_slot'];v=example['values'][key];keep=np.isfinite(fa)&np.isfinite(v)&(fa>0)&(fa<=1);x=fa[keep];y=v[keep]
 xl=np.quantile(x,[.005,.995]);yl=np.quantile(y,[.005,.995]);vis=(x>=xl[0])&(x<=xl[1])&(y>=yl[0])&(y<=yl[1]);ax.scatter(x[vis],y[vis],s=2,alpha=.18,color='black')
 xx=np.linspace(max(.001,xl[0]),min(1,xl[1]),300);ax.plot(xx,np.sqrt(1/xx-1),color='darkorange',label='RMS identity')
 ax.set(xlim=xl,ylim=yl,xlabel='Fraction active',ylabel=label,title=f'rho={spearmanr(x,y).statistic:.3f}, n={len(x):,}')
 identity_stats.append(dict(metric=key,rho=float(spearmanr(x,y).statistic),n=len(x)))
axs[0].legend(frameon=False);fig.suptitle('RMS CV is a re-expression of participation; the other CVs need not follow the same curve')
save(fig,'participation_cv_identity','The participation/CV identity','ATL027.2023-07-27.701, environment 3 (first experience slot), all finite cells. Orange is sqrt(1 / fraction active - 1), not a fit. Axes show marginal central 99%; rho uses every finite positive-participation pair. Exact RMS equivalence assumes the same finite trials, population variance (ddof=0), and nonzero mean; epsilon stabilization, floating point, or different missing-trial handling can introduce small deviations. Other CVs measure a different trial amplitude.')
# Same mice at both endpoints, to expose effects of changing cohort support.
paired=[]
fig,axs=plt.subplots(4,3,figsize=(13,11),layout='constrained')
for i,(key,label) in enumerate(level):
 for j,subset in enumerate(subsets):
  ax=axs[i,j]
  for slot,col in enumerate(colors):
   z=summ.query('metric==@key and subset==@subset and slot==@slot and within in [1,5]');p=z.pivot(index='mouse',columns='within',values='median').reindex(columns=[1,5]).dropna();delta=p[5]-p[1]
   for mouse,value in delta.items():paired.append(dict(mouse=mouse,slot=slot,subset=subset,metric=key,first=p.loc[mouse,1],fifth=p.loc[mouse,5],delta=value))
   if len(delta):
    ax.scatter(slot+np.linspace(-.12,.12,len(delta)),delta,color=col,s=18,alpha=.7);mean=delta.mean();sem=delta.std()/np.sqrt(len(delta)) if len(delta)>1 else np.nan;ax.errorbar(slot,mean,yerr=sem,fmt='_',color=col,markersize=20,lw=2)
   ax.text(slot,.98,f'n={len(delta)}',transform=ax.get_xaxis_transform(),ha='center',va='top',color=col)
  ax.axhline(0,color='.6',lw=.8);ax.set(xticks=range(3),xticklabels=slots,ylabel='Change in ROI median (5th - 1st)',title=f'{label}\n{subsetlabels[j]}')
fig.suptitle('Within-mouse change from first to fifth exposure | equal mice, paired session summaries')
save(fig,'activity_paired_change','Same-mouse activity changes: first to fifth exposure','Dots are individual mice with finite session summaries at both exposures. Bars are the equal-mouse mean change +/- SEM. This comparison holds the mouse cohort fixed across endpoints but does not track the same cells; quality membership and recorded ROI population can change. Fifth exposure has only 3 mice for the third slot, so read that result cautiously.')
pd.DataFrame(paired).to_csv(OUT/'activity_paired_changes.csv',index=False);pd.DataFrame(curve_rows).drop_duplicates().to_csv(OUT/'activity_curve_support.csv',index=False)
manifest=dict(sessions=len(rows),mice=len(mice),session_environments=len(records),figures=len(sections),max_identity_error=max(r['max_abs_error'] for r in audit),example_correlations=identity_stats)
(OUT/'activity_manifest.json').write_text(json.dumps(manifest,indent=2))
intro=f'''<h1>Activity levels, trial participation and familiarity</h1><p>{len(rows)} cached v9 sessions; {len(mice)} mice; {len(records)} observed session/environment combinations. {len(sections)} exploratory figures, using black/blue/green for experience slots 1/2/3. No raw sessions or fits were recomputed. Least-squares gain is not yet present in these cached v9 results.</p>
<h2>What does the tight scatter mean?</h2><p>Let a be a cell's trial RMS amplitudes over T trials. The normalized participation ratio is (sum a)^2 / (T sum a^2) = mean(a)^2 / mean(a^2). With population variance, CV^2 = variance(a)/mean(a)^2, hence <strong>fraction active = 1 / (1 + RMS CV^2)</strong>. Equivalently RMS CV = sqrt(1/fraction active - 1). This is an algebraic identity, not evidence for a separate mechanism. The maximum absolute difference from the identity among matched positive finite pairs across this cache is {manifest['max_identity_error']:.3g}; per-session audits are downloadable below.</p>
<p>The implementation first normalizes RMS amplitudes to sum to approximately 1, then computes normalized participation ratio. That normalization cancels algebraically. Multiplying every trial's amplitude by a constant leaves both participation and CV unchanged. Participation can rise because more trials are active, because existing active trials become more similar, or both. It is not a direct count of trials with events. A binary equal-amplitude on/off signal is a special case where participation equals the active-trial fraction.</p>
<p>The same identity holds if participation is computed separately from each other amplitude measure, but the stored fraction active uses RMS only. Therefore PF projection, mean-activity and Gaussian-gain CV are correlated with it rather than identically determined by it. Similarities across these amplitudes may reflect shared trial-to-trial modulation. These plots do not establish that recruitment alone explains the familiarity effect.</p>
<h2>Activity measurements beyond participation</h2><p>Mean trial RMS, mean activity across positions/trials, PF projection, PF peak and overall RMS measure amplitude and can change even with fixed participation. Mean Gaussian gain is a relative scale tied to its own template and is near 1 in these results, so it is not an activity-level measure.</p>
<p><strong>Normalization limit:</strong> pfpred_quality divides each ROI's activity by its full-session standard deviation (and subtracts its median when configured). These figures show standardized activity levels within environments, not absolute fluorescence, event rates or firing rates. Temporal variance is also relative to full-session scale. Changing cell populations and a quality screen that includes participation can affect the curves.</p>
<p>To distinguish recruitment from amplitude stabilization, useful next measurements from trial-resolved data are: the actual fraction of trials exceeding a fixed noise-calibrated event threshold; amplitude conditional on exceeding that threshold; and mean unstandardized event signal or event count per running second. A fixed threshold in standardized units is possible but inherits the session normalization; counting any positive sigrebase activity would require checking its baseline/noise first. Conditional-amplitude and thresholded trial-fraction measures are not recoverable from the cached mean/variance/CV summaries alone.</p>
<p><a href="report.html">Original report</a> | <a href="activity_session_summaries.csv">Session summaries</a> | <a href="activity_paired_changes.csv">Paired mouse changes</a> | <a href="activity_curve_support.csv">Curve values and mouse support</a> | <a href="participation_identity_audit.csv">Identity audit</a> | <a href="make_activity_report.py">Reproduction script</a></p>'''
nav='<ol>'+''.join(f'<li><a href="#{n}">{html.escape(t)}</a></li>' for n,t,c in sections)+'</ol>'
body=''.join(f'<section id="{n}"><h2>{html.escape(t)}</h2><p>{html.escape(c)}</p><a href="figures/{n}.png"><img src="figures/{n}.png" loading="lazy"></a></section>' for n,t,c in sections)
(OUT/'activity_report.html').write_text('<!doctype html><meta charset="utf-8"><title>Activity and familiarity</title><style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#222}p{line-height:1.5;max-width:1150px}img{width:100%;height:auto}section{margin-top:40px;border-top:1px solid #ddd}a{color:#164eac}</style>'+intro+nav+body,encoding='utf-8')
original=OUT/'report.html';s=original.read_text(encoding='utf-8');link='<p><strong>Follow-up: <a href="activity_report.html">Activity levels, participation identity, and familiarity — six new figures</a></strong></p>'
s=s.replace('familiarity — six new figures','familiarity — eight figures')
if link not in s:s=s.replace('<h1>Place-cell properties: exploratory report</h1>','<h1>Place-cell properties: exploratory report</h1>'+link);original.write_text(s,encoding='utf-8')
print(json.dumps(manifest,indent=2))
