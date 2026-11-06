"""Population report from saved v11 PF/RMS results; no raw-data loading or refits.
Run with the vrAnalysis Conda interpreter. Outputs HTML, PNG/SVG and summary CSVs.
"""
from pathlib import Path
import html
import pickle
import sqlite3
import tomllib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
DEFS = {'all': 'All trials (shared)', 'heldout': 'Held-out RMS', 'independent': 'Independent trials'}
SAVED = {'all': 'all_from_all', 'heldout': 'cv', 'independent': 'independent_cv'}
COLORS = {'all': '#b77828', 'heldout': '#4679b8', 'independent': '#168477'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 150, 'axes.titleweight': 'bold'})
SECTIONS = []

def save(fig, name, title, caption):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    SECTIONS.append((name, title, caption))
    print(name, flush=True)

def read():
    with (ROOT / 'paths.toml').open('rb') as f:
        store = Path(tomllib.load(f)['paths']['storage']) / 'dimensionality-manuscript/cache/pipeline_v2'
    # Immutable is safe for this offline snapshot: refuse a live WAL to avoid missing committed rows.
    assert not Path(str(store / 'results.db') + '-wal').exists(), 'Store has a live WAL; take a consistent snapshot first.'
    con = sqlite3.connect((store / 'results.db').as_uri() + '?mode=ro&immutable=1', uri=True)
    rows = pd.read_sql_query("SELECT result_uid,session_id,analysis_key,analysis_summary,computed_at FROM results WHERE analysis_type='pfpred_quality' AND schema_version='v11' AND result_stored=1 ORDER BY session_id", con)
    con.close()
    assert len(rows.analysis_key.unique()) == 1 and not rows.session_id.duplicated().any(), 'Ambiguous configuration'
    chunks = []
    for row in rows.itertuples():
        d = pickle.loads((store / 'blobs' / f'{row.result_uid}.pkl').read_bytes())
        best = int(d['best_env_slot']) if np.isfinite(d['best_env_slot']) else -1
        for slot, env in enumerate(d['env_slot_ids']):
            if not np.isfinite(env): continue
            values = {name: np.asarray(d[f'pf_rms_r_{key}_slot'][slot], float) for name, key in SAVED.items()}
            values.update(spatial=np.asarray(d['reliability_slot'][slot], float),
                          rms_cv=np.asarray(d['trial_rms_cv'][slot], float),
                          gain_cv=np.asarray(d['trial_gain_least_squares_cv'][slot], float),
                          gaussian_cv=np.asarray(d['trial_gain_cv'][slot], float),
                          active=np.asarray(d['fraction_active_slot'][slot], float),
                          quality=np.asarray(d['quality_filtered_roi_mask_slot'][slot], bool))
            n = len(values['spatial'])
            assert all(v.shape == (n,) for v in values.values())
            for name, key in SAVED.items():
                assert np.allclose(values[name], d[f'pf_rms_r_{key}'], equal_nan=True) if slot == best else True
                assert np.all(np.abs(values[name][np.isfinite(values[name])]) <= 1 + 1e-9)
            chunks.append(pd.DataFrame(values).assign(mouse=row.session_id.rsplit('.', 2)[0],
                session=row.session_id, slot=slot + 1, environment=int(env), best=slot == best))
    data = pd.concat(chunks, ignore_index=True)
    rows.to_csv(OUT / 'provenance.csv', index=False)
    return data, rows, store

def aggregate(frame, group, values):
    # Equal environments within session, sessions within mouse, retaining mice.
    return frame.groupby(['mouse', 'session'] + group, observed=True)[values].mean().groupby(['mouse'] + group, observed=True).mean().reset_index()

def distribution(data, metric, edges):
    rows = []
    for (mouse, session), g in data.groupby(['mouse', 'session']):
        v = g[metric].to_numpy(); v = v[np.isfinite(v)]
        if not len(v): continue
        p = np.histogram(v, edges)[0] / len(v)
        assert np.isclose(p.sum(), 1)
        rows.extend(dict(mouse=mouse, session=session, bin=i, probability=a) for i, a in enumerate(p))
    return pd.DataFrame(rows).groupby(['mouse', 'bin']).probability.mean().unstack('bin')

def curve(data, x, y, edges, subset='all'):
    rows = []
    for (mouse, session), g in data.groupby(['mouse', 'session']):
        if subset == 'active': g = g[g.active > .1]
        a, b = g[x].to_numpy(), g[y].to_numpy()
        for i in range(len(edges) - 1):
            good = np.isfinite(a) & np.isfinite(b) & (a >= edges[i]) & ((a < edges[i+1]) | ((i == len(edges)-2) & (a == edges[i+1])))
            if good.sum() >= 20:
                rows.append(dict(mouse=mouse, session=session, bin=i, mean=b[good].mean(), n=int(good.sum())))
    z = pd.DataFrame(rows)
    z.assign(predictor=x, outcome=y, subset=subset).to_csv(OUT / f'curve_{x}_{y}_{subset}.csv', index=False)
    return z.groupby(['mouse', 'bin'])['mean'].mean().unstack('bin').reindex(columns=range(len(edges)-1))

def draw_curve(ax, p, centers, color, label, thin=False):
    if thin:
        for _, row in p.iterrows(): ax.plot(centers, row, color=color, alpha=.19, lw=.8)
    n = p.count(); mean = p.mean(); sem = p.std(ddof=1) / np.sqrt(n)
    good = n >= 3
    mean = mean.where(good); sem = sem.where(good)
    ax.plot(centers, mean, color=color, lw=2.5, label=label)
    ax.fill_between(centers, mean-sem, mean+sem, color=color, alpha=.15)
    return n

def main():
    FIG.mkdir(exist_ok=True)
    data, provenance, store = read()
    best = data[data.best].copy()
    mice = sorted(best.mouse.unique())
    palette = dict(zip(mice, plt.cm.tab20(np.linspace(0, 1, len(mice)))))
    print('loaded', len(provenance), 'sessions', len(mice), 'mice', len(best), 'best-env ROI observations', flush=True)
    summaries = []
    for (mouse, session, slot), g in data.groupby(['mouse', 'session', 'slot']):
        for subset, mask in [('all', np.ones(len(g), bool)), ('quality', g.quality.to_numpy())]:
            for name in DEFS:
                v = g.loc[mask, name].to_numpy(); v = v[np.isfinite(v)]
                summaries.append(dict(mouse=mouse, session=session, slot=slot, best=bool(g.best.iloc[0]), subset=subset,
                    metric=name, n_total=int(mask.sum()), n_finite=len(v), mean=np.mean(v) if len(v) else np.nan,
                    median=np.median(v) if len(v) else np.nan, positive=np.mean(v>0) if len(v) else np.nan))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'session_summaries.csv', index=False)
    mouse_summary = aggregate(summary[summary.best], ['subset', 'metric'], ['mean', 'median', 'positive'])
    mouse_summary.to_csv(OUT / 'mouse_summaries.csv', index=False)
    coverage = best.groupby('mouse').agg(sessions=('session', 'nunique'), roi_observations=('independent', 'size'), finite_independent=('independent', 'count'))
    coverage.to_csv(OUT / 'coverage.csv')
    provenance.assign(included_best=provenance.session_id.isin(best.session)).to_csv(OUT / 'session_coverage.csv', index=False)
    edges = np.linspace(-1, 1, 41); centers = (edges[:-1]+edges[1:])/2
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), layout='constrained')
    for ax, name in zip(axs, DEFS):
        p = distribution(best, name, edges)
        for mouse, row in p.iterrows(): ax.plot(centers, row/.05, color=palette[mouse], alpha=.5, lw=.9)
        draw_curve(ax, p/.05, centers, COLORS[name], 'Equal-mouse mean ± SEM')
        ax.set(title=DEFS[name], xlabel='PF mean–RMS correlation r', ylabel='Probability density', xlim=(-1,1))
        ax.axvline(0, color='.6', ls=':', lw=1)
    axs[0].legend(fontsize=8)
    save(fig, '01_distributions', 'How strongly does variability follow the mean field?', 'Thin lines: individual mice. Within each mouse, average the normalized session histograms; thick line and band: equal-mouse mean ± between-mouse SEM. All cells, best environment per session. All bins are 0.05 wide; no smoothing. The all-trial comparison shares data between the mean and residual map, so its high correlation alone is not independent evidence.')
    fig, axs = plt.subplots(4, 4, figsize=(14, 11), layout='constrained', sharex=True, sharey=True)
    for ax, mouse in zip(axs.flat, mice):
        g = best[best.mouse == mouse]
        for _, session in g.groupby('session'):
            v = session.independent.dropna(); ax.hist(v, bins=edges, density=True, histtype='step', color=palette[mouse], alpha=.16)
        p = distribution(g, 'independent', edges).iloc[0]
        ax.plot(centers, p/.05, color=palette[mouse], lw=2)
        ax.set_title(f'{mouse} | {g.session.nunique()} sessions', fontsize=10)
        ax.axvline(0, color='.6', ls=':', lw=1)
        ax.set(xlabel='Independent-trial r', ylabel='Density')
    for ax in list(axs.flat)[len(mice):]: ax.set_visible(False)
    save(fig, '02_each_mouse', 'Neuron distributions within each mouse', 'Independent-trial comparison. Faint steps: each session’s neuron distribution. Colored curve: equal-session average for that mouse. These are repeated ROI observations; the same neuron may appear on multiple days, and cell identities have not been tracked.')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for name in DEFS:
        for subset, ls in [('all', '-'), ('quality', '--')]:
            p = distribution(best if subset == 'all' else best[best.quality], name, edges).cumsum(axis=1)
            axs[0].plot(edges[1:], p.mean(), color=COLORS[name], ls=ls, label=f'{DEFS[name]} / {subset}')
        z = mouse_summary.query('metric == @name and subset == "all"').set_index('mouse').reindex(mice)
        x = np.arange(len(mice))
        axs[1].plot(x, z['mean'], 'o-', color=COLORS[name], alpha=.8, ms=4, label=DEFS[name])
    axs[0].set(xlabel='PF mean–RMS r', ylabel='Cumulative fraction', xlim=(-1,1), ylim=(0,1)); axs[0].legend(fontsize=7)
    axs[1].set(ylabel='Mouse mean r', xticks=np.arange(len(mice)), xticklabels=mice); axs[1].tick_params(axis='x', rotation=65); axs[1].legend(fontsize=8)
    save(fig, '03_subsets', 'Distribution tails, selection, and mouse means', 'CDFs use the same equal-session then equal-mouse weights as Figure 1. Dashed: quality selection (spatial reliability > 0.3 and fraction active > 0.1); solid: all cells. This selection directly depends on spatial reliability. Right: each dot is the mean of session neuron means, with mice ordered alphabetically.')
    spatial_edges = np.linspace(-1, 1, 21); sc = (spatial_edges[:-1]+spatial_edges[1:])/2
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    spatial_curves = {}
    for ax, name in zip(axs, DEFS):
        p = curve(best, 'spatial', name, spatial_edges); spatial_curves[name] = p
        n = draw_curve(ax, p, sc, COLORS[name], 'All cells', thin=True)
        q = curve(best, 'spatial', name, spatial_edges, 'active')
        draw_curve(ax, q, sc, '#444444', 'Fraction active > 0.1')
        ax.set(title=DEFS[name], xlabel='Spatial reliability', ylabel='Mean PF mean–RMS r', ylim=(-1,1.02))
        ax.axhline(0, color='.7', ls=':', lw=1)
    axs[0].legend(fontsize=8)
    save(fig, '04_spatial_curves', 'The relationship with spatial reliability', 'Fixed reliability bins, width 0.1. Each session bin needs ≥20 finite neuron pairs; average session-bin means within mouse, then mice equally. Thin colored lines: mice. Thick curves ± between-mouse SEM shown only with ≥3 mice. Gray: activity-selected sensitivity analysis without selecting on spatial reliability. Curves describe association; reliability and PF/RMS summaries use overlapping trials.')
    correlations = []
    for (mouse, session), g in best.groupby(['mouse', 'session']):
        for x in ['spatial', 'rms_cv', 'gain_cv', 'gaussian_cv']:
            for name in DEFS:
                v = g[[x, name]].dropna()
                if len(v) >= 20 and v[x].nunique()>1 and v[name].nunique()>1:
                    correlations.append(dict(mouse=mouse, session=session, predictor=x, metric=name, n=len(v), rho=spearmanr(v[x], v[name]).statistic))
    corr = pd.DataFrame(correlations); corr.to_csv(OUT / 'session_spearman.csv', index=False)
    mc = aggregate(corr, ['predictor', 'metric'], ['rho']); mc.to_csv(OUT / 'mouse_spearman.csv', index=False)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), layout='constrained', sharey=True)
    labels = {'spatial': 'Spatial reliability', 'rms_cv': 'Trial RMS CV', 'gain_cv': 'Least-squares gain CV', 'gaussian_cv': 'Gaussian gain CV'}
    for ax, predictor in zip(axs, labels):
        z = mc[mc.predictor == predictor].pivot(index='mouse', columns='metric', values='rho').reindex(mice)
        for mouse, row in z.iterrows(): ax.plot(range(3), row[list(DEFS)], color=palette[mouse], lw=.8, alpha=.55)
        for i, name in enumerate(DEFS):
            v = z[name].dropna(); ax.errorbar(i, v.mean(), yerr=v.std(ddof=1)/np.sqrt(len(v)), fmt='o', color=COLORS[name], capsize=4, ms=7)
        ax.axhline(0, color='.5', ls=':'); ax.set(title=labels[predictor], xticks=range(3), xticklabels=['Shared', 'Held-out', 'Independent'], ylabel='Within-session Spearman ρ → mouse mean'); ax.tick_params(axis='x', rotation=30)
    save(fig, '05_associations', 'Association strength across mice', 'Compute Spearman correlation across neurons separately within each session, then average sessions per mouse. Thin connected lines: mice; points ± SEM: equal-mouse summaries. Higher amplitude CV means less consistent amplitude, so its sign has the opposite interpretation to spatial reliability. No neuron-level significance tests: sessions and neurons are repeated observations.')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, predictor in zip(axs, ['rms_cv', 'gain_cv', 'gaussian_cv']):
        amp_edges = np.r_[np.arange(0, 6.01, .5), np.inf]
        amp_centers = np.r_[np.arange(.25,6,.5),6.25]
        for name in ['heldout', 'independent']:
            p = curve(best, predictor, name, amp_edges)
            draw_curve(ax, p, amp_centers, COLORS[name], DEFS[name], thin=name=='independent')
        ax.set(title=labels[predictor], xlabel='Amplitude CV (lower = more consistent)', ylabel='Mean PF mean–RMS r', ylim=(-1,1.02), xticks=[0,2,4,6.25], xticklabels=['0','2','4','≥6'])
    axs[0].legend(fontsize=8)
    save(fig, '06_amplitude', 'Does mean–variability coupling depend on amplitude consistency?', 'Same hierarchy and ≥20-pair session-bin / ≥3-mouse thresholds as spatial curves. Fixed CV bins of width 0.5; last bin contains every finite CV ≥6. Negative CVs are excluded from these curves (counts reported below); all finite CVs enter Spearman correlations. RMS is the primary amplitude summary; gain definitions are sensitivity analyses.')
    # Two-dimensional stratification: equal sessions per mouse per cell, then mice.
    joint = []
    xe, ye = np.array([-1,0,.3,.6,1.000001]), np.array([0,1,2,3,6,np.inf])
    for (mouse, session), g in best.groupby(['mouse','session']):
        for i in range(len(xe)-1):
            for j in range(len(ye)-1):
                v=g.loc[(g.spatial>=xe[i])&(g.spatial<xe[i+1])&(g.rms_cv>=ye[j])&(g.rms_cv<ye[j+1]),'independent'].dropna()
                if len(v)>=20: joint.append(dict(mouse=mouse,session=session,x=i,y=j,mean=v.mean(),n=len(v)))
    joint=pd.DataFrame(joint); joint.to_csv(OUT/'joint_session_bins.csv',index=False)
    jm=aggregate(joint,['x','y'],['mean']); mean=jm.groupby(['y','x'])['mean'].mean().unstack('x').reindex(index=range(5),columns=range(4)); n=jm.groupby(['y','x'])['mean'].count().unstack('x').reindex_like(mean).fillna(0)
    fig, ax=plt.subplots(figsize=(8,5),layout='constrained'); im=ax.imshow(mean.where(n>=3),origin='lower',aspect='auto',vmin=-.3,vmax=1,cmap='viridis')
    for j in range(5):
        for i in range(4):
            ax.text(i,j,f'{mean.iloc[j,i]:.2f}\n{int(n.iloc[j,i])} mice' if n.iloc[j,i]>=3 else f'{int(n.iloc[j,i])} mice',ha='center',va='center',color='white' if n.iloc[j,i]>=3 and mean.iloc[j,i]<.5 else 'black',fontsize=10)
    ax.set(xticks=range(4),xticklabels=['[-1, 0)','[0, .3)','[.3, .6)','[.6, 1]'],yticks=range(5),yticklabels=['[0, 1)','[1, 2)','[2, 3)','[3, 6)','≥6'],xlabel='Spatial reliability',ylabel='Trial RMS CV',title='Independent-trial PF mean–RMS r');fig.colorbar(im,ax=ax,label='Equal-mouse mean r')
    save(fig,'07_joint','Spatial reliability and amplitude consistency together','Coarse joint bins show the spatial relationship at comparable amplitude CV. Each session cell requires ≥20 finite neurons; average sessions within mouse, then mice. Labels give contributing mice; colors require ≥3. Descriptive stratification does not establish independent causal contributions.')
    envrows=summary.query('subset == "all"')
    envm=aggregate(envrows,['slot','metric'],['mean']);envm.to_csv(OUT/'environment_mouse_means.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharey=True)
    for ax,name in zip(axs,DEFS):
        z=envm[envm.metric==name].pivot(index='mouse',columns='slot',values='mean')
        for mouse,row in z.iterrows():ax.plot(z.columns,row,color=palette[mouse],alpha=.55,lw=.9)
        ax.errorbar(z.columns,z.mean(),yerr=z.std()/np.sqrt(z.count()),color=COLORS[name],lw=2.5,marker='o',capsize=4)
        ax.set(title=DEFS[name],xlabel='Environment experience-order slot',ylabel='Mouse mean r',xticks=[1,2,3]);ax.text(.04,.04,'Mice: '+', '.join(f'{k}: {v}' for k,v in z.count().items()),transform=ax.transAxes,fontsize=8)
    save(fig,'08_environments','Sensitivity to environment choice','All observed environments, summarized separately by the order each mouse first experienced them. Equal sessions per mouse, then equal mice ± SEM. Different slots can contain different sessions and mice; these are not matched causal estimates of familiarity. Primary figures use only the environment with the most trials in each session.')
    # Narrative and auditable summary numbers derive from the displayed hierarchy.
    stats = mouse_summary.query('subset == "all"').groupby('metric')[['mean','positive']].agg(['mean','std','count'])
    bullets=[]
    for name in DEFS:
        v=mouse_summary.query('subset == "all" and metric == @name')
        bullets.append(f'<li>{DEFS[name]}: mean r = <b>{v["mean"].mean():.3f}</b> ± {v["mean"].std()/np.sqrt(len(v)):.3f} SEM; {100*v.positive.mean():.1f}% of finite ROI observations have r &gt; 0 (mouse-balanced).</li>')
    rel=[]
    for x in ['spatial','rms_cv']:
        v=mc.query('predictor == @x and metric == "independent"').rho
        rel.append(f'{labels[x]}: average within-session Spearman ρ = <b>{v.mean():.3f}</b> ± {v.std()/np.sqrt(len(v)):.3f} SEM across {len(v)} mice; {(v>0).sum()} positive, {(v<0).sum()} negative mouse means.')
    missing=best[list(DEFS)].isna().sum().to_dict()
    negative={x:int((best[x]<0).sum()) for x in ['rms_cv','gain_cv','gaussian_cv']}
    toc=''.join(f'<li><a href="#{name}">{title}</a></li>' for name,title,_ in SECTIONS)
    sections=''.join(f'<section id="{name}"><h2>{title}</h2><p>{caption}</p><a href="figures/{name}.svg">SVG</a><img loading="lazy" src="figures/{name}.png" alt="{html.escape(title)}"></section>' for name,title,caption in SECTIONS)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PF mean and RMS variability — population report</title>
<style>body{{font:16px system-ui;color:#20303d;max-width:1350px;margin:45px auto;padding:0 30px;line-height:1.6}}h1{{font-size:34px}}h2{{font-size:25px}}p{{max-width:1100px}}img{{width:100%;height:auto}}section{{border-top:1px solid #d5dde1;margin-top:45px;padding-top:16px}}a{{color:#176b89}}.callout{{background:#edf5f4;padding:18px 25px;border-radius:8px}}table{{border-collapse:collapse}}td,th{{padding:6px 12px;border-bottom:1px solid #ddd}}code{{font-size:13px}}</style>
<h1>Does place-field variability follow the mean?</h1><p>Population exploration · saved v11 results · {best.session.nunique()} included / {len(provenance)} stored sessions · {len(mice)} mice · {len(best):,} best-environment ROI observations. Generated 3 November 2026.</p>
<div class="callout"><b>At a glance</b><ul>{''.join(bullets)}</ul><p>{'</p><p>'.join(rel)}</p><p>The shared-trial, held-out, and fully independent constructions answer different questions. Use the independent-trial result as the strongest check that mean–variability coupling generalizes across separate trials.</p></div>
<h2>What is being measured?</h2><p>For each neuron, PF is mean activity across trials at each position. RMS is the square root of the mean squared residual from a PF template at that position. The reported scalar is <b>Pearson r across position between PF and RMS</b>, not mean firing rate, temporal prediction RMS, or spatial reliability itself. A positive r means variability is higher where mean activity is higher; r measures map shape agreement, not proportionality or variance magnitude.</p>
<p>All-trial RMS uses deviations from the all-trial mean: this is a position-dependent population SD (ddof=0). F1 and F2 alternate chronological trials. <b>Held-out</b> averages r(PF F1, RMS on F2 relative to PF F1) and its reverse. The PF enters that residual, so it can contribute template-mismatch effects. <b>Independent</b> averages r(PF F1, within-F2 RMS relative to PF F2) and its reverse; each correlation’s two maps use disjoint trials. The two directional r values are averaged arithmetically, exactly as saved. Both directions must be finite.</p>
<p>Spatial reliability is the stored environment-specific reliability. Amplitude consistency is represented by across-trial CV (SD/mean); lower CV means more consistent amplitude. Trial RMS is the primary amplitude measure, with least-squares and Gaussian PF gain CV as alternatives. These CVs are not correlation coefficients and are not bounded by one.</p>
<h2>Reading population statistics</h2><p>Primary analysis uses each session’s environment with the most trials. Neurons are summarized within session; sessions receive equal weight within mouse; mice receive equal weight across the population. Thus large recordings and mice with many sessions do not dominate. All cells are included unless explicitly labeled. No cross-session cell matching is assumed: counts refer to ROI observations, not unique neurons. Bands/error bars are descriptive between-mouse SEM, not 95% confidence intervals. No neuron-level p-values are used.</p><ol>{toc}</ol>{sections}
<section><h2>Coverage and reproducibility</h2>{coverage.to_html()}<p>{len(provenance)-best.session.nunique()} stored sessions have no usable experience-slot best environment and are excluded from primary figures (see session_coverage.csv). Nonfinite best-environment correlations: {html.escape(str(missing))}. Each metric uses its own finite denominator; constant maps or too few paired positions yield NaN. Negative amplitude CV observations excluded only from amplitude-bin curves: {html.escape(str(negative))}. Curves are omitted where fewer than three mice meet the per-session bin threshold; thin mouse curves can extend beyond the population curve.</p>
<p>Configuration: <code>{html.escape(provenance.analysis_summary.iloc[0])}</code>. Stored computation times span {html.escape(str(provenance.computed_at.min()))} to {html.escape(str(provenance.computed_at.max()))}. The report reads result blobs without recomputing models or changing the store. Source: <code>{html.escape(str(store))}</code>.</p><p><a href="make_report.py">Rebuild script</a> · <a href="provenance.csv">Result provenance</a> · <a href="coverage.csv">Coverage</a> · <a href="session_summaries.csv">Session summaries</a> · <a href="mouse_summaries.csv">Mouse summaries</a> · <a href="session_spearman.csv">Session correlations</a> · <a href="mouse_spearman.csv">Mouse correlations</a></p></section></html>'''
    (OUT/'report.html').write_text(text,encoding='utf-8')
    print(stats.to_string()); print('DONE',OUT/'report.html',flush=True)

if __name__ == '__main__':
    main()


