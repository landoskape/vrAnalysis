"""S1-style estimator audit on one reproducibly random real session.

Read-only with respect to stored analyses. Outputs a plot, numerical spectra,
and a short report in artifacts/s1_spectrum_validation/.
"""
from itertools import combinations, permutations, product
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vrAnalysis.database import get_database
from vrAnalysis.helpers import cross_validate_trials, edge2center
from vrAnalysis.processors.placefields import get_placefield
from vrAnalysis.processors.support import convolve_toeplitz, get_gauss_kernel
from dimensionality_manuscript import ResultsAggregator, ResultsStore, TilburyFitConfig, get_data_config
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.configs.cvpca import CVPCAConfig
from dimensionality_manuscript.configs.stimspace import _cvpca_svd, _cvsvd, _to_g
from dimensionality_manuscript.configs.tilbury_fit import _eval_tilbury
from dimensionality_manuscript.figure_scripts.figure4.placefield_fits import _load_fit_curves
from dimilibi import gaussian_filter
from dimilibi.pca import PCA
from dimilibi.cvpca import CVPCA
from threadpoolctl import threadpool_limits

SEED = 731
OUT = Path(__file__).resolve().parent / "s1_spectrum_validation"


def pca(matrix, spatial):
    return PCA(center=True).fit(matrix.T if spatial else matrix).get_eigenvalues().numpy()


def stim_smooth(matrix, positions, full_positions, width):
    # The production placefield smoother normalizes the kernel at boundaries;
    # CVPCAConfig's fixed smoother does not. Preserve both implementations.
    if not width:
        return matrix
    dense=np.zeros((matrix.shape[0],len(full_positions)))
    indices=np.array([np.argmin(np.abs(full_positions-x)) for x in positions])
    dense[:,indices]=matrix.numpy()
    valid=np.zeros((1,len(full_positions)))
    valid[:,indices]=1
    kernel=get_gauss_kernel(full_positions,width)
    sm=convolve_toeplitz(dense,kernel,axis=1)
    denom=convolve_toeplitz(valid,kernel,axis=1)
    return torch.tensor((sm/denom)[:,indices],dtype=matrix.dtype)


def estimate(repeats, positions, full_positions, width, native=False):
    """Use the actual three estimator implementations, preserving rank order/signs."""
    smooth = [stim_smooth(x, positions, full_positions, width) for x in repeats]
    gs, gr = [_to_g(x) for x in smooth], [_to_g(x) for x in repeats]
    cfg = CVPCAConfig(fixed_smooth_width=width, use_spatial_eigenvectors=True)
    if native:
        # Production StimSpace averages the four ordered combinations of four folds.
        stim_triples = list(combinations(range(4), 3))
        reg_triples = [(i, (i+1) % 3, (i+2) % 3) for i in range(3)]
    else:
        # Like the paper: six repeats, every reference and disjoint scoring pair.
        stim_triples = [(i,j,k) for i in range(len(repeats))
                        for j,k in combinations([x for x in range(len(repeats)) if x != i], 2)]
        reg_triples = stim_triples
    cv = np.mean([_cvpca_svd(smooth[i], repeats[j], repeats[k]).numpy()
                  for i,j,k in stim_triples], axis=0)
    ss = np.mean([_cvsvd(gs[i], gr[j], gr[k]).numpy()
                  for i,j,k in stim_triples], axis=0)
    models = [cfg.fit_fixed_model(x, positions, full_positions) if width
              else CVPCA(center=True,on_stimuli=True).fit(x)
              for x in repeats]
    reg = np.mean([models[i].score(repeats[j], repeats[k]).numpy()
                   for i,j,k in reg_triples], axis=0)
    return dict(ss_cvpca=cv, ss_cv=ss, reg_covariances_fixed=reg)


def independent_sign_expectation(signal, noise, positions, full_positions, width):
    """Exact score-sign average; keeps empirical noise amplitudes/spatial structure.

    With independent +/- noise signs on the three inputs, covariance scores
    average to the signal scored in the noisy training basis. ss_cv reuses its
    projection input in basis fitting AND scoring, so its projection noise remains.
    """
    cfg = CVPCAConfig(fixed_smooth_width=width)
    rows = [signal + sign * e for e in noise for sign in (-1,1)]
    smooth = [stim_smooth(x, positions, full_positions, width) for x in rows]
    cv = np.mean([_cvpca_svd(x, signal, signal).numpy() for x in smooth], axis=0)
    reg = np.mean([cfg.fit_fixed_model(x, positions, full_positions).score(signal, signal).numpy()
                   for x in rows], axis=0)
    ss = []
    for i,k in permutations(range(len(noise)),2):
        for a,b in product(range(2),repeat=2):
            ss.append(_cvsvd(_to_g(smooth[2*i+a]), _to_g(signal), _to_g(rows[2*k+b])).numpy())
    return dict(ss_cvpca=cv, ss_cv=np.mean(ss,axis=0), reg_covariances_fixed=reg)


def write_report(info):
    m=info["metrics"]
    rows=[]
    for key,values in m.items():
        rows.append(f"| {key} | {values['truth_sum']:.5f} | {values['clean_sum_ratio']:.6f} | {values['s1_sum_ratio']:.3f} | {values['balanced_sum_ratio']:.3f} | {values['balanced_rank10_50_signed_sum_ratio']:.3f} |")
    report=f"""# S1-style spectrum check

Session: `{info['session']}`, chosen with seed {info['seed']};
{info['n_neurons']} successfully fitted neurons, {info['n_positions']} spatial bins,
best-sampled environment {info['environment']}.

## Construction

- Known signal: the stored unregularized generalized/Tilbury tuning curves.
- Signal and empirical responses converted to full-recording frame-SD units.
- Six disjoint trial-group means from the same neurons/environment/positions.
- Empirical residual E_i = measured repeat i - mean of the six measured repeats.
- Surrogate repeat i = known signal + E_i, following the paper's Figure S1 recipe.
- Evaluate all 60 choices of reference repeat and unordered disjoint scoring pair.
  This uses production estimator functions but more fold assignments than the
  default four-fold StimSpace / three-fold CVPCA pipelines. It does not reproduce
  an existing stored session spectrum.
- Basis smoothing: 5 cm, using the normalized placefield smoother for StimSpace
  and CVPCAConfig.fit_fixed_model for reg_covariances_fixed. Scoring inputs are
  unsmoothed. No sum normalization, log smoothing, or rank re-sorting is applied.

## Ground truth and scaling

ss_cvpca and reg_covariances_fixed center across neurons and use SVD^2/(N-1).
ss_cv centers each neuron across positions and uses SVD^2/(P-1).
The dimilibi PCA class uses SVD^2/N or SVD^2/P respectively. Reference PCA is
converted to sample-covariance units before the comparison.
For this session the corrections are N/(N-1) = {info['n_neurons']/(info['n_neurons']-1):.8f}
and P/(P-1) = {info['n_positions']/(info['n_positions']-1):.8f}.
Centering axes differ, so there is no universal scalar conversion between the
spatial and neural covariance spectra. The two ground-truth totals below differ
for that reason as well as the different denominators.

## Measurements

All ratios use each method's matching known-signal PCA target.
The final column sums ranks 10 through 50 inclusive, retaining negative estimates.

| Estimator | Target total | Clean/no smoothing total ratio | Literal S1 total ratio | Independent sign control total ratio | Independent sign control ranks 10-50 ratio |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

The clean, unsmoothed spectra agree elementwise with their targets to relative
error below 1e-8 (actual maxima are recorded in metrics.json). This rules out a
large multiplicative normalization error for these inputs.

## Why the literal S1 result is low

Residuals estimated by subtracting a finite repeat mean are dependent: they sum
to zero. Their average cross-repeat covariance is negative. In the spatial
covariance methods the exact resulting total-variance bias is minus the mean
residual covariance trace divided by R-1, here -{info['expected_residual_trace_bias']:.5f}.
This accounts for their observed roughly 20% deficit; it is not a scaling factor
inside cvPCA.

The independent sign control gives each input's empirical residual an independent
random +/- sign, preserving its spatial/cross-neuron covariance while making
cross-repeat noise covariance zero. The plotted result is the exact average over
scoring signs (computed analytically), averaged over training signs and repeat
assignments. It is not one additional noisy realization.

ss_cvpca and reg_covariances_fixed then recover the total signal variance exactly,
though estimated bases redistribute variance between ranks and distort the very
small tail. ss_cv still loses about 3.5% of the total and about 24% in ranks 10-50.
Its learned left/right cross-SVD bases differ from a PCA basis; its projection
input is also reused in basis fitting and scoring. This check establishes an
estimator/basis effect, not a unique causal attribution or a universal correction.

## Reproduction and scope

Run `python -u -m artifacts.s1_spectrum_validation` in the vrAnalysis Conda
environment. No stored fits or production analysis code are changed.

Outputs: s1_comparison.png, metrics.json, spectra.npz, prepared.npz.
This is a one-session diagnostic, not a validation across all sessions.
"""
    (OUT/"report.md").write_text(report,encoding="utf-8")


def main():
    start=time.perf_counter()
    torch.set_num_threads(4)
    threadpool_limits(limits=4)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    sessions=list(get_database("vrSessions").iter_sessions(imaging=True,experimentType="Blender VR"))
    results=ResultsAggregator(TilburyFitConfig(),ResultsStore(),sessions)
    sid=str(np.random.default_rng(SEED).permutation(results.session_ids)[0])
    print("Random session:",sid,flush=True)
    registry=PopulationRegistry(registry_params=get_data_config("even").to_registry_params(),autosave=False)
    fit=_load_fit_curves(results,registry,sid,{"activity_parameters_name":"default"},target_full_session=True)
    valid=np.logical_and.reduce([np.isfinite(fit[k]).all(axis=1) for k in ("params","params_control","params_shrinkage")])
    session=fit["session"]
    population,behavior=registry.get_population(session,"sigrebase")
    original_rois=fit["idx_neurons"][fit["idx_keep"]][valid]
    signal=np.array([_eval_tilbury(fit["theta"],p) for p in fit["params"][valid]]) * fit["fluorescence_scale"][valid,None]
    print("Preparing empirical repeats:",len(original_rois),"neurons",flush=True)
    full=population.data[original_rois]
    scaled=population.apply_split(full,time_idx=None,prefiltered=False,scale=True,scale_type="std",pre_split=True)
    data=np.asarray(scaled[:,population.idx_samples]).T
    del full,scaled
    edges=np.linspace(0,session.env_length[0],TilburyFitConfig().num_bins+1)
    centers=edge2center(edges)
    trial_groups=cross_validate_trials(session.trial_environment,[1]*6)
    maps=[get_placefield(data,behavior,edges,trial_filter=group,average=True,zero_to_nan=True,
                        use_fast_sampling=True,session=session).filter_by_environment(fit["best_env"])
          for group in trial_groups]
    indices=np.array([np.argmin(np.abs(centers-x)) for x in fit["theta"]])
    assert np.allclose(centers[indices],fit["theta"])
    observed=np.array([m.placefield[0].T[:,indices] for m in maps])
    counts=np.array([m.count[0][indices] for m in maps])
    good=np.all(counts>0,axis=0) & np.all(np.isfinite(observed),axis=(0,1))
    positions=fit["theta"][good]
    observed=observed[:,:,good]
    signal=torch.tensor(signal[:,good],dtype=torch.float64)
    assert signal.shape[0]>signal.shape[1]>2
    noise=torch.tensor(observed-observed.mean(axis=0,keepdims=True),dtype=torch.float64)
    np.savez(OUT/"prepared.npz",signal=signal.numpy(),noise=noise.numpy(),positions=positions,
             centers=centers,session=sid,environment=int(fit["best_env"]))
    surrogate=[signal+e for e in noise]
    print("Signal shape:",tuple(signal.shape),"env:",fit["best_env"],flush=True)
    spatial_pca=pca(signal,True)
    neuron_pca=pca(signal,False)
    n,p=signal.shape
    truth={"ss_cvpca":spatial_pca*n/(n-1),"reg_covariances_fixed":spatial_pca*n/(n-1),
           "ss_cv":neuron_pca*p/(p-1)}
    print("Noise-free / no smoothing control",flush=True)
    clean=estimate([signal]*6,positions,centers,0)
    print("S1-style empirical residual experiment",flush=True)
    s1=estimate(surrogate,positions,centers,5.)
    print("Independent sign-balanced control",flush=True)
    balanced=independent_sign_expectation(signal,noise,positions,centers,5.)
    clean_smooth=estimate([signal]*6,positions,centers,5.)
    noisy_pca={key:np.mean([pca(x,key!="ss_cv") for x in surrogate],axis=0) *
                   (p/(p-1) if key=="ss_cv" else n/(n-1)) for key in truth}
    metrics={}
    for key,target in truth.items():
        sl=slice(9,min(50,len(target)))
        strong=target>target[0]*1e-10
        metrics[key]={
            "truth_sum":float(target.sum()),
            "pca_library_sum":float((neuron_pca if key=="ss_cv" else spatial_pca).sum()),
            "clean_sum_ratio":float(clean[key].sum()/target.sum()),
            "clean_max_relative_error":float(np.max(np.abs(clean[key][strong]-target[strong])/target[strong])),
            "clean_smoothed_sum_ratio":float(clean_smooth[key].sum()/target.sum()),
            "s1_sum_ratio":float(s1[key].sum()/target.sum()),
            "balanced_sum_ratio":float(balanced[key].sum()/target.sum()),
            "s1_rank10_50_signed_sum_ratio":float(s1[key][sl].sum()/target[sl].sum()),
            "balanced_rank10_50_signed_sum_ratio":float(balanced[key][sl].sum()/target[sl].sum()),
            "s1_negative_count":int((s1[key]<0).sum()),
        }
        print(key,json.dumps(metrics[key]),flush=True)
    centered_noise = noise - noise.mean(dim=1, keepdim=True)
    expected_residual_trace_bias = float(centered_noise.square().sum(dim=(1,2)).mean() / (n-1) / (len(noise)-1))
    for key in truth:
        assert metrics[key]["clean_max_relative_error"] < 1e-8
        assert abs(metrics[key]["clean_sum_ratio"]-1) < 1e-10
    payload=dict(expected_residual_trace_bias=expected_residual_trace_bias, session=sid,seed=SEED,n_neurons=n,n_positions=p,environment=int(fit["best_env"]),
                 units="full-recording frame-SD units; no spectrum sum normalization",
                 residual_noise_rms=float(noise.square().mean().sqrt()),
                 signal_rms=float(signal.square().mean().sqrt()),metrics=metrics)
    (OUT/"metrics.json").write_text(json.dumps(payload,indent=2))
    write_report(payload)

    np.savez(OUT/"spectra.npz",signal=signal.numpy(),noise=noise.numpy(),positions=positions,
             **{f"{kind}_{key}":value for kind,group in (("truth",truth),("clean",clean),("clean_smoothed",clean_smooth),
                  ("s1",s1),("balanced",balanced),("noisy_pca",noisy_pca)) for key,value in group.items()})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,key in zip(axes,truth):
        rank=np.arange(1,len(truth[key])+1)
        ax.loglog(rank,truth[key],color="black",lw=2,label="Known-signal PCA (sample covariance)")
        ax.loglog(rank,noisy_pca[key],color="0.7",lw=1,label="Ordinary PCA after adding noise")
        for group,color,label in ((s1,"#31688e","S1: empirical residuals"),(balanced,"#c44e52","Independent sign-balanced control")):
            vals=group[key]
            ax.loglog(rank,np.where(vals>0,vals,np.nan),".-",color=color,ms=3,lw=1,label=label)
            neg=vals<0
            ax.loglog(rank[neg],-vals[neg],"x",color=color,ms=4)
        ax.set(title=key,xlabel="Component",ylabel="Variance (absolute units)",ylim=(1e-8,max(truth[key][0],noisy_pca[key][0])*2))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7,loc="lower left")
    fig.suptitle(f"S1 estimator check: {sid} | N={n}, P={p} | 5 cm basis smoothing; crosses = negative estimates",fontsize=10)
    fig.savefig(OUT/"s1_comparison.png",dpi=180)
    plt.close(fig)
    print("Saved",OUT,"elapsed",round(time.perf_counter()-start,1),"s",flush=True)


if __name__=="__main__":
    main()
