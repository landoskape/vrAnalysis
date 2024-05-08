import time
import random
from copy import copy
from tqdm import tqdm
from pathlib import Path
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import faststats as fs

from vrAnalysis import helpers
from vrAnalysis import fileManagement as fm


def get_variance_estimates(samples):
    """
    Solve b = Ax where b is measured and x is unknowns
    b represents variance of each variable (indexed by "n")
    for the across stim samples, then for each stim independently
    x represents the noise variance then the stim variance
    """

    def _make_full_design(S):
        indices = np.arange(S)
        subsets = [*[[s] for s in range(S)], [s for s in range(S)]]
        num_subsets = len(subsets)
        dm = np.zeros((num_subsets, S + 1))
        dm[:, -1] = 1.0
        for ii, subset in enumerate(subsets):
            dm[ii, subset] = 1 / len(subset)
        return dm, subsets

    def _make_covs(samples, subsets):
        # Make covariance matrices using all combinations of subsets over samples
        return [np.cov(np.concatenate([samples[s] for s in subset], axis=1)) for subset in tqdm(subsets)]

    def _make_target(covs, n):
        # Make b in b = Ax (see above)
        return np.array([cov[n, n] for cov in covs])

    S = len(samples)
    N = samples[0].shape[0]
    for sample in samples:
        assert N == sample.shape[0], "samples should all have same number of variables"

    # make design matrix and subset combinations
    dm, subsets = _make_full_design(S)

    # make covariance matrices
    covs = _make_covs(samples, subsets)
    # make targets
    targets = [_make_target(covs, n) for n in tqdm(range(N))]
    # solve for each variable
    solution = [
        sp.optimize.lsq_linear(dm, target, bounds=(0, np.inf)).x for target in tqdm(targets)
    ]  # get covariance across all and single subset covariances
    # get estimate of noise covariance
    estimate_noise = covs[-1]
    # get each estimate of corrected stim covariance
    estimate_stim_noise = [covs[s] - estimate_noise for s in range(S)]

    # fill in main diagonal with solutions to least squares problem
    for n in tqdm(range(N)):
        estimate_noise[n, n] = solution[n][-1]
        for s in range(S):
            estimate_stim_noise[s][n, n] = solution[n][s]

    # return results
    return estimate_stim_noise, estimate_noise


def stringer2019():
    fpath = Path(r"C:\Users\Andrew\Documents\literatureData\stringerPachitariu2021")
    files = [
        Path("gratings_drifting_GT1_2019_04_12_1.npy"),
        Path("gratings_static_GT1_2019_04_17_1.npy"),
    ]
    dataset = [np.load(fpath / f, allow_pickle=True).item() for f in files]

    istim = dataset[idata]["istim"]
    isin = np.sin(istim)
    icos = np.cos(istim)
    data = dataset[idata]["sresp"]
    ndata = sp.stats.zscore(data, axis=1)

    orientation = True
    multiplier = 2 if orientation else 1
    modulo = 2 * np.pi

    num_bins = 5
    bins = np.linspace(0, modulo, num_bins + 1)
    centers = helpers.edge2center(bins)

    stim = np.mod(istim * multiplier, modulo)
    idx = np.digitize(stim, bins) - 1

    # Hack to get preferred stimulus
    istim_argsort = np.argsort(stim)
    ndata_sorted = ndata[:, istim_argsort][:, :-2]
    ndata_ds_mean = np.mean(np.reshape(ndata_sorted, (ndata.shape[0], 20, 214)), axis=2)
    ndata_ds_argmax = np.argmax(ndata_ds_mean, axis=1)

    plt.imshow(ndata_sorted[np.argsort(ndata_ds_argmax)], aspect="auto", vmin=-1, vmax=1, cmap="bwr")
    plt.show()

    # created randomly subsampled data for measuring noise covariance
    nidx = np.random.permutation(ndata.shape[0])[:7000]

    data_by_stim = [ndata[nidx][:, idx == i] for i in range(num_bins)]  # data by stimulus
    cdata_by_stim = [dbs - dbs.mean(axis=1, keepdims=True) for dbs in data_by_stim]  # centered data by stimulus

    # get estimates of covariance for responses to orientations
    estimate_stim_noise, estimate_noise = get_variance_estimates(cdata_by_stim)

    # show results of noise covariance estimation
    isort = np.argsort(ndata_ds_argmax[nidx])
    figdim = 2
    fig, ax = plt.subplots(1, num_bins + 1, figsize=((num_bins + 1) * figdim, figdim), layout="constrained")
    for b in range(num_bins):
        ax[b].imshow(estimate_stim_noise[b][isort][:, isort], aspect="auto", vmin=-0.1, vmax=0.1, cmap="bwr")
    ax[-1].imshow(estimate_noise[isort][:, isort], aspect="auto", vmin=-0.1, vmax=0.1, cmap="bwr")
    plt.show()


def simulated():
    # good code for simulating and estimating covariance matrices here:
    def create_cov(N, lam=1.05):
        # Create a random covariance matrix
        A = np.random.randn(N, N)
        Q, R = np.linalg.qr(A)
        V = Q @ np.diag(np.sign(np.diag(R)))
        D = lam ** np.arange(1, -N + 1, -1)
        cov = V @ np.diag(D) @ V.T
        return cov, V, D

    N = 1000
    S = 20
    R = 3000

    lam = 1.1

    # create covariance matrices
    sigma_stim = [create_cov(N, lam=lam)[0] for _ in range(S)]
    sigma_noise = create_cov(N, lam=lam)[0]

    # create data from covariance matrices
    samples_stim = [np.random.multivariate_normal(np.zeros(N), ss, R).T for ss in sigma_stim]
    samples_noise = [np.random.multivariate_normal(np.zeros(N), sigma_noise, R).T for _ in range(S)]
    samples = [ss + sn for ss, sn in zip(samples_stim, samples_noise)]

    # Get estimates of the noise matrices
    print("starting estimates...")
    estimate_stim_noise, estimate_noise = get_variance_estimates(samples)

    # Simple estimate
    cov_noise = np.cov(np.concatenate(samples, axis=1))
    cov_stim = [np.cov(sample) - cov_noise for sample in samples]

    
    # Print results (for a random stimulus)
    s = 0
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    ax[0].scatter(estimate_noise.flatten(), sigma_noise.flatten(), c="k", alpha=0.1)
    ax[0].axline((0, 0), slope=1)

    ax[1].scatter(estimate_stim_noise[s].flatten(), sigma_stim[s].flatten(), c="k", alpha=0.1)
    ax[1].axline((0, 0), slope=1)

    error_noise = estimate_noise - sigma_noise
    estimate_sim = estimate_stim_noise[s] - sigma_stim[s]
    ax[2].scatter(sigma_noise.flatten(), error_noise.flatten(), c="k", alpha=0.1)
    ax[2].scatter(sigma_stim[s].flatten(), estimate_sim.flatten(), c="r", alpha=0.1)
    plt.show()

if __name__ == "__main__":
    print("just storing code, nothing in main function yet")
