import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import torch
import faststats as fs
from .indexing import cvFoldSplit
from .wrangling import named_transpose
from .signals import vectorCorrelation

device = "cuda" if torch.cuda.is_available() else "cpu"


def batch_cov(input, centered=True, correction=True):
    """
    Performs batched covariance on input data of shape (batch, dim, samples) or (dim, samples)

    Where the resulting batch covariance matrix has shape (batch, dim, dim) or (dim, dim)
    and bcov[i] = torch.cov(input[i]) if input.ndim==3

    if centered=True (default) will subtract the means first

    if correction=True, will use */(N-1) otherwise will use */N
    """
    if isinstance(input, np.ndarray):
        was_numpy = True
        input = torch.tensor(input)
    else:
        was_numpy = False

    assert (input.ndim == 2) or (input.ndim == 3), "input must be a 2D or 3D tensor"
    assert isinstance(correction, bool), "correction must be a boolean variable"

    # check if batch dimension was provided
    no_batch = input.ndim == 2

    # add an empty batch dimension if not provided
    if no_batch:
        input = input.unsqueeze(0)

    # measure number of samples of each input matrix
    S = input.size(2)

    # subtract mean if doing centered covariance
    if centered:
        input = input - input.mean(dim=2, keepdim=True)

    # measure covariance of each input matrix
    bcov = torch.bmm(input, input.transpose(1, 2))

    # correct for number of samples
    bcov /= S - 1.0 * correction

    # remove empty batch dimension if not provided
    if no_batch:
        bcov = bcov.squeeze(0)

    if was_numpy:
        return np.array(bcov)
    return bcov


def smart_pca(input, centered=True, use_rank=True, correction=True):
    """
    smart algorithm for pca optimized for speed

    input should either have shape (batch, dim, samples) or (dim, samples)
    if dim > samples, will use svd and if samples < dim will use covariance/eigh method

    will center data when centered=True

    if it fails, will fall back on performing sklearns IncrementalPCA whenever forcetry=True
    """
    if isinstance(input, np.ndarray):
        was_numpy = True
        input = torch.tensor(input)
    else:
        was_numpy = False

    assert (input.ndim == 2) or (input.ndim == 3), "input should be a matrix or batched matrices"
    assert isinstance(correction, bool), "correction should be a boolean"

    if input.ndim == 2:
        no_batch = True
        input = input.unsqueeze(0)  # create batch dimension for uniform code
    else:
        no_batch = False

    _, D, S = input.size()
    if D > S:
        # if more dimensions than samples, it's more efficient to run svd
        v, w, _ = named_transpose([torch.linalg.svd(inp) for inp in input])
        # convert singular values to eigenvalues
        w = [ww**2 / (S - 1.0 * correction) for ww in w]
        # append zeros because svd returns w in R**k where k = min(D, S)
        w = [torch.concatenate((ww, torch.zeros(D - S))) for ww in w]

    else:
        # if more samples than dimensions, it's more efficient to run eigh
        bcov = batch_cov(input, centered=centered, correction=correction)
        w, v = named_transpose([eigendecomposition(C, use_rank=use_rank, hermitian=True) for C in bcov])

    # return to stacked tensor across batch dimension
    w = torch.stack(w)
    v = torch.stack(v)

    # if no batch originally provided, squeeze out batch dimension
    if no_batch:
        w = w.squeeze(0)
        v = v.squeeze(0)

    # return eigenvalues and eigenvectors
    if was_numpy:
        return np.array(w), np.array(v)
    return w, v


def eigendecomposition(C, use_rank=True, hermitian=True):
    """
    helper for getting eigenvalues and eigenvectors of covariance matrix

    will measure eigenvalues and eigenvectors with torch.linalg.eigh()
    the output will be sorted from highest to lowest eigenvalue (& eigenvector)

    if use_rank=True, will measure the rank of the covariance matrix and zero
    out any eigenvalues beyond the rank (that are usually nonzero numerical errors)
    """
    if isinstance(C, np.ndarray):
        was_numpy = True
        C = torch.tensor(C)
    else:
        was_numpy = False

    try:
        # measure eigenvalues and eigenvectors
        if hermitian:
            w, v = torch.linalg.eigh(C)
        else:
            w, v = torch.linalg.eig(C)

    except torch._C._LinAlgError as error:
        # this happens if the algorithm failed to converge
        # try with sklearn's incrementalPCA algorithm
        return sklearn_pca(C, use_rank=use_rank)

    except Exception as error:
        # if any other exception, raise it
        raise error

    # sort by eigenvalue from highest to lowest
    w_idx = torch.argsort(-w)
    w = w[w_idx]
    v = v[:, w_idx]

    # iff use_rank=True, will set eigenvalues to 0 for probable numerical errors
    if use_rank:
        crank = torch.linalg.matrix_rank(C)  # measure rank of covariance
        w[crank:] = 0  # set eigenvalues beyond rank to 0

    # return eigenvalues and eigenvectors
    if was_numpy:
        return np.array(w), np.array(v)
    return w, v


def sklearn_pca(input, use_rank=True, rank=None):
    """
    sklearn incrementalPCA algorithm serving as a replacement for eigh when it fails

    input should be a tensor with shape (num_samples, num_features) or it can be a
    covariance matrix with (num_features, num_features)

    if use_rank=True, will set num_components to the rank of input and then fill out the
    rest of the components with random orthogonal components in the null space of the true
    components and set the eigenvalues to 0

    if use_rank=False, will attempt to fit all the components
    if rank is not None, will attempt to fit #=rank components without measuring the rank directly
    (will ignore "rank" if use_rank=False)

    returns w, v where w is eigenvalues and v is eigenvectors sorted from highest to lowest
    """
    if isinstance(input, np.ndarray):
        was_numpy = True
        input = torch.tensor(input)
    else:
        was_numpy = False

    # dimension
    num_samples, num_features = input.shape

    # measure rank (or set to None)
    rank = None if not use_rank else (rank if rank is not None else fast_rank(input))

    # create and fit IncrementalPCA object on input data
    ipca = IncrementalPCA(n_components=rank).fit(input)

    # eigenvectors are the components
    v = ipca.components_

    # eigenvalues are the scaled singular values
    w = ipca.singular_values_**2 / num_samples

    # if v is a subspace of input (e.g. not a full basis, fill it out)
    if v.shape[0] < num_features:
        msg = "adding this because I think it should always be true, and if not I want to find out"
        assert w.shape[0] == v.shape[0], msg
        v_kernel = sp.linalg.null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))

    if was_numpy:
        return w, v

    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T


def fast_rank(input):
    """uses transpose to speed up rank computation, otherwise normal"""
    if isinstance(input, np.ndarray):
        input = torch.tensor(input)
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))


def SVCA(X):
    """
    Shared Variance Component Analysis

    From Stringer et al 2019, https://www.biorxiv.org/content/10.1101/679324v1
    This is adapted almost directly from their github repository:
    https://github.com/MouseLand/stringer-et-al-2019/blob/master/utils.py#L172
    """
    # compute power law
    # SVCA
    # X -= X.mean(axis=1)[:,np.newaxis]

    NN, NT = X.shape

    # split cells into test and train
    norder = np.random.permutation(NN)
    nhalf = int(norder.size / 2)
    ntrain = norder[:nhalf]
    ntest = norder[nhalf:]

    # split time into test and train
    torder = np.random.permutation(NT)
    thalf = int(torder.size / 2)
    ttrain = torder[:thalf]
    ttest = torder[thalf:]
    # if ntrain.size > ttrain.size:
    #    cov = X[np.ix_(ntrain, ttrain)].T @ X[np.ix_(ntest, ttrain)]
    #    u,sv,v = svdecon(cov, k=min(1024, nhalf-1))
    #    u = X[np.ix_(ntrain, ttrain)] @ u
    #    u /= (u**2).sum(axis=0)**0.5
    #    v = X[np.ix_(ntest, ttrain)] @ v
    #    v /= (v**2).sum(axis=0)**0.5
    # else:
    cov = X[np.ix_(ntrain, ttrain)] @ X[np.ix_(ntest, ttrain)].T
    u = PCA(n_components=min(1024, nhalf - 1), svd_solver="randomized").fit_transform(cov)
    u /= (u**2).sum(axis=0) ** 0.5
    v = cov.T @ u
    v /= (v**2).sum(axis=0) ** 0.5

    strain = u.T @ X[np.ix_(ntrain, ttest)]
    stest = v.T @ X[np.ix_(ntest, ttest)]

    # covariance k is uk.T * F * G.T * vk / npts
    scov = (strain * stest).mean(axis=1)
    varcov = (strain**2 + stest**2).mean(axis=1) / 2

    return scov, varcov


def cvPCA_from_MouseLandGithub(X1, X2, nc=None):
    """X is stimuli x neurons"""
    S, N = X1.shape
    assert X2.shape == (S, N), "shape of X1 and X2 is not the same"
    nc = get_num_components(nc, (S, N))
    pca = PCA(n_components=nc).fit(X1.T)
    u = pca.components_.T
    sv = pca.singular_values_

    xproj = X1.T @ (u / sv)
    cproj0 = X1 @ xproj
    cproj1 = X2 @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss


def cvPCA_paper_stimuli(X1, X2, nc=None):
    """X is stimuli x neurons"""
    S, N = X1.shape
    assert X2.shape == (S, N), "shape of X1 and X2 is not the same"
    nc = get_num_components(nc, (S, N))
    pca = PCA(n_components=nc).fit(X1.T)
    u = pca.components_.T

    cproj0 = X1.T @ u
    cproj1 = X2.T @ u
    ss = (cproj0 * cproj1).mean(axis=0)
    return ss


def cvPCA_paper_neurons(X1, X2, nc=None):
    """X is stimuli x neurons"""
    S, N = X1.shape
    assert X2.shape == (S, N), "shape of X1 and X2 is not the same"
    nc = get_num_components(nc, (S, N))
    pca = PCA(n_components=nc).fit(X1)
    u = pca.components_.T

    cproj0 = X1 @ u
    cproj1 = X2 @ u
    ss = (cproj0 * cproj1).mean(axis=0)
    return ss


def get_num_components(nc, shape, maxnc=80):
    return nc if nc is not None else min(maxnc, min(shape))


def shuff_cvPCA(X1, X2, nshuff=5, cvmethod=cvPCA_from_MouseLandGithub):
    """X is stimuli x neurons"""
    S, N = X1.shape
    assert X2.shape == (S, N), "shape of X1 and X2 is not the same"
    nc = get_num_components(None, (S, N))
    ss = np.zeros((nshuff, nc))
    for k in range(nshuff):
        iflip = np.random.rand(S) > 0.5
        X1c = X1.copy()
        X2c = X2.copy()
        X1c[iflip] = X2[iflip]
        X2c[iflip] = X1[iflip]
        ss[k] = cvmethod(X1c, X2c, nc)
    return ss


def cvFOURIER(train, test, basis, covariance=False):
    """
    train/test are neurons x stimuli(position)
    basis is (num_bases x stimuli(position))
    """
    # project onto columns
    train = train @ basis.T
    test = test @ basis.T

    # get real/imaginary components (sine/cosine)
    cos_train = np.real(train)
    sin_train = np.imag(train)
    cos_test = np.real(test)
    sin_test = np.imag(test)

    # get correlation of projections across neurons for train/test
    cos_corr = vectorCorrelation(cos_train, cos_test, axis=0, covariance=covariance)
    sin_corr = vectorCorrelation(sin_train, sin_test, axis=0, covariance=covariance)

    # stack for simple variable handling
    corr = np.stack((cos_corr, sin_corr))

    # return correlation and projections
    return corr, cos_train, sin_train, cos_test, sin_test


def _prepare_cv(spkmap, extra=None, by_trial=False, noise_corr=False, center=True, max_trials=None, max_neurons=None):
    """
    helper for preparing cross-validated spkmap datasets

    spkmap is a (num_rois, num_trials, num_bins) array
    this will turn it into a train/test set with several options

    if extra is not None, will tile extra on second dimension (axis=1) to have same number of trial repeats as spkmap

    if by_trial: will expand across trials (grouping by trial randomly in a single permutation)
    if noise_corr: will only look at noise correlations (subtract mean from data -- requires by_trial)
    if center: will subtract mean after preparing train/test set
    max_trials: will filter by max trials (random trials selected) to normalize number of trials across cvPCA repeats
    max_neurons: will filter by max neurons (random neurons selected) to normalize number of neurons across cvPCA repeats
    """
    # reduce number of neurons if requested
    if max_neurons is not None:
        idx_keep = np.random.permutation(spkmap.shape[0])[:max_neurons]
        spkmap = spkmap[idx_keep]

    # get shape of spkmap
    num_rois, num_trials, num_bins = spkmap.shape

    # generate a train/test set
    train, test = cvFoldSplit(num_trials, 2, even=True)

    # clip trials if max provided
    if max_trials is not None:
        train = train[:max_trials]
        test = test[:max_trials]

    # retrieve and divide by train/test trials
    spk_train = spkmap[:, train]
    spk_test = spkmap[:, test]

    # average across trials or concatenate across trials depending on request
    if not by_trial:
        if noise_corr:
            print("note: noise_corr set to True, but only used when by_trial=True")

        # average across trials
        spk_train = fs.nanmean(spk_train, axis=1)
        spk_test = fs.nanmean(spk_test, axis=1)

    else:
        # concatenate across trials
        num_use_trials = len(train)
        if noise_corr:
            spk_train = spk_train - fs.nanmean(spk_train, axis=1, keepdims=True)
            spk_test = spk_test - fs.nanmean(spk_test, axis=1, keepdims=True)
        spk_train = np.reshape(spk_train, (num_rois, num_use_trials * num_bins))
        spk_test = np.reshape(spk_test, (num_rois, num_use_trials * num_bins))

        if extra is not None:
            extra = np.tile(extra, [1, num_use_trials])

    # center data if requested
    if center:
        spk_train = spk_train - fs.nanmean(spk_train, axis=1, keepdims=True)
        spk_test = spk_test - fs.nanmean(spk_test, axis=1, keepdims=True)

    # return train/test set and extra if included
    if extra is not None:
        return spk_train, spk_test, extra

    # otherwise just train/test set
    return spk_train, spk_test


def cvpca(
    spkmap, by_trial=False, noise_corr=False, center=True, max_trials=None, max_neurons=None, nshuff=3, cvshuff=1, cvmethod=cvPCA_paper_neurons
):
    """
    cvpca method -- run cvPCA on spkmap with various options and repeats of train/test set and cv-shuffling

    nshuff is how many times to repeat the train/test set generation
    cvshuff is how many times to do the specialized cvPCA shuffling method (see shuff_cvPCA method above)
    cvmethod is which method to use to directly calculate the cv-eigenspectrum

    spkmap is a (num_rois, num_trials, num_bins) array

    if by_trial: will expand across trials (grouping by trial randomly in a single permutation)
    if noise_corr: will only look at noise correlations (subtract mean from data -- requires by_trial)
    if center: will subtract mean after preparing train/test set
    max_trials: will filter by max trials (random trials selected) to normalize number of trials across cvPCA repeats
    max_neurons: will filter by max neurons (random neurons selected) to normalize number of neurons across cvPCA repeats

    """

    ss = []
    for _ in range(nshuff):
        # generate train/test set
        spk_train, spk_test = _prepare_cv(
            spkmap, by_trial=by_trial, noise_corr=noise_corr, center=center, max_trials=max_trials, max_neurons=max_neurons
        )

        # do cvPCA (with optional cv shuffling) for this train/test set
        c_ss = shuff_cvPCA(spk_train.T, spk_test.T, nshuff=cvshuff, cvmethod=cvmethod)
        c_ss = np.nanmean(c_ss, axis=0)
        ss.append(c_ss)

    # return average of all shuffles
    return np.nanmean(np.stack(ss), axis=0)


def cv_fourier(
    spkmap, basis, by_trial=False, noise_corr=False, max_trials=None, max_neurons=None, center=True, covariance=False, return_full=False, nshuff=3
):
    """
    cv_fourier method -- run cv_fourier on spkmap with various options and repeats of train/test set and cv-shuffling

    nshuff is how many times to repeat the train/test set generation

    spkmap is a (num_rois, num_trials, num_bins) array

    if by_trial: will expand across trials (grouping by trial randomly in a single permutation)
    if noise_corr: will only look at noise correlations (subtract mean from data -- requires by_trial)
    if center: will subtract mean after preparing train/test set
    if covariance: will measure covariance of projections rather than correlation
    max_trials: will filter by max trials (random trials selected) to normalize number of trials across cvPCA repeats
    max_neurons: will filter by max neurons (random neurons selected) to normalize number of neurons across cvPCA repeats

    returns
    -------
    the average correlation (2 x num frequencies) across shuffles
    if return_full=True, also returns the projections onto train/test for cosine/sine bases

    """
    # if this isn't true, generate an immediate error
    assert basis.shape[1] == spkmap.shape[2], "shape of basis doesn't match number of spatial bins in spkmap"

    # find any positions with nan across train/test sets
    idx_not_nan = np.any(np.isnan(spkmap), axis=(0, 1))

    # remove from spkmap and basis (basis has gaps where appropriate rather than discontinuous stitched sine/cosines)
    spkmap = spkmap[:, :, ~idx_not_nan]
    basis = basis[:, ~idx_not_nan]

    corr = []
    if return_full:
        cos_train, sin_train, cos_test, sin_test = [], [], [], []

    for _ in range(nshuff):
        # generate train/test set
        spk_train, spk_test, c_basis = _prepare_cv(
            spkmap, extra=basis, by_trial=by_trial, noise_corr=noise_corr, center=center, max_trials=max_trials, max_neurons=max_neurons
        )

        # do cvFOURIER method
        c_corr, c_cos_train, c_sin_train, c_cos_test, c_sin_test = cvFOURIER(spk_train, spk_test, c_basis, covariance=covariance)

        # add correlation of this shuffle to result
        corr.append(c_corr)

        if return_full:
            cos_train.append(c_cos_train)
            cos_test.append(c_cos_test)
            sin_train.append(c_sin_train)
            sin_test.append(c_sin_test)

    # return average of all shuffles
    corr = np.nanmean(np.stack(corr), axis=0)

    # return all if requested
    if return_full:
        return corr, np.stack(cos_train), np.stack(cos_test), np.stack(sin_train), np.stack(sin_test)

    # otherwise just return correlation averages
    return corr
