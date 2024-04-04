import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import torch
from .indexing import cvFoldSplit
from .wrangling import named_transpose

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


def cvpca(spkmap, by_trial=False, noise_corr=False, max_trials=None, max_neurons=None, nshuff=3, cvmethods=[cvPCA_paper_neurons]):
    # reduce number of neurons if requested
    if max_neurons is not None:
        idx_keep = np.random.permutation(spkmap.shape[0])[:max_neurons]
        spkmap = spkmap[idx_keep]

    # get shape of spkmap and define "train" vs "test" trials
    num_rois, num_trials, num_bins = spkmap.shape
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
        spk_train = np.mean(spk_train, axis=1)
        spk_test = np.mean(spk_test, axis=1)

    else:
        # concatenate across trials
        num_use_trials = len(train)
        if noise_corr:
            spk_train = spk_train - np.mean(spk_train, axis=1, keepdims=True)
            spk_test = spk_test - np.mean(spk_test, axis=1, keepdims=True)
        spk_train = np.reshape(spk_train, (num_rois, num_use_trials * num_bins))
        spk_test = np.reshape(spk_test, (num_rois, num_use_trials * num_bins))

    # center data
    spk_train = spk_train - np.mean(spk_train, axis=1, keepdims=True)
    spk_test = spk_test - np.mean(spk_test, axis=1, keepdims=True)

    # inherited from stringer/pachitariu
    ss = [shuff_cvPCA(spk_train.T, spk_test.T, nshuff=nshuff, cvmethod=cvm) for cvm in cvmethods]
    ss = [np.nanmean(s, axis=0) for s in ss]

    return ss
