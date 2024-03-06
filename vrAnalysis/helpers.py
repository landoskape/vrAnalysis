import sys
from warnings import warn
import inspect
import math
from copy import copy
from itertools import chain, combinations
import numpy as np
import scipy as sp
from sklearn.decomposition import IncrementalPCA
import torch
import matplotlib
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------- plotting helpers ----------------------------------
def scale(data, vmin=0, vmax=1, prctile=(0, 100)):
    """scale data to arbitrary range using conservative percentile estimate"""
    xmin = np.percentile(data, prctile[0])
    xmax = np.percentile(data, prctile[1])
    sdata = (data - xmin) / (xmax - xmin)
    sdata *= vmax - vmin
    sdata += vmin
    return sdata


def errorPlot(x, data, axis=-1, se=False, ax=None, **kwargs):
    """
    convenience method for making a plot with errorbars
    kwargs go into fill_between and plot, so they have to work for both...
    to make that more flexible, we could add a list of kwargs that work for
    one but not the other and pop them out as I did with the 'label'...
    """
    if ax is None:
        ax = plt.gca()
    meanData = np.mean(data, axis=axis)
    correction = data.shape[axis] if se else 1
    errorData = np.std(data, axis=axis) / correction
    fillBetweenArgs = kwargs.copy()
    fillBetweenArgs.pop("label")
    ax.fill_between(x, meanData + errorData, meanData - errorData, **fillBetweenArgs)
    kwargs.pop("alpha")
    ax.plot(x, meanData, **kwargs)


def discreteColormap(name="Spectral", N=10):
    return matplotlib.colormaps[name].resampled(N)


def ncmap(name="Spectral", vmin=10, vmax=None):
    if vmax is None:
        vmax = vmin - 1
        vmin = 0

    cmap = matplotlib.cm.get_cmap(name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    def getcolor(val):
        return cmap(norm(val))

    return getcolor


def edge2center(edges):
    assert isinstance(edges, np.ndarray) and edges.ndim == 1, "edges must be a 1-d numpy array"
    return edges[:-1] + np.diff(edges) / 2


# ------------------------------------ data wrangling ------------------------------------
def fractional_histogram(*args, **kwargs):
    """wrapper of np.histogram() with relative counts instead of total or density"""
    counts, bins = np.histogram(*args, **kwargs)
    counts = counts / np.sum(counts)
    return counts, bins


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))


# ------------------------------------ index handling ------------------------------------
def all_pairs(idx_ses):
    """Return all pairs without replacement of elements in iterable idx_ses"""
    return np.array([np.array(pair) for pair in combinations(idx_ses, 2)], dtype=int)


def index_on_dim(numpy_array, index, dim):
    """Return data from **numpy_array** from indices **index** on dimension **dim**"""
    slices = [slice(None)] * numpy_array.ndim
    slices[dim] = index
    return numpy_array[tuple(slices)]


def argsort(seq):
    """Native python for getting index of sort of a sequence"""
    return sorted(range(len(seq)), key=seq.__getitem__)


def index_in_order(seq):
    """Native python for getting argsort index but in order"""
    sorted_indices = argsort(seq)
    order = [0] * len(sorted_indices)
    for i, j in enumerate(sorted_indices):
        order[j] = i
    return order


def powerset(iterable, ignore_empty=False):
    """
    return chain of subsets in powerset of an iterable
    (https://docs.python.org/3/library/itertools.html#itertools-recipes)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1 if ignore_empty else 0, len(s) + 1))


def index_in_target(value, target):
    """returns boolean array for whether each value is in target and location array such that target[loc_target] = value"""
    value = check_iterable(value)
    target_to_index = {value: index for index, value in enumerate(target)}
    in_target = np.array([val in target_to_index for val in value], dtype=bool)
    loc_target = np.array([target_to_index[val] if in_t else -1 for in_t, val in zip(in_target, value)], dtype=int)
    return in_target, loc_target


def cvFoldSplit(samples, numFold, even=False):
    if type(samples) == int:
        numSamples = copy(samples)
        samples = np.arange(samples)
    else:
        numSamples = len(samples)
    # generates list of indices of equally sized randomly selected samples to be used in numFold-crossvalidation for a given number of samples
    minimumSamples = int(np.floor(numSamples / numFold))
    remainder = numSamples - numFold * minimumSamples
    samplesPerFold = [
        int(minimumSamples + 1 * (f < remainder)) for f in range(numFold)
    ]  # each fold gets minimum number of samples, assign remainder evenly to as many as necessary
    sampleIdxPerFold = [
        0,
        *np.cumsum(samplesPerFold),
    ]  # defines where to start and stop for each fold
    randomOrder = samples[np.random.permutation(numSamples)]  # random permutation of samples
    foldIdx = [randomOrder[sampleIdxPerFold[i] : sampleIdxPerFold[i + 1]] for i in range(numFold)]  # assign samples to each cross-validation fold
    if even:
        foldIdx = [fi[:minimumSamples] for fi in foldIdx]
    return foldIdx


def nearestpoint(x, y, mode="nearest"):
    # fast implementation of nearest neighbor index between two arrays
    # based on file-exchange from https://uk.mathworks.com/matlabcentral/fileexchange/8939-nearestpoint-x-y-m
    # returns index of y closest to each point in x and distance between points
    # if mode set to previous or next, will exact match or next closest in specified direction
    # if mode set to nearest and x[i] exactly in between two y's, will take next y rather than previous
    assert mode == "nearest" or mode == "previous" or mode == "next", "mode must be one of the following strings: ('nearest', 'previous', 'next')"
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.ndim == 1 and y.ndim == 1, "arrays must be one-dimensional numpy arrays"

    # If either x or y is empty, return empty arrays
    if len(x) == 0 or len(y) == 0:
        return np.array(()), np.array(())

    # Otherwise, find nearest neighbor, return indices and distance
    xi = np.argsort(x)
    xs = x[xi]  # sorted x
    xi = np.argsort(xi)  # return to x order
    nx = len(x)
    cx = np.zeros(nx)  # used to identify which are x's in concatenate
    qx = np.isnan(xs)  # index of nans

    yi = np.argsort(y)
    ys = y[yi]
    ny = len(y)
    cy = np.ones(ny)

    # Created concatenate value vector and identity vector
    xy = np.concatenate((xs, ys))
    cxy = np.concatenate((cx, cy))

    # Sort values and identities
    xyi = np.argsort(xy)
    cxy = cxy[xyi]  # cxy[i] = 0 -> xy[i] belongs to x, = 1 -> xy[i] belongs to y
    ii = np.cumsum(cxy)  # what index of y is equal to or least greater than each x (index of "next" nearest neighbor)
    ii = ii[cxy == 0]  # only keep indices for each x value

    # Note: if m==previous or next, then will use previous or next closest even if exact match exists! (maybe include a conditional for that?)
    if mode == "previous":
        equalOrPrevious = np.array(
            [yidx if (yidx < ny and xs[xidx] - ys[yidx.astype(int)] == 0) else yidx - 1 for (xidx, yidx) in enumerate(ii)]
        )  # preserve index if equal, otherwise use previous from y
        ind = np.where(equalOrPrevious >= 0, equalOrPrevious, np.nan)  # Preserve index if within y, otherwise nan
    elif mode == "next":
        ind = np.where(ii < ny, ii, np.nan)  # Preserve index if within y, otherwise nan
    else:
        ii = np.stack((ii, ii - 1))
        ii[ii >= ny] = ny - 1
        ii[ii < 0] = 0
        yy = ys[ii.astype(np.uint64)]
        dy = np.argmin(np.abs(xs - yy), axis=0)
        ind = ii[dy, np.arange(nx)].astype(np.float64)

    ind[qx] = np.nan  # reset nan indices back to nan
    ind = ind[xi]  # resort to x
    ind = np.array([yi[idx.astype(np.uint64)] if not np.isnan(idx) else np.nan for idx in ind])  # resort to y

    # compute distance between nearest points
    ynearest = np.array([y[idx.astype(np.uint64)] if not np.isnan(idx) else np.nan for idx in ind])
    d = np.abs(x - ynearest)

    return ind, d


def digitizeEqual(data, mn, mx, nbins):
    # digitizeEqual returns the bin of each element from data within each equally spaced bin between mn,mx
    # (it's like np.digitize but faster and only works with equal bins)
    binidx = (data - mn) / (mx - mn) * float(nbins)
    binidx[binidx < 0] = 0
    binidx[binidx > nbins - 1] = nbins - 1
    return binidx.astype(int)


# ------------------------------------ sparse handling -----------------------------------
def sparse_equal(a, b):
    """
    helper method for checking if two sparse arrays a & b are equal
    """
    # if they are equal, the number of disagreeing values will be 0
    return (a != b).nnz == 0


def sparse_filter_by_idx(csr, idx):
    """
    helper method for symmetric filtering of a sparse array in csr format

    will perform the equivalent of csr.toarray()[idx][:, idx] using efficient
    sparse indexing (the second slice done after converting to csc)

    ::note, tocsc() and tocsr() actually take longer than just doing it directly
    (even though the row indexing is faster than column indexing for a csr and
    vice-versa...)

    returns a sliced csr array
    """
    return csr[idx][:, idx]


# ---------------------------------- signal processing ----------------------------------
def pairdist(XA, XB):
    """
    measure euclidean distance between set of points in XA and XB

    XA and XB are (M x N) arrays of M observations in N dimensions
    returns: (M, ) array of distance between each matched pair in XA, XB
    """
    difference = XA - XB
    return np.sqrt(np.sum(difference**2, axis=1))


def crossCorrelation(x, y):
    """
    measure the cross correlation between each column in x with every column in y

    sets the cross-correlation to NaN for any element if it has 0 variation
    """
    assert x.ndim == y.ndim == 2, "x and y must be 2-d numpy arrays"
    assert x.shape[0] == y.shape[0], "x and y need to have the same number of dimensions (=rows)!"
    N = x.shape[0]
    xDev = x - np.mean(x, axis=0, keepdims=True)
    yDev = y - np.mean(y, axis=0, keepdims=True)
    xSampleStd = np.sqrt(np.sum(xDev**2, axis=0, keepdims=True) / (N - 1))
    ySampleStd = np.sqrt(np.sum(yDev**2, axis=0, keepdims=True) / (N - 1))
    xIdxValid = xSampleStd > 0
    yIdxValid = ySampleStd > 0
    xSampleStdCorrected = xSampleStd + 1 * (~xIdxValid)
    ySampleStdCorrected = ySampleStd + 1 * (~yIdxValid)
    xDev /= xSampleStdCorrected
    yDev /= ySampleStdCorrected
    std = xDev.T @ yDev / (N - 1)
    std[:, ~yIdxValid[0]] = np.nan
    std[~xIdxValid[0]] = np.nan
    return std


def vectorCorrelation(x, y, axis=-1):
    """measure the correlation of every element in x with every element in y on axis=axis"""
    assert x.shape == y.shape, "x and y need to have the same shape!"
    N = x.shape[axis]
    xDev = x - np.mean(x, axis=axis, keepdims=True)
    yDev = y - np.mean(y, axis=axis, keepdims=True)
    xSampleStd = np.sqrt(np.sum(xDev**2, axis=axis, keepdims=True) / (N - 1))
    ySampleStd = np.sqrt(np.sum(yDev**2, axis=axis, keepdims=True) / (N - 1))
    xIdxValid = xSampleStd > 0
    yIdxValid = ySampleStd > 0
    xSampleStdCorrected = xSampleStd + 1 * (~xIdxValid)
    ySampleStdCorrected = ySampleStd + 1 * (~yIdxValid)
    xDev /= xSampleStdCorrected
    yDev /= ySampleStdCorrected
    std = np.sum(xDev * yDev, axis=axis) / (N - 1)
    std *= 1 * np.squeeze(xIdxValid & yIdxValid)
    return std


def vectorRSquared(x, y, axis=-1):
    """
    get r squared between x and y across a particular axis
    treats x as the predictor and y as the data (e.g. SS_total comes from y)
    broadcasting rules apply
    """
    ss_residual = np.sum((y - x) ** 2, axis=axis)
    ss_total = np.sum((y - np.mean(y, axis=axis, keepdims=True)) ** 2, axis=axis)
    ss_total[ss_total == 0] = np.nan
    r_squared = 1 - ss_residual / ss_total
    return r_squared


def diffsame(data, zero=0):
    """
    diffsame returns the diff of a 1-d np.ndarray "data" with the same size as data by appending a zero to the front or back.
    zero=0 means front, zero=1 means back
    """
    assert isinstance(data, np.ndarray) and data.ndim == 1, "data must be a 1-d numpy array"
    if zero == 0:
        return np.append(0, np.diff(data))
    else:
        return np.append(np.diff(data), 0)


def getGaussKernel(timestamps, width, nonzero=True):
    """
    create gaussian kernel (sum=1) around the "timestamps" array with width in units of timestamps
    if nonzero=True, will remove zeros from the returned values (numerical zeros, obviously)
    """
    kx = timestamps - np.mean(timestamps)
    kk = np.exp(-(kx**2) / (2 * width) ** 2)
    kk = kk / np.sum(kk)
    if nonzero:
        # since sp.linalg.convolution_matrix only needs nonzero values, this is way faster
        kk = kk[kk > 0]
    return kk


def fivePointDer(signal, h, axis=-1, returnIndex=False):
    # takes the five point stencil as an estimate of the derivative
    assert isinstance(signal, np.ndarray), "signal must be a numpy array"
    assert -1 <= axis <= signal.ndim, "requested axis does not exist"
    N = signal.shape[axis]
    assert N >= 4 * h + 1, "h is too large for the given array -- it needs to be less than (N-1)/4!"
    signal = np.moveaxis(signal, axis, 0)
    n2 = slice(0, N - 4 * h)
    n1 = slice(h, N - 3 * h)
    p1 = slice(3 * h, N - h)
    p2 = slice(4 * h, N)
    fpd = (1 / (12 * h)) * (-signal[p2] + 8 * signal[p1] - 8 * signal[n1] + signal[n2])
    fpd = np.moveaxis(fpd, 0, axis)
    if returnIndex:
        return fpd, slice(2 * h, N - 2 * h)  # index of central points for each computation
    return fpd


def butterworthbpf(image, lowcut, highcut, order=1, fs=None, returnFull=False):
    """
    2D butterworth filter in frequency domain
    Maps the butterworth transfer function onto a 2D frequency grid corresponding to the fftshifted output out np.fft.fft2
    If fs is None, then assumes that lowcut and highcut are in units of halfcycles/sample, just like scipy.signal.
    returnFull switch allows return of multiple useful variables for plotting and debugging results
    Allows for lowpass or highpass filter by setting respective cut frequency to None
    """
    if (lowcut is not None) and (highcut is not None):
        mode = "bandpass"
        assert highcut > lowcut, "highcut frequency should be greater than lowcut frequency"
        assert lowcut > 0, "frequencies must be positive"
    elif (lowcut is not None) and (highcut is None):
        mode = "highpass"
        assert lowcut > 0, "frequencies must be positive"
    elif (lowcut is None) and (highcut is not None):
        mode = "lowpass"
        assert highcut > 0, "frequencies must be positive"
    else:
        raise ValueError("both lowcut and highcut are None, nothing to do!")

    # comment a little better -- and specify precisely what lowcut and highcut mean here, (with respect to pixel frequency)
    ny, nx = image.shape[-2:]
    ndfty, ndftx = 2 * ny + 1, 2 * nx + 1

    # establish frequency grid for ffts
    if fs is None:
        fs = 2  # as in scipy, this assumes the frequencies scale from 0 to 1, where 1 is the nyquist frequency
    yfreq = np.fft.fftshift(np.fft.fftfreq(ndfty, 1 / fs))  # dft frequencies along x axis (second axis)
    xfreq = np.fft.fftshift(np.fft.fftfreq(ndftx, 1 / fs))  # dft frequencies along y axis (first axis)
    freq = np.sqrt(yfreq.reshape(-1, 1) ** 2 + xfreq.reshape(1, -1) ** 2)  # 2-D dft frequencies corresponds to fftshifted fft2 output

    # transfer function for butterworth filter
    gain = lambda freq, cutoff, order: 1 / (1 + (freq / cutoff) ** (2 * order))  # gain of butterworth filter
    # highpass component
    if not (mode == "lowpass"):
        highpass = 1 - gain(freq, lowcut, order)  # lowpass transfer function
    else:
        highpass = np.ones_like(freq)
    # lowpass component
    if not (mode == "highpass"):
        lowpass = gain(freq, highcut, order)  # highpass transfer function
    else:
        highpass = np.ones_like(freq)
    # create bandpass filter (in 2D, corresponding to fftshifted fft2 output)
    bandpass = lowpass * highpass

    # measure fourier transform with extra points to ensure symmetric frequency spacing and good coverage
    fftImage = np.fft.fftshift(np.fft.fft2(image, (ndfty, ndftx)), axes=(-2, -1))

    # filter image, shift back, take the real part
    filteredImage = np.fft.ifft2(np.fft.ifftshift(bandpass * fftImage, axes=(-2, -1)))[:ny, :nx].real

    # if returnFull, provide all of these outputs because it's useful for making plots and debugging
    if returnFull:
        return filteredImage, fftImage, bandpass, xfreq, yfreq
    # otherwise just return filtered image
    return filteredImage


def phaseCorrelation(staticImage, shiftedImage, eps=0, window=None):
    # phaseCorrelation computes the phase correlation between the two inputs
    # the result is the fftshifted correlation map describing the phase-specific overlap after shifting "shiftedImage"
    # softens ringing with eps when provided
    # if provided, window should be a 1-d or 2-d window function (if 1-d, uses the outer product of itself)
    # -- note -- I tried an ndim phase correlation (where both static and shifted images are 3-d), and it's slower than just doing the 2d version in a loop...
    assert staticImage.shape[-2:] == shiftedImage.shape[-2:], "images must have same shape in last two dimensions"
    assert not (
        staticImage.ndim == 3 and shiftedImage.ndim == 3
    ), "can do multiple comparisons, but only one of the static or shifted image can be 3-dimensional"
    if window is not None:
        if window.ndim == 1:
            window = np.outer(window, window)
        assert window.shape == staticImage.shape[-2:], "window must have same shape as images"
        staticImage = window * staticImage
        shiftedImage = window * shiftedImage
    fftStatic = np.fft.fft2(staticImage)
    fftShiftedConjugate = np.conjugate(np.fft.fft2(shiftedImage))
    R = fftStatic * fftShiftedConjugate
    R /= eps + np.absolute(R)
    return np.fft.fftshift(np.fft.ifft2(R).real, axes=(-2, -1))


def convolveToeplitz(data, kk, axis=-1, mode="same", device="cpu"):
    # convolve data on requested axis (default:-1) using a toeplitz matrix of kk
    # equivalent to np.convolve(data,kk,mode=mode) for each array on requested axis in data
    # uses torch for possible GPU speed up, default device set after imports
    assert -1 <= axis <= data.ndim, "requested axis does not exist"
    data = np.moveaxis(data, axis, -1)  # move target axis
    dataShape = data.shape
    with torch.no_grad():
        # if there are not many signals to convolve, this is a tiny slower
        # if there are many signals to convolve (order of ROIs in a recording), this is waaaaayyyy faster
        convMat = torch.tensor(sp.linalg.convolution_matrix(kk, dataShape[-1], mode=mode).T).to(device)
        dataReshape = torch.tensor(np.reshape(data, (-1, dataShape[-1]))).to(device)
        output = torch.matmul(dataReshape, convMat).cpu().numpy()
    newDataShape = (*dataShape[:-1], convMat.shape[1])
    output = np.reshape(output, newDataShape)
    if device == "cuda":
        # don't keep data on GPU
        del convMat, dataReshape  # delete variables
        torch.cuda.empty_cache()  # clear memory
    return np.moveaxis(output, -1, axis)


# ---------------------------------- workspace management ----------------------------------
def readableBytes(numBytes):
    if numBytes == 0:
        return "0B"
    sizeUnits = ("B", "KB", "MB", "GB", "TB")
    sizeIndex = int(math.floor(math.log(numBytes, 1024)))
    sizeBytes = math.pow(1024, sizeIndex)
    readableSize = round(numBytes / sizeBytes, 2)
    return f"{readableSize} {sizeUnits[sizeIndex]}"


def printWorkspace(maxToPrint=12):
    """
    Original author: https://stackoverflow.com/users/1870254/jan-glx
    Reference: https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    """
    callerGlobals = inspect.currentframe().f_back.f_globals
    variables = [(name, sys.getsizeof(value)) for name, value in callerGlobals.items()]
    variables = sorted(variables, key=lambda x: -x[1])  # Sort by variable size
    # variables = [(name, sys.getsizeof(value)) for name, value in callerLocals.items()]
    totalMemory = sum([v[1] for v in variables])
    print(f"Workspace Size: {readableBytes(totalMemory)}")
    print("Note: this method is not equipped to handle complicated variables, it is possible underestimating the workspace size!\n")
    for name, size in variables[: min(maxToPrint, len(variables))]:
        print(f"{name:>30}: {readableBytes(size):>8}")


# ---------------------------------- type checks ----------------------------------
def check_iterable(val):
    """duck-type check if val is iterable, if so return, if not, make it a list"""
    try:
        # I am a duck and ducks go quack
        _ = iter(val)
    except:
        # not an iterable, woohoo!
        return [val]  # now it is ha ha ha ha!
    else:
        # it's 5pm somewhere
        return val


# ------------------------- linear algebra ---------------------------
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
