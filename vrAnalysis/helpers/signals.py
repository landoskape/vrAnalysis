import numpy as np
import scipy as sp
import torch
from numba import njit, prange

from .wrangling import transpose_list, named_transpose


# ---------------------------------- signal processing ----------------------------------
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


def norm01(data, axis=None):
    """
    normalize data to 0-1 range on a particular axis (or all)
    """
    datamin = np.nanmin(data, axis=axis, keepdims=True)
    datamax = np.nanmax(data, axis=axis, keepdims=True)
    return (data - datamin) / (datamax - datamin)


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


def vectorCorrelation(x, y, covariance=False, axis=-1, ignore_nan=False):
    """
    measure the correlation of every element in x with every element in y on axis=axis
    if covariance=True, will measure the covariance
    if ignore_nan=True, will ignore NaN values in the correlation calculation
    """
    assert x.shape == y.shape, "x and y need to have the same shape!"
    meanfunc = np.mean if not ignore_nan else np.nanmean
    sumfunc = np.sum if not ignore_nan else np.nansum

    N = x.shape[axis]
    xDev = x - meanfunc(x, axis=axis, keepdims=True)
    yDev = y - meanfunc(y, axis=axis, keepdims=True)
    if not covariance:
        xSampleStd = np.sqrt(sumfunc(xDev**2, axis=axis, keepdims=True) / (N - 1))
        ySampleStd = np.sqrt(sumfunc(yDev**2, axis=axis, keepdims=True) / (N - 1))
        xIdxValid = xSampleStd > 0
        yIdxValid = ySampleStd > 0
        xSampleStdCorrected = xSampleStd + 1 * (~xIdxValid)
        ySampleStdCorrected = ySampleStd + 1 * (~yIdxValid)
    else:
        xSampleStdCorrected = 1
        ySampleStdCorrected = 1
    xDev /= xSampleStdCorrected
    yDev /= ySampleStdCorrected
    std = sumfunc(xDev * yDev, axis=axis) / (N - 1)
    if not covariance:
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


@torch.no_grad()
def convolveToeplitz(data, kk, axis=-1, mode="same", device="cpu", force_torch=False):
    # convolve data on requested axis (default:-1) using a toeplitz matrix of kk
    # equivalent to np.convolve(data,kk,mode=mode) for each array on requested axis in data
    # uses torch for possible GPU speed up, default device set after imports
    assert -1 <= axis <= data.ndim, "requested axis does not exist"

    use_torch = isinstance(data, torch.Tensor)
    if use_torch:
        data = data.to(device)
        data = torch.moveaxis(data, axis, -1)
        data_shape = data.shape
        conv_mat = torch.tensor(sp.linalg.convolution_matrix(kk, data_shape[-1], mode=mode).T).to(device)
        data_reshaped = data.reshape(-1, data_shape[-1]).contiguous()
        output = torch.matmul(data_reshaped, conv_mat)
        new_data_shape = (*data_shape[:-1], conv_mat.shape[1])
        output = output.reshape(new_data_shape)
        if device == "cuda":
            # don't keep data on GPU
            del conv_mat, data_reshaped  # delete variables
            torch.cuda.empty_cache()  # clear memory
        return output.moveaxis(-1, axis)
    elif force_torch:
        data = np.moveaxis(data, axis, -1)  # move target axis
        dataShape = data.shape
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
    else:
        data = np.moveaxis(data, axis, -1)  # move target axis
        dataShape = data.shape
        convMat = sp.linalg.convolution_matrix(kk, dataShape[-1], mode=mode).T
        dataReshape = np.reshape(data, (-1, dataShape[-1]))
        output = np.matmul(dataReshape, convMat)
        newDataShape = (*dataShape[:-1], convMat.shape[1])
        output = np.reshape(output, newDataShape)
        return np.moveaxis(output, -1, axis)


def get_fourier_basis(L, Fs=1.0):
    """
    create discrete fourier basis set
    returns
    -------
    f: frequencies
    basis: (N x L) basis set of fourier modes

    """

    x = np.linspace(0, L - 1, L)  # each column is a different position
    n = sp.fft.rfftfreq(L, 1 / L)  # get one-sided frequencies (in integers)
    f = n * Fs / L  # get frequencies (according to sampling rate)

    # get a one-sided fourier basis
    basis = np.exp(-2j * np.pi * x.reshape(1, -1) * n.reshape(-1, 1) / L)

    # return frequency and basis
    return f, basis


def fit_exponentials(data, bias=False):
    """
    Fit exponential functions to data.

    Assumes that each row of data is the exponential to fit,
    with x values being np.arange(0, data.shape[1])

    returns a list of popt for each row of data
    """

    def _exp_func_bias(x, amplitude, decay, bias):
        return amplitude * np.exp(-x / decay) + bias

    def _exp_func(x, amplitude, decay):
        return amplitude * np.exp(-x / decay)

    assert data.ndim == 2, "data has to be a 2D numpy array"

    # Prepare initial estimates of the parameters
    init_amplitude = data[:, 0]
    init_decay = data.shape[1] / 10 * np.ones(data.shape[0])

    # set bounds for amplitude and decay
    bounds = [[-np.inf, np.inf], [0, np.inf]]

    # handle possibility of bias term
    if bias:
        init_bias = data[:, -1]
        bounds.append([-np.inf, np.inf])

    _func = _exp_func_bias if bias else _exp_func

    # convert to expected format for scipy.optimize.curve_fit
    bounds = list(zip(*bounds))

    # Do fit
    x_vals = np.arange(0, data.shape[1])
    popts = []
    for idx, y in enumerate(data):
        # set initial parameter
        p0 = [init_amplitude[idx], init_decay[idx]]
        if bias:
            p0.append(init_bias[idx])

        # fit data
        popts.append(sp.optimize.curve_fit(_func, x_vals, y, p0=p0, bounds=bounds)[0])

    # Measure R2
    residuals = data - np.array([_func(x_vals, *popt) for popt in popts])
    ss_res = np.sum(residuals**2, axis=1)
    ss_tot = np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2, axis=1)
    r_squared = 1 - ss_res / ss_tot

    return transpose_list(popts), r_squared


def gaussian(x, amplitude, center, width):
    """gaussian function"""
    return amplitude * np.exp(-(((x - center) / width) ** 2))


def get_fit(args):
    """
    Fit a function to data

    x: x values
    y: y values
    func: function to fit
    p0: initial parameters
    bounds: bounds for the parameters

    returns the popt and pcov
    """
    x, y, func, p0 = args
    try:
        popt = sp.optimize.curve_fit(func, x, y, p0=p0, bounds=None)[0]
    except RuntimeError:
        # if fit failed, append nans
        popt = [np.nan, np.nan, np.nan]
    except Exception as e:
        # raise all other exceptions
        raise e
    return popt


def fit_gaussians(data, x=None):
    """
    Fit gaussians to data

    Assumes that each row of data is the data to fit,
    with x values being np.arange(0, data.shape[1])
    unless provided

    returns a list of popt for each row of data
    and an r2 value for each fit

    note: will ignore nans, so those need to be handled as expected outside this method!
    """

    assert data.ndim == 2, "data has to be a 2D numpy array"
    if x is None:
        x = np.arange(data.shape[1])

    # Prepare initial estimates of the parameters
    init_amplitude = np.nanmax(data, axis=1)
    init_center = x[np.nanargmax(data, axis=1)]
    init_width = 5

    # Do fit
    popts = []
    for idx, y in enumerate(data):
        # set initial parameter
        p0 = [init_amplitude[idx], init_center[idx], init_width]
        idx_nan = np.isnan(y)

        # fit data
        try:
            popts.append(sp.optimize.curve_fit(gaussian, x[~idx_nan], y[~idx_nan], p0=p0)[0])
        except RuntimeError:
            # if fit failed, append nans
            popts.append([np.nan, np.nan, np.nan])
        except Exception as e:
            # raise all other exceptions
            raise e

    # Measure R2
    residuals = data - np.array([gaussian(x, *popt) for popt in popts])
    ss_res = np.nansum(residuals**2, axis=1)
    ss_tot = np.nansum((data - np.nanmean(data, axis=1, keepdims=True)) ** 2, axis=1)
    r_squared = 1 - ss_res / ss_tot

    amplitude, center, width = named_transpose(popts)
    return amplitude, center, width, r_squared


@njit(parallel=True, fastmath=True, cache=True)
def pad_array(arr, pad_width, mode, cval=0.0):
    """
    Pad array edges with different modes.

    Parameters:
    arr: 1D numpy array
    pad_width: Number of values padded to each edge
    mode: String indicating padding mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
    cval: Constant value for 'constant' mode

    Returns:
    Padded array
    """
    n = len(arr)
    padded = np.zeros(n + 2 * pad_width, dtype=arr.dtype)

    # Fill the center with input array
    padded[pad_width : pad_width + n] = arr

    # Handle different padding modes
    if mode == "reflect":
        for i in prange(pad_width):
            padded[pad_width - 1 - i] = arr[i]
            padded[pad_width + n + i] = arr[n - 1 - i]
    elif mode == "constant":
        padded[:pad_width] = cval
        padded[pad_width + n :] = cval
    elif mode == "nearest":
        padded[:pad_width] = arr[0]
        padded[pad_width + n :] = arr[-1]
    elif mode == "mirror":
        for i in prange(pad_width):
            padded[pad_width - 1 - i] = arr[i + 1]
            padded[pad_width + n + i] = arr[n - 2 - i]
    elif mode == "wrap":
        for i in prange(pad_width):
            padded[pad_width - 1 - i] = arr[n - 1 - i]
            padded[pad_width + n + i] = arr[i]
    return padded


@njit(parallel=True, fastmath=True, cache=True)
def percentile_filter(arr, window_size, percentile, mode="reflect", cval=0.0):
    """
    Numba-accelerated percentile filter with boundary handling.

    Operates on columns! Is like percentile_filter(..., axis=0)

    Parameters:
    arr: 2D numpy array of shape (rows, cols)
    window_size: Integer size of the sliding window
    percentile: Percentile value between 0 and 100
    mode: Boundary handling mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
    cval: Constant value for 'constant' mode

    Returns:
    2D numpy array of same shape as input with percentile filtered values
    """
    rows, cols = arr.shape
    result = np.zeros_like(arr)

    # Calculate padding size
    pad_width = window_size // 2

    # Process each column independently
    for col in prange(cols):
        # Pad column according to mode
        col_data = pad_array(arr[:, col], pad_width, mode, cval)

        # Slide window and compute percentile
        for i in range(rows):
            window = col_data[i : i + window_size]
            result[i, col] = np.percentile(window, percentile)

    return result
