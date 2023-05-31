import numpy as np
import scipy as sp

def diffsame(data, zero=0):
    # diffsame returns the diff of a 1-d np.ndarray "data" with the same size as data by appending a zero to the front or back. 
    # zero=0 means front, zero=1 means back
    assert isinstance(data, np.ndarray) and data.ndim==1, "data must be a 1-d numpy array"
    if zero==0: return np.append(0, np.diff(data))
    else: return np.append(np.diff(data),0)

def getGaussKernel(timestamps, width, nonzero=True):
    kx = timestamps - np.mean(timestamps)
    kk = np.exp(-kx**2/(2*width)**2)
    kk = kk / np.sum(kk)
    if nonzero:
        kk = kk[kk>0]
    return kk

def nearestpoint(x,y,mode='nearest'):
    # fast implementation of nearest neighbor index between two arrays
    # based on file-exchange from https://uk.mathworks.com/matlabcentral/fileexchange/8939-nearestpoint-x-y-m
    # returns index of y closest to each point in x and distance between points
    # if mode set to previous or next, will exact match or next closest in specified direction
    # if mode set to nearest and x[i] exactly in between two y's, will take next y rather than previous
    assert mode=='nearest' or mode=='previous' or mode=='next', "mode must be one of the following strings: ('nearest', 'previous', 'next')"
    assert isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.ndim==1 and y.ndim==1, "arrays must be one-dimensional numpy arrays"
    
    # If either x or y is empty, return empty arrays
    if len(x)==0 or len(y)==0: return np.array(()),np.array(())
     
    # Otherwise, find nearest neighbor, return indices and distance
    xi = np.argsort(x)
    xs = x[xi] # sorted x
    xi = np.argsort(xi) # return to x order
    nx = len(x)
    cx = np.zeros(nx) # used to identify which are x's in concatenate
    qx = np.isnan(xs) # index of nans

    yi = np.argsort(y)
    ys = y[yi]
    ny = len(y)
    cy = np.ones(ny)
    
    # Created concatenate value vector and identity vector
    xy = np.concatenate((xs,ys))
    cxy = np.concatenate((cx,cy))

    # Sort values and identities
    xyi = np.argsort(xy)
    cxy = cxy[xyi] # cxy[i] = 0 -> xy[i] belongs to x, = 1 -> xy[i] belongs to y
    ii = np.cumsum(cxy) # what index of y is equal to or least greater than each x (index of "next" nearest neighbor)
    ii = ii[cxy==0] # only keep indices for each x value

    # Note: if m==previous or next, then will use previous or next closest even if exact match exists! (maybe include a conditional for that?)
    if mode=='previous':
        equalOrPrevious = np.array([yidx if (yidx<ny and xs[xidx]-ys[yidx.astype(int)]==0) else yidx-1 for (xidx,yidx) in enumerate(ii)]) # preserve index if equal, otherwise use previous from y
        ind = np.where(equalOrPrevious>=0, equalOrPrevious, np.nan) # Preserve index if within y, otherwise nan
    elif mode=='next':
        ind = np.where(ii<ny, ii, np.nan) # Preserve index if within y, otherwise nan
    else:
        ii = np.stack((ii,ii-1))
        ii[ii>=ny]=ny-1
        ii[ii<0]=0
        yy = ys[ii.astype(np.uint64)]
        dy = np.argmin(np.abs(xs-yy),axis=0)
        ind = ii[dy,np.arange(nx)].astype(np.float64)

    ind[qx] = np.nan # reset nan indices back to nan
    ind = ind[xi] # resort to x
    ind = np.array([yi[idx.astype(np.uint64)] if not np.isnan(idx) else np.nan for idx in ind]) # resort to y

    # compute distance between nearest points
    ynearest = np.array([y[idx.astype(np.uint64)] if not np.isnan(idx) else np.nan for idx in ind])
    d = np.abs(x-ynearest)
    
    return ind, d

def fivePointDer(signal,h,axis=-1,returnIndex=False):
    # takes the five point stencil as an estimate of the derivative
    assert isinstance(signal,np.ndarray), "signal must be a numpy array"
    assert -1 <= axis <= signal.ndim, "requested axis does not exist"
    N = signal.shape[axis]
    assert N >= 4*h+1, "h is too large for the given array -- it needs to be less than (N-1)/4!"
    signal = np.moveaxis(signal, axis, 0)
    n2 = slice(0,N-4*h)
    n1 = slice(h,N-3*h)
    p1 = slice(3*h,N-h)
    p2 = slice(4*h,N)
    fpd = (1/(12*h)) * (-signal[p2] + 8*signal[p1] - 8*signal[n1] + signal[n2])
    fpd = np.moveaxis(fpd, 0, axis)
    if returnIndex: return fpd, slice(2*h,N-2*h) # index of central points for each computation
    return fpd

def phaseCorrelation(staticImage,shiftedImage,eps=0,window=None):
    # phaseCorrelation computes the phase correlation between the two inputs
    # the result is the fftshifted correlation map describing the phase-specific overlap after shifting "shiftedImage"
    # softens ringing with eps when provided
    # if provided, window should be a 1-d or 2-d window function (if 1-d, uses the outer product of itself)
    assert staticImage.shape[-2:]==shiftedImage.shape[-2:], "images must have same shape in last two dimensions"
    assert not (staticImage.ndim==3 and shiftedImage.ndim==3), "can do multiple comparisons, but only one of the static or shifted image can be 3-dimensional"
    if window is not None:
        if window.ndim==1: window = np.outer(window,window)
        assert window.shape==staticImage.shape[-2:], "window must have same shape as images"
        staticImage = window * staticImage
        shiftedImage = window * shiftedImage
    fftStatic = np.fft.fft2(staticImage)
    fftShiftedConjugate = np.conjugate(np.fft.fft2(shiftedImage))
    R = fftStatic * fftShiftedConjugate
    R /= (eps+np.absolute(R))
    return np.fft.fftshift(np.fft.ifft2(R).real, axes=(-2,-1))

def convolveToeplitz(data, kk, axis=-1, mode='same'):
    # convolve data on requested axis (default:-1) using a toeplitz matrix of kk
    # equivalent to np.convolve(data,kk,mode=mode) for each array on requested axis in data
    assert -1 <= axis <= data.ndim, "requested axis does not exist"
    data = np.moveaxis(data, axis, -1) # move target axis
    dataShape = data.shape
    convMat = sp.linalg.convolution_matrix(kk, dataShape[-1], mode=mode).T
    dataReshape = np.reshape(data, (-1, dataShape[-1]))
    output = dataReshape @ convMat
    newDataShape = (*dataShape[:-1],convMat.shape[1])
    output = np.reshape(output, newDataShape)
    return np.moveaxis(output, -1, axis)

def edge2center(edges):
    assert isinstance(edges, np.ndarray) and edges.ndim==1, "edges must be a 1-d numpy array"
    return edges[:-1] + np.diff(edges)/2

def digitizeEqual(data, mn, mx, nbins):
    # digitizeEqual returns the bin of each element from data within each equally spaced bin between mn,mx
    # (it's like np.digitize but faster and only works with equal bins)
    binidx = (data-mn)/(mx-mn)*float(nbins)
    binidx[binidx<0]=0
    binidx[binidx>nbins-1]=nbins-1
    return binidx.astype(int)

def vectorCorrelation(x,y):
    # for each column in x, measure the correlation with each column in y
    assert x.shape==y.shape, "x and y need to have the same shape!"
    N = x.shape[0]
    xDev = x - np.mean(x,axis=0)
    yDev = y - np.mean(y,axis=0)
    xSampleStd = np.sqrt(np.sum(xDev**2,axis=0)/(N-1))
    ySampleStd = np.sqrt(np.sum(yDev**2,axis=0)/(N-1))
    xStandard = xDev/xSampleStd
    yStandard = yDev/ySampleStd
    return np.sum(xStandard * yStandard,axis=0) / (N-1) 

def cvFoldSplit(numSamples, numFold):
    # generates list of indices of equally sized randomly selected samples to be used in numFold-crossvalidation for a given number of samples
    minimumSamples = np.floor(numSamples / numFold) 
    remainder = numSamples - numFold*minimumSamples 
    samplesPerFold = [int(minimumSamples + 1*(f<remainder)) for f in range(numFold)] # each fold gets minimum number of samples, assign remainder evenly to as many as necessary
    sampleIdxPerFold = [0, *np.cumsum(samplesPerFold)] # defines where to start and stop for each fold
    randomOrder = np.random.permutation(numSamples) # random permutation of samples
    foldIdx = [randomOrder[sampleIdxPerFold[i]:sampleIdxPerFold[i+1]] for i in range(numFold)] # assign samples to each cross-validation fold
    return foldIdx




