from copy import copy
from itertools import chain, combinations
import numpy as np
from .wrangling import check_iterable


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
    # each fold gets minimum number of samples, assign remainder evenly to as many as necessary
    samplesPerFold = [int(minimumSamples + 1 * (f < remainder)) for f in range(numFold)]
    # defines where to start and stop for each fold
    sampleIdxPerFold = [0, *np.cumsum(samplesPerFold)]
    randomOrder = samples[np.random.permutation(numSamples)]  # random permutation of samples
    foldIdx = [randomOrder[sampleIdxPerFold[i] : sampleIdxPerFold[i + 1]] for i in range(numFold)]  # assign samples to each cross-validation fold
    if even:
        foldIdx = [fi[:minimumSamples] for fi in foldIdx]
    return foldIdx
