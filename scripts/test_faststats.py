import os
import sys
import time
from tqdm import tqdm
import numpy as np

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis.faststats import faststat, _requires_q

methods = [
    "ptp",
    "percentile",
    "nanpercentile",
    "quantile",
    "nanquantile",
    "median",
    "average",
    "mean",
    "std",
    "var",
    "nanmedian",
    "nanmean",
    "nanstd",
    "nanvar",
]


def get_npmethod(method):
    return getattr(np, method)


def test_method(data, axis, method, q=None):
    # do standard numpy operation
    t = time.time()
    if method in _requires_q:
        npout = get_npmethod(method)(data, q, axis=axis)
    else:
        npout = get_npmethod(method)(data, axis=axis)
    nptime = time.time() - t

    # do faststat version of numpy operation
    t = time.time()
    if method in _requires_q:
        fastout = faststat(data, method, axis=axis, q=q)
    else:
        fastout = faststat(data, method, axis=axis)
    fasttime = time.time() - t

    # check if results are same
    valid = np.allclose(npout, fastout)

    # return whether results are same and times of operations
    return valid, nptime, fasttime


if __name__ == "__main__":
    N = [1000, 5000, 10000]
    ND = [2]
    num_n = len(N)
    num_nd = len(ND)
    num_tests = 10
    speedup = [np.zeros((num_n, num_nd)) for _ in methods]
    valid = [np.zeros((num_n, num_nd)) for _ in methods]
    for m, method in tqdm(enumerate(methods)):
        for i, n in tqdm(enumerate(N)):
            for j, nd in enumerate(ND):
                c_valid = True
                c_nptime = 0
                c_fstime = 0
                shape = (n,) * nd
                data = np.random.normal(0, 1, shape)
                for nt in range(num_tests):
                    cc_valid, cc_nptime, cc_fasttime = test_method(data, -1, method, q=0.5)
                    if nt > 0:
                        c_valid &= cc_valid
                        c_nptime += cc_nptime
                        c_fstime += cc_fasttime
                valid[m][i, j] = c_valid
                if c_fstime == 0:
                    if c_nptime == 0:
                        speedup[m][i, j] = 1
                    else:
                        speedup[m][i, j] = np.inf
                else:
                    speedup[m][i, j] = c_nptime / c_fstime

    for m, method in enumerate(methods):
        print("method:", method, "allvalid:", np.all(valid[m]), "speedup factor:")
        print(speedup[m])
        print("")
