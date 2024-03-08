import os
import sys
import time
from tqdm import tqdm
import numpy as np

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

import faststats as fs
from faststats.faststats import _requires_q

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


def get_fsmethod(method):
    return getattr(fs, method)


def get_npmethod(method):
    return getattr(np, method)


def test_method(data, axis, method, q=None):
    # do standard numpy operation
    if method in _requires_q:
        t = time.time()
        npout = get_npmethod(method)(data, q, axis=axis)
        nptime = time.time() - t
    else:
        t = time.time()
        npout = get_npmethod(method)(data, axis=axis)
        nptime = time.time() - t

    # do faststat version of numpy operation
    if method in _requires_q:
        t = time.time()
        fastout = get_fsmethod(method)(data, axis=axis, q=q)
        fasttime = time.time() - t
    else:
        t = time.time()
        fastout = get_fsmethod(method)(data, axis=axis)
        fasttime = time.time() - t

    # check if results are same
    valid = np.allclose(npout, fastout)

    # return whether results are same and times of operations
    return valid, nptime, fasttime


if __name__ == "__main__":
    shapes_and_axes = [
        [(10, 1000), 1],  # small/medium <-- 1
        [(10, 1000), 0],  # small/medium <-- 0
        [(1000, 1000), 1],  # medium/medium <-- 1
        [(1000, 1000), 0],  # medium/medium <-- 0
        [(1000, 10000), 1],  # medium/large <-- 1
        [(1000, 10000), 0],  # medium/large <-- 0
        [(10000, 10000), 1],  # large/large <-- 1
        [(10000, 10000), 0],  # large/large <-- 0
        # now 3 dim shapes
        [(10, 10, 1000), 2],  # small/medium <-- 2
        [(10, 10, 1000), (1, 2)],  # small/medium <-- (1, 2)
        [(10, 1000, 1000), 2],  # medium/medium <-- 2
        [(10, 1000, 1000), (1, 2)],  # medium/medium <-- (1, 2)
        # [(10, 1000, 10000), 2],  # medium/large <-- 2
        # [(10, 1000, 10000), (1, 2)],  # medium/large <-- (1, 2))
        # [(10, 10000, 10000), 2],  # large/large <-- 2
        # [(10, 10000, 10000), (1, 2)],  # large/large <-- (1, 2)
    ]

    dtype = np.float32

    num_tests = len(shapes_and_axes)

    data = []
    for shape, _ in tqdm(shapes_and_axes, desc="making data", leave=True, total=num_tests):
        data.append(np.random.normal(0, 1, shape).astype(dtype))

    num_repeats = 50
    nptime = [np.zeros(num_tests) for _ in methods]
    fstime = [np.zeros(num_tests) for _ in methods]
    valid = [np.zeros(num_tests) for _ in methods]
    for m, method in tqdm(enumerate(methods), desc="method variants", leave=True, total=len(methods)):
        for i, (shape, axis) in tqdm(enumerate(shapes_and_axes), desc="shape/axis variants", leave=False, total=num_tests):
            c_valid = True
            c_np_time = 0.0
            c_fs_time = 0.0
            for repeat in tqdm(range(num_repeats), desc="repeat", leave=False, total=num_repeats):
                cc_valid, cc_np_time, cc_fs_time = test_method(data[i], axis, method, q=0.5)
                if repeat > 0:
                    c_valid &= cc_valid
                    c_np_time += cc_np_time
                    c_fs_time += cc_fs_time

            # save results
            valid[m][i] = c_valid
            nptime[m][i] = c_np_time / (num_repeats - 1)
            fstime[m][i] = c_fs_time / (num_repeats - 1)

    for m, method in enumerate(methods):
        print("method:", method, "allvalid:", np.all(valid[m]))
        for i, (shape, axis) in enumerate(shapes_and_axes):
            if fstime[m][i] > 0:
                speedup = nptime[m][i] / fstime[m][i]
            else:
                if nptime[m][i] == 0:
                    speedup = 1.0
                else:
                    speedup = np.inf
            print(f"shape={shape}, axis={axis}, speedup={speedup}, nptime={nptime[m][i]}, fstime={fstime[m][i]}")
        print("")
