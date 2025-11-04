import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


# ---------------------------------------------------------------------------------------------------
# -------------------------------------- oasis processing methods -----------------------------------
# ---------------------------------------------------------------------------------------------------
def _process_fc(fc: np.ndarray, g: float, deconvolve: callable) -> np.ndarray:
    """Process a single fluorescence trace using oasis.

    Parameters:
    ----------
    fc: np.ndarray
        The fluorescence trace to process.
    g: float
        The g parameter for oasis deconvolution.
    deconvolve: callable
        The oasis deconvolve function.
    """
    # do oasis and cast as single to match with suite2p data
    c_oasis = deconvolve(fc, g=(g,), penalty=1)[1].astype(np.single)
    # oasis sometimes produces random highly negative values... just set them to 0
    c_oasis = np.maximum(c_oasis, 0)
    # return deconvolved trace
    return c_oasis


def oasis_deconvolution(fcorr: np.ndarray, g: float, num_processes: int = cpu_count() - 1) -> list[np.ndarray]:
    """Perform oasis deconvolution on a batch of fluorescence traces.

    Parameters:
    ----------
    fcorr: np.ndarray
        The fluorescence traces to process.
    g: float
        The g parameter for oasis deconvolution.
    num_processes: int
        The number of processes to use for oasis deconvolution.
    """
    if fcorr.ndim != 2:
        raise ValueError("fcorr must be a 2D numpy array.")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")

    # Lazy import of deconvolve method from oasis to not break registration
    # if oasis_deconvolution isn't used.
    try:
        from oasis.functions import deconvolve
    except ImportError as error:
        print("Failed to import deconvolve from oasis.")
        raise error

    # Create partial function with fixed parameters
    process_func = partial(_process_fc, g=g, deconvolve=deconvolve)

    with Pool(num_processes) as pool:
        results = tqdm(pool.imap(process_func, fcorr), total=len(fcorr))
        return list(results)
