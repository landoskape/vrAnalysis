import sys
import inspect
import math
import numpy as np
from itertools import chain
from hashlib import sha256


def get_confirmation(message: str = ""):
    message = message + " Are you sure you want to proceed? (type 'proceed' to continue):"
    confirmation = input(message)
    return confirmation == "proceed"


class AttributeDict(dict):
    """
    helper class for accessing dictionary values as attributes

    for example, if d = {"a": 1, "b": 2}, then you can access d["a"] as d.a
    very helpful for situations when a method requires "args" as an input, which
    is usually the output of argparse.ArgumentParser.parse_args(), but you want to
    run that method without an argparse object
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def asdict(self):
        return dict(self)


# ------------------------------------ data wrangling ------------------------------------
def compare_nested_objects(reference, comparison, path="root"):
    """
    Deep comparison of nested dictionaries and lists with detailed error reporting.

    Parameters
    ----------
    reference : dict or list
        The reference object to compare against
    comparison : dict or list
        The object to compare with the reference
    path : str
        Current path in the nested structure (for error reporting)

    Returns
    -------
    bool
        True if objects are deeply equal, False otherwise
    """
    # Handle different types
    if type(reference) != type(comparison):
        print(f"Type mismatch at {path}: {type(reference).__name__} vs {type(comparison).__name__}")
        return False

    if isinstance(reference, dict):
        # Check if all keys in reference exist in comparison
        ref_keys = set(reference.keys())
        comp_keys = set(comparison.keys())
        if ref_keys != comp_keys:
            missing_in_comp = ref_keys - comp_keys
            extra_in_comp = comp_keys - ref_keys
            if missing_in_comp:
                print(f"Missing keys in comparison at {path}: {missing_in_comp}")
            if extra_in_comp:
                print(f"Extra keys in comparison at {path}: {extra_in_comp}")
            return False

        # Recursively compare each value
        for key in reference:
            if not compare_nested_objects(reference[key], comparison[key], f"{path}.{key}"):
                return False
        return True

    elif isinstance(reference, list):
        # Check if lists have same length
        if len(reference) != len(comparison):
            print(f"List length mismatch at {path}: {len(reference)} vs {len(comparison)}")
            return False

        # Compare each element
        for i in range(len(reference)):
            if not compare_nested_objects(reference[i], comparison[i], f"{path}[{i}]"):
                return False
        return True

    elif isinstance(reference, np.ndarray):
        # Handle numpy arrays
        if not isinstance(comparison, np.ndarray):
            print(f"Array type mismatch at {path}: numpy array vs {type(comparison).__name__}")
            return False

        # Check shapes first
        if reference.shape != comparison.shape:
            print(f"Array shape mismatch at {path}: {reference.shape} vs {comparison.shape}")
            return False

        # Handle object arrays (containing Python objects)
        if reference.dtype == object:
            for i in range(reference.size):
                if not compare_nested_objects(reference.flat[i], comparison.flat[i], f"{path}[{i}]"):
                    return False
            return True

        # Handle numerical arrays
        try:
            if not np.allclose(reference, comparison):
                print(f"Array values differ at {path}: shapes {reference.shape} vs {comparison.shape}")
                return False
            return True
        except Exception as e:
            print(f"Array comparison error at {path}: {e}")
            return False

    else:
        # For primitive types, use direct comparison
        try:
            if reference != comparison:
                print(f"Value mismatch at {path}: {reference} vs {comparison}")
                return False
            return True
        except Exception as e:
            print(f"Comparison error at {path}: {e}")
            return False


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists, map_func=list):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(map_func, zip(*list_of_lists))


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


# ---------------------------------- workspace management ----------------------------------
def readable_bytes(numBytes):
    if numBytes == 0:
        return "0B"
    sizeUnits = ("B", "KB", "MB", "GB", "TB")
    sizeIndex = int(math.floor(math.log(numBytes, 1024)))
    sizeBytes = math.pow(1024, sizeIndex)
    readableSize = round(numBytes / sizeBytes, 2)
    return f"{readableSize} {sizeUnits[sizeIndex]}"


def print_workspace(maxToPrint=12):
    """
    Original author: https://stackoverflow.com/users/1870254/jan-glx
    Reference: https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    """
    callerGlobals = inspect.currentframe().f_back.f_globals
    variables = [(name, sys.getsizeof(value)) for name, value in callerGlobals.items()]
    variables = sorted(variables, key=lambda x: -x[1])  # Sort by variable size
    # variables = [(name, sys.getsizeof(value)) for name, value in callerLocals.items()]
    totalMemory = sum([v[1] for v in variables])
    print(f"Workspace Size: {readable_bytes(totalMemory)}")
    print("Note: this method is not equipped to handle complicated variables, it is possible underestimating the workspace size!\n")
    for name, size in variables[: min(maxToPrint, len(variables))]:
        print(f"{name:>30}: {readable_bytes(size):>8}")


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


# --------------------------------- difference handling ---------------------------------
def get_all_pairwise_stats(data: np.ndarray, axis: int = 0, method: str = "difference"):
    """
    helper method for getting the difference between two keys in a dictionary
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    data = np.swapaxes(data, axis, 0)

    N = data.shape[0]  # number of elements
    S = data.shape[1]  # number of samples
    if method == "difference":
        stat = data[None] - data[:, None]
    elif method == "correlation":
        raise NotImplementedError("Correlation not implemented yet")
    else:
        raise ValueError(f"Invalid method: {method}")

    num_comparisons = N - 1
    pwstats = np.full((num_comparisons, S), np.nan)
    for i in range(1, num_comparisons + 1):
        idx0 = np.arange(0, N - i)
        idx1 = np.arange(i, N)
        pwstats[i - 1] = np.mean(stat[idx0, idx1], axis=0)

    return pwstats


def concat_with_spacer(listN, x, axis=1):
    if len(listN) == 1:
        return listN[0]
    # chain.from_iterable avoids making intermediate Python lists
    interleaved = chain.from_iterable(zip(listN[:-1], [x] * (len(listN) - 1)))
    return np.concatenate((*interleaved, listN[-1]), axis=axis)


def stable_hash(*items, shorten: bool = True) -> str:
    """Create a stable hash of a list of items.

    Parameters
    ----------
    *items : Any
        The items to hash.
    shorten : bool
        If True, the hash will be shortened to 8 characters. Default is True.

    Returns
    -------
    str
        The stable hash of the items.
    """
    h = sha256()
    for x in items:
        h.update(repr(x).encode())
    digest = h.hexdigest()
    if shorten:
        return digest[:8]
    return digest
