import sys
import inspect
import math


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
