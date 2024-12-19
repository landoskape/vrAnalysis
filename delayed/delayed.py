from warnings import warn
from typing import Any, Tuple, Dict, Callable, Set
from collections.abc import Mapping, Sequence
from functools import wraps
import numpy as np
from networkx import DiGraph

from .types import DelayedData, GenericType, T
from .graph import get_uncached_nodes, correct_computed_status, validate_no_cycles


def compute_nested(obj: Any, force_recompute: bool = False, dont_cache: bool = False, depth: int = 0, maximum_depth: int = None) -> Any:
    """Recursively computes any Delayed objects found within nested data structures."""
    if maximum_depth is not None and depth > maximum_depth:
        warn(f"Maximum depth ({maximum_depth}) reached in delayed.compute_nested at object: {type(obj).__name__}. Returning object as-is.")
        return obj

    depth += 1
    kwargs = dict(
        force_recompute=force_recompute,
        dont_cache=dont_cache,
        depth=depth,
        maximum_depth=maximum_depth,
    )

    # Handle Delayed objects
    if isinstance(obj, Delayed):
        maximum_depth = maximum_depth - 1 if maximum_depth is not None else None
        return obj.compute(force_recompute=force_recompute, dont_cache=dont_cache, maximum_depth=maximum_depth)

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return np.array([compute_nested(x, **kwargs) for x in obj.flat]).reshape(obj.shape)

    # Handle mappings (dict-like objects)
    if isinstance(obj, Mapping):
        return type(obj)({compute_nested(key, **kwargs): compute_nested(value, **kwargs) for key, value in obj.items()})

    # Handle sequences (list-like objects, excluding strings)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(compute_nested(item, **kwargs) for item in obj)

    # Handle sets
    if isinstance(obj, Set):
        return type(obj)(compute_nested(item, **kwargs) for item in obj)

    # Handle generators and iterators
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, Mapping, Sequence, Set)):
        return (compute_nested(item, **kwargs) for item in obj)

    # Base case: return the object as-is
    return obj


class Delayed(GenericType):
    def __init__(self, func: Callable[..., T], *args: Tuple[Any, ...], disable_cache: bool = False, **kwargs: Dict[str, Any]):
        self._func = func
        self._args = tuple(args)
        self._kwargs = kwargs
        self.disable_cache = disable_cache
        self.ddata = DelayedData()
        self._computing = False  # For tracking computation status

    @property
    def func(self) -> Callable[..., T]:
        return self._func

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        return self._kwargs

    def update_func(self, func: Callable[..., T]) -> None:
        self._func = func
        self.ddata.clear()

    def update_args(self, *args: Tuple[Any, ...]) -> None:
        self._args = args
        self.ddata.clear()

    def update_kwargs(self, full_reset: bool = False, **kwargs: Dict[str, Any]) -> None:
        if full_reset:
            self._kwargs = kwargs
        else:
            self._kwargs.update(kwargs)
        self.ddata.clear()

    def compute(self, force_recompute: bool = False, recompute_dependencies: bool = False, dont_cache: bool = False, maximum_depth: int = None) -> T:
        """Compute the result and cache if enabled.

        Will iteratively compute all dependencies in arguments and key-word arguments
        if they are also Delayed objects.

        Parameters
        ----------
        force_recompute : bool, optional
            If True, the result will be recomputed even if it is already cached, by default False.
            Will not recompute dependencies unless recompute_dependencies is also True.
        recompute_dependencies : bool, optional
            If True, all dependencies will be recomputed even if they are cached, by default False.
        dont_cache : bool, optional
            If True, the result will not be cached, by default False.
        maximum_depth : int, optional
            The maximum depth to recurse into nested data structures, by default None.
        """
        if self._computing:
            raise RecursionError(f"Circular dependency detected in delayed computation of {self.func.__name__}")

        G = self.get_dependency_graph()
        validate_no_cycles(G)

        # Update dependents of any changed nodes
        correct_computed_status(G)

        # Get nodes that need recomputing
        uncached_nodes = get_uncached_nodes(G)

        # Check if the result is already computed and cached and if we aren't forcing a recompute
        needs_compute = force_recompute or recompute_dependencies or uncached_nodes
        if not needs_compute and self.ddata:
            return self.ddata()

        try:
            # This Delayed object is now trying to compute it's result
            self._computing = True

            kwargs = dict(
                force_recompute=recompute_dependencies,
                dont_cache=dont_cache,
                maximum_depth=maximum_depth,
            )
            computed_args = [compute_nested(arg, **kwargs) for arg in self.args]
            computed_kwargs = {key: compute_nested(value, **kwargs) for key, value in self.kwargs.items()}
            result = self.func(*computed_args, **computed_kwargs)
            if not self.disable_cache and not dont_cache:
                self.ddata.set(result)
            return result

        except Exception as e:
            raise type(e)(f"Error in delayed computation of {self.func.__name__}: {str(e)}") from e

        finally:
            # Reset the computation flag
            self._computing = False

    def __bool__(self) -> bool:
        """Return True if the result has been computed."""
        return bool(self.ddata)

    def __repr__(self) -> str:
        """Return the string representation of the delayed computation."""
        return f"Delayed({self.func.__name__}):{'Computed' if self.ddata else 'Not computed'}"

    def __getitem__(self, key):
        """Enable dictionary-like access for delayed computations that return mappings.

        This allows accessing dictionary items before computation, returning a new
        Delayed object that will compute and access the item when needed.

        Parameters
        ----------
        key : Any
            The key to access in the resulting dictionary

        Returns
        -------
        Delayed
            A new Delayed object that will return the dictionary item
        """

        def get_item(d, k):
            return d[k]

        return Delayed(get_item, self, key)

    def __len__(self):
        """Enable len() for delayed computations that return sized objects."""

        def get_len(obj):
            return len(obj)

        return Delayed(get_len, self)

    def __iter__(self):
        """Enable iteration for delayed computations that return iterables."""

        def get_iter(obj):
            return iter(obj)

        return Delayed(get_iter, self)

    def __getattr__(self, name):
        """Enable attribute access for delayed computations."""

        def get_attr(obj, attr):
            return getattr(obj, attr)

        return Delayed(get_attr, self, name)

    def clear_all_caches(self) -> None:
        """Clear all cached results in this computation graph."""
        self.ddata.clear()
        for arg in self.args:
            if isinstance(arg, Delayed):
                arg.clear_all_caches()
        for value in self.kwargs.values():
            if isinstance(value, Delayed):
                value.clear_all_caches()

    def _get_node_id(self) -> str:
        """Generate a compact node ID."""
        name = self.func.__name__
        instance_id = str(id(self))[-4:]  # Use fewer digits

        # Create simplified arg representations
        arg_strs = []
        for arg in self.args:
            if isinstance(arg, Delayed):
                # Just use the function name and id of delayed args
                arg_strs.append(arg.func.__name__ + f"#{str(id(arg))[-4:]}")
            else:
                # For non-delayed args, use a short hash
                try:
                    arg_str = str(hash(arg))[-4:]
                except TypeError:
                    arg_str = type(arg).__name__[:4]
                arg_strs.append(arg_str)

        # Handle kwargs similarly but more concisely
        kwarg_strs = []
        for k, v in sorted(self.kwargs.items()):
            if isinstance(v, Delayed):
                kwarg_strs.append(f"{k[:4]}={v.func.__name__}")
            else:
                try:
                    v_str = str(hash(v))[-4:]
                except TypeError:
                    v_str = type(v).__name__[:4]
                kwarg_strs.append(f"{k[:4]}={v_str}")

        # Combine everything into a compact string
        content = f"{name}({','.join(arg_strs + kwarg_strs)})#{instance_id}"
        return content

    def get_dependency_graph(self) -> DiGraph:
        """Build and return a directed graph of computation dependencies.

        Returns
        -------
        DiGraph
            A directed graph where nodes are computations and edges represent dependencies.
            Node attributes include:
            - 'label': A readable description of the computation
            - 'computed': Boolean indicating if the result is cached
            - 'func_name': Name of the function
        """
        G = DiGraph()
        self._build_graph(G, set())
        return G

    def _build_graph(self, G: DiGraph, visited: Set[str]) -> None:
        """Recursively build the dependency graph.

        Parameters
        ----------
        G : DiGraph
            The graph to build
        visited : Set[str]
            Set of node IDs already processed to avoid redundant traversal
        """
        node_id = self._get_node_id()
        if node_id in visited:
            return

        visited.add(node_id)

        # Add this node to the graph
        G.add_node(node_id, label=self.func.__name__, computed=bool(self.ddata), delayed_obj=self)

        # Process arguments
        for arg in self.args:
            if isinstance(arg, Delayed):
                arg._build_graph(G, visited)
                G.add_edge(arg._get_node_id(), node_id)

        # Process keyword arguments
        for v in self.kwargs.values():
            if isinstance(v, Delayed):
                v._build_graph(G, visited)
                G.add_edge(v._get_node_id(), node_id)


def delayed(func=None, *, disable_cache=False):
    """Creates a delayed computation that executes only when explicitly evaluated.

    Parameters
    ----------
    func : callable or None
        The function to be delayed. Will be None if decorator is called with
        parameters.
    disable_cache : bool, optional
        If True, disables caching of computation results. Each call to compute()
        will re-execute the function, by default False.

    Returns
    -------
    callable or Delayed
        If used as @delayed:
            Returns a Delayed object holding the function and arguments
            for later execution.
        If used as @delayed(no_cache=...):
            Returns a decorator function that will create a Delayed object.

    See Also
    --------
    Delayed : The class that handles lazy evaluation of functions

    Examples
    --------
    Basic usage with default caching:

    >>> @delayed
    ... def expensive_computation(x):
    ...     return x * 2
    ...
    >>> result = expensive_computation(10)  # No computation yet
    >>> result.compute()  # Now computes
    20

    Disable caching for always-fresh results:

    >>> @delayed(no_cache=True)
    ... def always_recompute(x):
    ...     return x * 2
    ...
    >>> result = always_recompute(10)
    >>> result.compute()  # Computes without caching
    20

    Notes
    -----
    The decorated function's computation is deferred until the .compute()
    method is called on the returned Delayed object. By default, results
    are cached based on input arguments unless no_cache=True.
    """
    # Called as @delayed(disable_cache=...)
    if func is None:
        return lambda f: delayed(f, disable_cache=disable_cache)

    # Called as @delayed or delayed(func, ...)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return Delayed(func, *args, **kwargs, disable_cache=disable_cache)

    return wrapper
