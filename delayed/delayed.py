from warnings import warn
from typing import Any, Tuple, Dict, Callable, Set
from collections.abc import Mapping, Sequence
import numpy as np
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt

from .types import DelayedData, GenericType, T


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
    def __init__(self, func: Callable[..., T], *args: Tuple[Any, ...], cache_data: bool = True, **kwargs: Dict[str, Any]):
        self._func = func
        self._args = tuple(args)
        self._kwargs = kwargs
        self.cache_data = cache_data
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
        self.validate_no_cycles(G)

        # Update dependents of any changed nodes
        self._correct_computed_status(G)

        # Get nodes that need recomputing
        uncached_nodes = self.get_uncached_nodes(G)

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
            if self.cache_data and not dont_cache:
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

    def get_computed_nodes(self, G: DiGraph = None) -> Set[str]:
        """Return set of node IDs that have been computed."""
        G = G or self.get_dependency_graph()
        return {node for node in G.nodes if G.nodes[node].get("computed", False)}

    def get_uncached_nodes(self, G: DiGraph = None) -> Set[str]:
        G = G or self.get_dependency_graph()
        return {node for node in G.nodes if not G.nodes[node].get("computed", False)}

    def _correct_computed_status(self, G: DiGraph = None) -> None:
        """Clear data from nodes that depend on any node that needs recomputing."""
        G = G or self.get_dependency_graph()
        uncached = self.get_uncached_nodes(G)
        dependents = set()
        to_process = set(uncached)
        while to_process:
            node = to_process.pop()  # remove and return an arbitrary element from the set
            successors = set(G.successors(node))  # get all nodes that depend on this node
            new_dependents = successors - dependents  # get the nodes that haven't been processed yet
            dependents.update(new_dependents)  # add them to list of dependents
            to_process.update(new_dependents)  # add them to the list of nodes to process

        # Clear data from dependent nodes so they'll be recomputed
        for node in dependents:
            G.nodes[node]["delayed_obj"].ddata.clear()

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze the computation graph's dependencies and structure.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing analysis metrics including:
            - depth: Maximum depth of the computation graph
            - n_nodes: Total number of computation nodes
            - n_edges: Total number of dependencies
            - leaf_nodes: Number of nodes with no dependencies
            - root_nodes: Number of nodes with no dependents
            - is_cyclic: Whether the graph contains cycles
            - max_in_degree: Maximum number of direct dependencies for any node
            - max_out_degree: Maximum number of direct dependents for any node
        """
        G = self.get_dependency_graph()

        # Get root nodes (those with no predecessors)
        root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]

        # Get leaf nodes (those with no successors)
        leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]

        # Calculate maximum depth (longest path from any root to any leaf)
        max_depth = 0
        for root in root_nodes:
            for leaf in leaf_nodes:
                try:
                    path_length = len(nx.shortest_path(G, root, leaf)) - 1
                    max_depth = max(max_depth, path_length)
                except nx.NetworkXNoPath:
                    continue

        return {
            "depth": max_depth,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "leaf_nodes": len(leaf_nodes),
            "root_nodes": len(root_nodes),
            "is_cyclic": not nx.is_directed_acyclic_graph(G),
            "max_in_degree": max(dict(G.in_degree()).values(), default=0),
            "max_out_degree": max(dict(G.out_degree()).values(), default=0),
        }

    def validate_no_cycles(self, G: DiGraph = None) -> None:
        """Validate that the computation graph has no cycles.

        Raises
        ------
        ValueError
            If cycles are detected in the dependency graph.
        """
        G = G or self.get_dependency_graph()
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            cycle_str = " -> ".join(cycles[0])  # Show first cycle
            raise ValueError(f"Circular dependency detected in computation graph: {cycle_str}")

    def get_optimized_pos(self, G, scale=1.0):
        """Get optimized node positions with proper scaling."""
        # Get base positions using hierarchical layout
        generations = list(nx.topological_generations(G))

        pos = {}
        y_step = 1.0 / (len(generations) + 1)
        for i, gen in enumerate(generations):
            y = 1 - y_step * (i + 1)
            x_step = 1.0 / (len(gen) + 1)
            for j, node in enumerate(sorted(gen)):
                x = x_step * (j + 1)
                pos[node] = (x, y)

        # Fine-tune with spring layout, using hierarchical as starting point
        pos = nx.spring_layout(G, k=2, iterations=50, pos=pos, fixed=None if scale != 1.0 else pos.keys())

        # Scale positions
        return {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

    def visualize(self, figsize=(8, 7), scale=1.0, jitter=0.0):
        G = self.get_dependency_graph()
        plt.figure(figsize=figsize)

        # Get optimized positions
        pos = self.get_optimized_pos(G, scale=scale)
        pos = {node: (x + np.random.uniform(-jitter, jitter), y + np.random.uniform(-jitter, jitter)) for node, (x, y) in pos.items()}

        # Draw with more spacing
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=1000,
            font_size=12,
            font_weight="bold",
            arrows=True,
            edge_color="gray",
            arrowsize=12,
            # Add minimum spacing between nodes
            min_target_margin=10,
            min_source_margin=10,
        )


def delayed(func=None, *, cache_data=True):
    """Decorator to create a delayed computation"""
    if func is None:  # Called as @delayed(cache=...)
        return lambda f: delayed(f, cache_data=cache_data)

    def wrapper(*args, **kwargs):
        return Delayed(func, *args, **kwargs, cache_data=cache_data)

    return wrapper
