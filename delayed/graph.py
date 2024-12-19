from typing import Any, Dict, Set, Tuple
import numpy as np
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt


def get_computed_nodes(G: DiGraph) -> Set[str]:
    """Return set of node IDs that have been computed."""
    return {node for node in G.nodes if G.nodes[node].get("computed", False)}


def get_uncached_nodes(G: DiGraph) -> Set[str]:
    return {node for node in G.nodes if not G.nodes[node].get("computed", False)}


def correct_computed_status(G: DiGraph) -> None:
    """Clear data from nodes that depend on any node that needs recomputing."""
    uncached = get_uncached_nodes(G)
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


def analyze_dependencies(G: DiGraph) -> Dict[str, Any]:
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


def validate_no_cycles(G: DiGraph) -> None:
    """Validate that the computation graph has no cycles.

    Raises
    ------
    ValueError
        If cycles are detected in the dependency graph.
    """
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        cycle_str = " -> ".join(cycles[0])  # Show first cycle
        raise ValueError(f"Circular dependency detected in computation graph: {cycle_str}")


def get_optimized_pos(G: DiGraph, scale: float = 1.0):
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


def visualize(G: DiGraph, figsize: Tuple[int] = (8, 7), scale: float = 1.0, jitter: float = 0.0):
    plt.figure(figsize=figsize)

    # Get optimized positions
    pos = get_optimized_pos(G, scale=scale)
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
