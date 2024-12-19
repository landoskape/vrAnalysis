from typing import Dict, Set, Optional
import networkx as nx
from networkx import DiGraph
import weakref
from .delayed import Delayed


class MasterGraph:
    """Maintains a master graph of all delayed computation dependencies."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.graph = DiGraph()
        self._delayed_objects: Dict[str, weakref.ref[Delayed]] = {}

    def register_node(self, delayed_obj: Delayed) -> None:
        """Register a new delayed computation object."""
        node_id = delayed_obj._get_node_id()
        self._delayed_objects[node_id] = weakref.ref(delayed_obj)
        self._update_subgraph(delayed_obj)

    def _update_subgraph(self, delayed_obj: Delayed) -> None:
        """Update the subgraph for a given delayed object."""
        node_id = delayed_obj._get_node_id()
        self.graph.add_node(node_id, label=delayed_obj.func.__name__, computed=bool(delayed_obj.ddata), func_name=delayed_obj.func.__name__)

        # Remove existing edges for this node (to reset them in case of changes)
        self.graph.remove_edges_from(list(self.graph.in_edges(node_id)) + list(self.graph.out_edges(node_id)))

        # Add dependencies on arguments
        for arg in delayed_obj.args:
            if isinstance(arg, Delayed):
                self._add_dependency(arg, node_id)
                arg_id = arg._get_node_id()
                self.graph.add_edge(arg_id, node_id)
                if arg_id not in self._delayed_objects:
                    self.register_node(arg)

        # Add dependencies on keyword arguments
        for value in delayed_obj.kwargs.values():
            if isinstance(value, Delayed):
                value_id = value._get_node_id()
                self.graph.add_edge(value_id, node_id)
                if value_id not in self._delayed_objects:
                    self.register_node(value)

    def _add_dependency(self, source_node: str, target_id: Delayed) -> None:
        """Add a dependency between two nodes."""
        source_id = source_node._get_node_id()
        self.graph.add_edge(source_id, target_id)
        if source_id not in self._delayed_objects:
            self.register_node(target_node)

    def _update_subgraph(self, delayed_obj: Delayed) -> None:
        """Update the subgraph for a given delayed object."""
        node_id = delayed_obj._get_node_id()

        # Remove existing edges for this node
        self.graph.remove_edges_from(list(self.graph.in_edges(node_id)) + list(self.graph.out_edges(node_id)))

        # Update node properties
        self.graph.add_node(node_id, label=delayed_obj.func.__name__, computed=bool(delayed_obj.ddata), func_name=delayed_obj.func.__name__)

        # Add edges for arguments
        for arg in delayed_obj.args:
            if isinstance(arg, Delayed):
                arg_id = arg._get_node_id()
                self.graph.add_edge(arg_id, node_id)
                # Recursively update if not already in graph
                if arg_id not in self._delayed_objects:
                    self.register_delayed(arg)

        # Add edges for keyword arguments
        for value in delayed_obj.kwargs.values():
            if isinstance(value, Delayed):
                value_id = value._get_node_id()
                self.graph.add_edge(value_id, node_id)
                # Recursively update if not already in graph
                if value_id not in self._delayed_objects:
                    self.register_delayed(value)

    def cleanup_stale_references(self) -> None:
        """Remove references to delayed objects that have been garbage collected."""
        stale_nodes = set()
        for node_id, delayed_ref in self._delayed_objects.items():
            if delayed_ref() is None:  # Object has been garbage collected
                stale_nodes.add(node_id)

        # Remove stale nodes and their references
        for node_id in stale_nodes:
            self.graph.remove_node(node_id)
            del self._delayed_objects[node_id]

    def get_affected_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes that depend on the given node."""
        return set(nx.descendants(self.graph, node_id))

    def propagate_recomputation(self, node_id: str) -> None:
        """Mark all dependent nodes as needing recomputation."""
        affected = self.get_affected_nodes(node_id)
        for affected_id in affected:
            delayed_ref = self._delayed_objects.get(affected_id)
            if delayed_ref is not None:
                delayed_obj = delayed_ref()
                if delayed_obj is not None:
                    delayed_obj.ddata.clear()

    def visualize_subgraph(self, root_node_id: str, depth: Optional[int] = None):
        """Visualize a subgraph starting from the given node up to specified depth."""
        if depth is None:
            nodes = {root_node_id} | self.get_affected_nodes(root_node_id)
        else:
            nodes = set()
            current_nodes = {root_node_id}
            for _ in range(depth + 1):
                nodes.update(current_nodes)
                next_nodes = set()
                for node in current_nodes:
                    next_nodes.update(self.graph.successors(node))
                current_nodes = next_nodes

        subgraph = self.graph.subgraph(nodes)
        delayed_ref = self._delayed_objects.get(root_node_id)
        if delayed_ref is not None:
            delayed_obj = delayed_ref()
            if delayed_obj is not None:
                delayed_obj.visualize(G=subgraph)
