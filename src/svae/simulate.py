"""Preferential-attachment simulation and parent-array conversion."""

from collections.abc import Iterable

import networkx as nx
import numpy as np
import torch


def simulate_graph(num_nodes: int, simulator_seed: int) -> nx.Graph:
    """Simulate one arrival-labelled BA tree with the project's fixed initial edge."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if isinstance(simulator_seed, bool) or not isinstance(simulator_seed, (int, np.integer)):
        raise ValueError("simulator_seed must be an integer")
    graph = nx.barabasi_albert_graph(
        num_nodes, m=1, seed=int(simulator_seed), initial_graph=nx.Graph([(0, 1)])
    )
    return nx.relabel_nodes(graph, {node: node + 1 for node in range(num_nodes)})


def simulate_parents(num_nodes: int, simulator_seed: int) -> np.ndarray:
    """Simulate one BA tree and return its arrival-labelled parent array."""
    return graph_to_parents(simulate_graph(num_nodes, simulator_seed))


def make_parent_dataset(num_nodes: int, num_graphs: int, *, first_seed: int) -> torch.Tensor:
    """Create a reproducible tensor of parent arrays from consecutive seeds."""
    if num_graphs < 1:
        raise ValueError("num_graphs must be positive")
    parents = [simulate_parents(num_nodes, first_seed + offset) for offset in range(num_graphs)]
    return torch.from_numpy(np.stack(parents))


def validate_parents(parents: Iterable[int]) -> np.ndarray:
    """Validate and return a one-dimensional arrival-labelled parent array."""
    array = np.asarray(parents)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("parents must be a one-dimensional array of length at least 2")
    if array[0] != -1 or array[1] != 1:
        raise ValueError("parents must begin with the root sentinel [-1, 1]")
    for node in range(3, len(array) + 1):
        parent = array[node - 1]
        if not isinstance(parent, np.integer) and not isinstance(parent, int):
            raise ValueError(f"node {node} must have an integer parent")
        if not 1 <= int(parent) < node:
            raise ValueError(f"node {node} must have a parent in 1..{node - 1}")
    return array.astype(np.int64, copy=False)


def parents_to_graph(parents: Iterable[int], through_node: int | None = None) -> nx.Graph:
    """Build a full graph or arrival prefix from a parent array."""
    array = validate_parents(parents)
    through_node = len(array) if through_node is None else int(through_node)
    if not 1 <= through_node <= len(array):
        raise ValueError(f"through_node must be in 1..{len(array)}")
    graph = nx.Graph()
    graph.add_nodes_from(range(1, through_node + 1))
    graph.add_edges_from((node, int(array[node - 1])) for node in range(2, through_node + 1))
    return graph


def graph_to_parents(graph: nx.Graph) -> np.ndarray:
    """Convert an arrival-labelled tree on nodes ``1..N`` to parent form."""
    num_nodes = len(graph)
    if set(graph) != set(range(1, num_nodes + 1)):
        raise ValueError("graph nodes must be consecutive arrival IDs 1..N")
    if num_nodes < 2 or not nx.is_tree(graph):
        raise ValueError("graph must be a tree with at least two nodes")
    parents = np.full(num_nodes, -1, dtype=np.int64)
    parents[1] = 1
    for node in range(3, num_nodes + 1):
        older_neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor < node]
        if len(older_neighbors) != 1:
            raise ValueError(f"node {node} must have exactly one older neighbor")
        parents[node - 1] = older_neighbors[0]
    return validate_parents(parents)
