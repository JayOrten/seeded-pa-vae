import networkx as nx
import numpy as np
import pytest

from svae.simulate import graph_to_parents, parents_to_graph, simulate_graph, validate_parents


def test_simulation_round_trips_through_parent_array():
    for seed in range(8):
        graph = simulate_graph(16, seed)
        reconstructed = parents_to_graph(graph_to_parents(graph))
        assert nx.utils.graphs_equal(graph, reconstructed)


@pytest.mark.parametrize(
    "parents",
    [np.array([-1]), np.array([0, 1]), np.array([-1, 1, 3]), np.array([-1, 1, 1.5])],
)
def test_invalid_parent_arrays_are_rejected(parents):
    with pytest.raises(ValueError):
        validate_parents(parents)


def test_parent_prefix_contains_only_requested_arrivals():
    graph = parents_to_graph(np.array([-1, 1, 1, 2, 2]), through_node=3)
    assert set(graph) == {1, 2, 3}
    assert set(map(frozenset, graph.edges)) == {frozenset((1, 2)), frozenset((1, 3))}
