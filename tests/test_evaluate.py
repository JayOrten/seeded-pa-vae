import networkx as nx

from svae.evaluate import (
    graph_statistics_experiment,
    structural_invariants_experiment,
)
from svae.simulate import simulate_graph


def test_graph_statistics_match_the_original_report_shape():
    collections = {
        "NetworkX": [simulate_graph(12, seed) for seed in range(32)],
        "VAE": [simulate_graph(12, 100 + seed) for seed in range(32)],
        "Independent": [simulate_graph(12, 200 + seed) for seed in range(32)],
    }
    result = graph_statistics_experiment(
        collections, num_nodes=12, bootstrap_samples=10, show=False
    )

    assert set(result) == {"hub_leaf_summary", "reinforcement"}
    assert set(result["hub_leaf_summary"]) == set(collections)
    assert set(result["reinforcement"]) == set(collections)
    assert all(nx.is_tree(graph) for graphs in collections.values() for graph in graphs)


def test_structural_invariants_accept_valid_ba_trees():
    collections = {
        "NetworkX": [simulate_graph(12, seed) for seed in range(4)],
        "VAE": [simulate_graph(12, 100 + seed) for seed in range(4)],
    }

    result = structural_invariants_experiment(
        collections, num_nodes=12, show=False
    )

    assert result["all_passed"] is True
    assert result["collections"]["NetworkX"]["passed"] == 4
    assert result["collections"]["VAE"]["failures"] == []


def test_structural_invariants_report_each_failure_and_graph_index():
    invalid_graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    invalid_graph.add_node(4)

    result = structural_invariants_experiment(
        {"candidate": [invalid_graph]}, num_nodes=4, show=False
    )

    summary = result["collections"]["candidate"]
    assert result["all_passed"] is False
    assert summary["failed"] == 1
    assert summary["failures"] == [
        {
            "graph_index": 0,
            "failed_invariants": (
                "connected",
                "acyclic",
                "one_older_parent",
                "zero_triangles",
            ),
        }
    ]
