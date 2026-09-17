"""Exact examples and small end-to-end checks for the evaluation plan."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from svae import evaluate as ev
from svae.config import PROFILES
from svae.generate import SeededPAGenerator, IndependentParentGenerator
from svae.simulate import parents_to_graph, simulate_graph
from svae.train import train


@pytest.fixture
def collections():
    graphs = [simulate_graph(12, seed) for seed in range(24)]
    return {"NetworkX": graphs, "VAE": graphs}


def test_plan_likelihood_worked_example():
    graph = parents_to_graph([-1, 1, 1, 1, 3, 1])
    result = ev.ba_sequence_likelihood_experiment({"NetworkX": [graph]}, show=False)
    assert result["NetworkX"]["values"][0] == pytest.approx(np.log(1 / 64))


def test_finite_recurrence_conserves_nodes_and_degree(collections):
    row = ev.finite_size_degree_counts_experiment(
        collections, num_nodes=12, bootstrap_samples=5, show=False
    )["NetworkX"]
    assert row["expected"].sum() == pytest.approx(12)
    assert row["expected"] @ np.arange(12) == pytest.approx(22)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("ba_sequence_likelihood_experiment", {}),
        ("attachment_kernel_experiment", {"bootstrap_samples": 5}),
        ("selected_parent_degree_experiment", {}),
        ("degree_distribution_experiment", {}),
        (
            "finite_size_degree_counts_experiment",
            {"num_nodes": 12, "bootstrap_samples": 5},
        ),
        ("degree_by_age_experiment", {"early_nodes": 3}),
        ("reinforcement_experiment", {"node_ids": [1, 2, 3], "bootstrap_samples": 5}),
        ("conditional_future_growth_experiment", {"node_ids": [1, 2]}),
        ("attachment_residuals_experiment", {}),
        ("hub_dynamics_experiment", {}),
        (
            "early_node_covariance_experiment",
            {"node_ids": [1, 2], "bootstrap_samples": 5},
        ),
        (
            "joint_attachment_experiment",
            {"arrival_pairs": [(8, 12)], "candidate_parents": [1, 2]},
        ),
        ("hub_concentration_experiment", {}),
        ("tree_geometry_experiment", {}),
        ("edge_degree_correlation_experiment", {}),
        ("distribution_comparison_experiment", {"cross_validation_folds": 2}),
    ],
)
def test_graph_experiments_display_and_return_data(collections, name, kwargs):
    result = getattr(ev, name)(collections, **kwargs)
    assert result
    plt.close("all")


def test_reference_distances_and_calibration_mass(collections):
    assert (
        ev.degree_distribution_experiment(collections, show=False)["VAE"][
            "total_variation"
        ]
        == 0
    )
    row = ev.selected_parent_degree_experiment(collections, show=False)["VAE"]
    assert row["observed"].sum() == 240
    assert row["expected"].sum() == pytest.approx(240)
    row = ev.distribution_comparison_experiment(
        collections, cross_validation_folds=2, show=False
    )["VAE"]
    assert row["energy_distance_squared"] == pytest.approx(0)


def test_latent_and_scaling_experiments():
    config = replace(
        PROFILES["smoke"], num_nodes=12, generated_graphs=16, test_graphs=16
    )
    run = train(config, checkpoint_dir=None, report=None, show=False)
    generator = SeededPAGenerator.from_model(run.model)
    assert (
        generator.graph(10).edges
        == generator.diagnostic_graph(generator.latent(10), 10).edges
    )
    entropy = ev.decoder_entropy_experiment(generator, seeds=range(4))
    assert all(np.all(row["entropy"] >= 0) for row in entropy.values())
    ablation = ev.latent_ablation_experiment(
        generator,
        seeds=range(4),
        samples_per_latent=2,
        baseline=IndependentParentGenerator(run.data.train),
    )
    assert "Independent" in ablation["collections"]
    assert ablation["replicate_features"].shape == (4, 2, 6)
    assert ev.latent_predictability_experiment(run)
    scaling = ev.scaling_experiment([config], training_seeds=[1], bootstrap_samples=3)
    assert len(scaling["runs"]) == 1
    plt.close("all")
