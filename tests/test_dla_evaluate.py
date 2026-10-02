import numpy as np

from dla.config import Config
from dla.evaluate import (
    collision_experiment,
    consistency_experiment,
    fractal_dimension_experiment,
    reconstruction_experiment,
    samples_experiment,
    timing_experiment,
    tree_statistics_experiment,
)
from dla.generate import generators_from_run
from dla.simulate import simulate_dla
from dla.train import train


def _simulated(num_nodes, seeds):
    return [simulate_dla(num_nodes, seed) for seed in seeds]


def test_simulator_trees_have_no_collisions_and_a_dla_like_dimension():
    collections = {"Simulator": _simulated(512, range(20))}
    collisions = collision_experiment(collections, show=False)
    dimension = fractal_dimension_experiment(collections, fit_min=32, show=False)

    assert collisions["Simulator"]["mean_collision_fraction"] == 0
    assert collisions["Simulator"]["median_first_collision_node"] is None
    assert 1.4 < dimension["Simulator"]["fractal_dimension"] < 2.0


def test_collisions_are_counted_per_node():
    # Node 4 steps back onto node 1's square.
    tree = (np.array([-1, 1, 1, 3]), np.array([-1, 2, 0, 1]))
    result = collision_experiment({"candidate": [tree]}, show=False)["candidate"]
    assert np.isclose(result["mean_collision_fraction"], 1 / 3)
    assert result["median_first_collision_node"] == 4


def test_tree_statistics_on_a_known_tree():
    # A path 1-2-3 with node 4 also on node 1: depths 0, 1, 2, 1.
    tree = (np.array([-1, 1, 2, 1]), np.array([-1, 0, 0, 1]))
    result = tree_statistics_experiment({"candidate": [tree]}, show=False)["candidate"]
    assert result["mean_depth"] == 1.0
    assert result["max_depth"] == 2
    assert result["leaf_fraction"] == 0.5
    assert result["max_children"] == 2


def test_full_evaluation_workflow(tmp_path):
    config = Config(
        profile="test", num_nodes=16, latent_dim=3, hidden=12,
        train_graphs=16, val_graphs=8, test_graphs=8, generated_graphs=4,
        batch_size=8, epochs=1, warmup_epochs=1, threads=1,
    )
    run = train(config, device="cpu", checkpoint_dir=tmp_path, report=None, show=False)
    generators = generators_from_run(run)

    reconstruction = reconstruction_experiment(run, show=False)
    collections = samples_experiment(
        run, generators["vae"], generators["independent"], show=False
    )
    collision_experiment(collections, show=False)
    fractal_dimension_experiment(collections, fit_min=4, show=False)
    tree_statistics_experiment(collections, show=False)
    consistency = consistency_experiment(generators["vae"], seeds=range(2), show=False)
    timing = timing_experiment(
        generators["vae"], num_nodes=16, threads=1, repeats=2, show=False
    )

    assert set(collections) == {"Simulator", "VAE", "Independent"}
    assert len(collections["VAE"]) == 4
    assert 0 <= reconstruction["mean_latent_test_direction_accuracy"] <= 1
    assert consistency["all_passed"] is True
    assert [row["arrival"] for row in timing["per_arrival"]] == [2, 8, 16]
