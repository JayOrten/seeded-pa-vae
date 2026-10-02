import numpy as np
import torch

from dla.generate import IndependentDLAGenerator, SeededDLAGenerator
from dla.simulate import make_dataset
from dla.train import DLAQueryVAE


def test_seeded_generator_is_deterministic_and_query_order_independent():
    model = DLAQueryVAE(num_nodes=10, latent_dim=4, hidden_dim=8)
    generator = SeededDLAGenerator.from_model(model)
    queries = list(range(1, 11))
    expected = generator.queries(123, queries)

    for order in [queries, queries[::-1], [5, 3, 5, 9, 3]]:
        assert generator.queries(123, order) == [expected[t - 1] for t in order]
    assert expected[0] == (None, None)
    assert expected[1][0] == 1  # node 2 always attaches to node 1

    parents, directions = generator.arrays(123)
    assert parents.tolist() == [-1, *(parent for parent, _ in expected[1:])]
    assert directions.tolist() == [-1, *(direction for _, direction in expected[1:])]
    positions, collisions = generator.positions(123)
    assert positions.shape == (10, 2) and 0 <= collisions < 10


def test_seeded_generator_probabilities_are_normalized():
    generator = SeededDLAGenerator.from_model(DLAQueryVAE(num_nodes=8, latent_dim=3, hidden_dim=8))
    for t in range(2, 9):
        parent_probabilities, direction_probabilities = generator.probabilities(7, t)
        assert len(parent_probabilities) == t - 1 and np.isclose(parent_probabilities.sum(), 1)
        assert len(direction_probabilities) == 4 and np.isclose(direction_probabilities.sum(), 1)


def test_independent_baseline_probabilities_are_normalized():
    parents, directions = make_dataset(12, 20, first_seed=0, processes=1)
    baseline = IndependentDLAGenerator(parents, directions)
    assert all(np.isclose(p.sum(), 1) for p in baseline.parent_probabilities.values())
    assert all(np.isclose(p.sum(), 1) for p in baseline.direction_probabilities.values())
    generated_parents, generated_directions = baseline.arrays(42)
    assert generated_parents.shape == generated_directions.shape == (12,)
    assert baseline.arrays(42)[1].tolist() == generated_directions.tolist()
