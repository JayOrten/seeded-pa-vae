import numpy as np
import torch

from svae.generate import IndependentParentGenerator, SeededPAGenerator
from svae.train import QueryVAE


def test_seeded_generator_is_deterministic_and_query_order_independent():
    model = QueryVAE(num_nodes=10, latent_dim=4, hidden_dim=8)
    generator = SeededPAGenerator.from_model(model)
    queries = list(range(1, 11))
    expected = generator.parents(123, queries)

    for order in [queries, queries[::-1], [5, 3, 5, 9, 3]]:
        assert generator.parents(123, order) == [expected[query - 1] for query in order]
    assert generator.parent_array(123).tolist() == [-1, *expected[1:]]
    assert generator.graph(123).number_of_edges() == 9


def test_independent_baseline_probabilities_are_normalized():
    training = torch.tensor([
        [-1, 1, 1, 2, 2],
        [-1, 1, 2, 1, 3],
    ])
    baseline = IndependentParentGenerator(training)
    assert all(np.isclose(probabilities.sum(), 1) for probabilities in baseline.probabilities.values())
    assert baseline.graph(42).number_of_edges() == 4
