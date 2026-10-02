import numpy as np
import pytest

from dla.simulate import make_dataset, parents_to_graph, rebuild_positions, simulate_dla, validate_dla


def test_same_seed_repeats_and_different_seeds_differ():
    first = simulate_dla(64, 3)
    again = simulate_dla(64, 3)
    other = simulate_dla(64, 4)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_simulated_clusters_are_valid_and_collision_free():
    for seed in range(8):
        parents, directions = simulate_dla(128, seed)
        validate_dla(parents, directions)
        positions, collisions = rebuild_positions(parents, directions)
        assert collisions == 0
        steps = np.abs(positions[1:] - positions[parents[1:] - 1]).sum(axis=1)
        assert np.all(steps == 1)


def test_make_dataset_matches_serial_simulation():
    parents, directions = make_dataset(32, 6, first_seed=10, processes=2)
    assert parents.shape == directions.shape == (6, 32)
    for offset in range(6):
        expected_parents, expected_directions = simulate_dla(32, 10 + offset)
        assert np.array_equal(parents[offset].numpy(), expected_parents)
        assert np.array_equal(directions[offset].numpy(), expected_directions)


def test_rebuild_counts_collisions():
    # Node 3 sits on node 1's +x side; node 4 steps back from node 3 onto node 1.
    parents = np.array([-1, 1, 1, 3])
    directions = np.array([-1, 2, 0, 1])
    _, collisions = rebuild_positions(parents, directions)
    assert collisions == 1
    with pytest.raises(ValueError):
        parents_to_graph(parents, directions)


@pytest.mark.parametrize(
    "parents, directions",
    [
        (np.array([-1, 2]), np.array([-1, 0])),  # parent not older
        (np.array([-1, 1]), np.array([-1, 4])),  # direction out of range
        (np.array([0, 1]), np.array([-1, 0])),  # root parent not -1
        (np.array([-1, 1, 1]), np.array([-1, 0])),  # length mismatch
    ],
)
def test_invalid_inputs_are_rejected(parents, directions):
    with pytest.raises(ValueError):
        validate_dla(parents, directions)
