"""Notebook-facing experiments for evaluating seeded DLA generators.

Each experiment owns its calculation, printed summary, and visualization, as in
``svae.evaluate``. Collections map a name to a list of ``(parents, directions)``
array pairs, so the simulator, the VAE, and the baseline are evaluated the same way.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import torch
from torch.nn import functional

from .generate import DLAGenerator
from .simulate import rebuild_positions, simulate_dla
from .train import NUM_DIRECTIONS, TrainingRun

Tree = tuple[np.ndarray, np.ndarray]
Collections = Mapping[str, Sequence[Tree]]


def reconstruction_experiment(
    run: TrainingRun, *, batch_size: int = 256, show: bool = True
) -> dict[str, object]:
    """Report mean-latent parent and direction reconstruction on the held-out test split."""
    model = run.model
    model.eval()
    test_parents_all, test_directions_all = run.data.test
    queries = torch.arange(2, run.config.num_nodes + 1, device=run.device)
    totals = {"parent_ce": 0.0, "parent_correct": 0.0, "direction_ce": 0.0, "direction_correct": 0.0}
    parent_predictions = direction_predictions = 0
    with torch.no_grad():
        for start in range(0, len(test_parents_all), batch_size):
            parents = test_parents_all[start : start + batch_size].to(run.device)
            directions = test_directions_all[start : start + batch_size].to(run.device)
            latent_mean, _ = model.encode(parents, directions)
            _, masked_parent_logits, _, direction_logits = model.decode(latent_mean, queries)
            # Node 2's parent is fixed, so parent metrics start at node 3.
            parent_logits = masked_parent_logits[:, 1:]
            parent_targets = parents[:, 2:] - 1
            direction_targets = directions[:, 1:]
            totals["parent_ce"] += functional.cross_entropy(
                parent_logits.flatten(0, 1), parent_targets.flatten(), reduction="sum"
            ).item()
            totals["parent_correct"] += (parent_logits.argmax(-1) == parent_targets).sum().item()
            totals["direction_ce"] += functional.cross_entropy(
                direction_logits.flatten(0, 1), direction_targets.flatten(), reduction="sum"
            ).item()
            totals["direction_correct"] += (
                direction_logits.argmax(-1) == direction_targets
            ).sum().item()
            parent_predictions += parent_targets.numel()
            direction_predictions += direction_targets.numel()

    result = {
        "mean_latent_test_parent_accuracy": totals["parent_correct"] / parent_predictions,
        "mean_latent_test_parent_cross_entropy": totals["parent_ce"] / parent_predictions,
        "mean_latent_test_direction_accuracy": totals["direction_correct"] / direction_predictions,
        "mean_latent_test_direction_cross_entropy": totals["direction_ce"] / direction_predictions,
        "uniform_direction_cross_entropy": float(np.log(NUM_DIRECTIONS)),
        "note": "reconstruction diagnostic; not fresh generation or marginal likelihood",
    }
    if show:
        print(result)
    return result


def samples_experiment(
    run: TrainingRun,
    generator: DLAGenerator,
    baseline: DLAGenerator,
    *,
    seeds: Iterable[int] | None = None,
    show: bool = True,
) -> dict[str, list[Tree]]:
    """Collect simulator test trees and generated trees, and plot three of each.

    Generated nodes that land on an occupied square are drawn in red.
    """
    if seeds is None:
        seeds = range(3_000_000, 3_000_000 + run.config.generated_graphs)
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one generation seed is required")

    test_parents, test_directions = run.data.test
    collections = {
        "Simulator": [
            (parents.numpy(), directions.numpy())
            for parents, directions in zip(test_parents, test_directions)
        ],
        "VAE": [generator.arrays(seed) for seed in seeds],
        "Independent": [baseline.arrays(seed) for seed in seeds],
    }
    if show:
        figure, axes = plt.subplots(3, 3, figsize=(10, 10))
        for row_index, (name, trees) in enumerate(collections.items()):
            for column_index, tree in enumerate(trees[:3]):
                plot_tree(*tree, ax=axes[row_index, column_index])
                axes[row_index, column_index].set_title(f"{name} sample {column_index}")
        plt.tight_layout()
        plt.show()
    return collections


def plot_tree(parents: np.ndarray, directions: np.ndarray, ax=None):
    """Draw a tree on the grid: edges in gray, nodes colored by arrival, collisions in red."""
    ax = ax or plt.gca()
    positions, _ = rebuild_positions(parents, directions)
    collided = _collision_mask(positions)
    for index in range(1, len(parents)):
        start, end = positions[parents[index] - 1], positions[index]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="0.75", lw=0.8, zorder=1)
    ax.scatter(
        positions[~collided, 0], positions[~collided, 1],
        c=np.arange(len(parents))[~collided], cmap="viridis", s=10, zorder=2,
    )
    if collided.any():
        ax.scatter(positions[collided, 0], positions[collided, 1], color="red", s=18, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def collision_experiment(collections: Collections, *, show: bool = True) -> dict[str, object]:
    """Measure how often rebuilt nodes land on an occupied square.

    The simulator never collides, so this is the most direct check of whether
    independent queries agree on one geometry.
    """
    result = {}
    curves = {}
    for name, trees in _nonempty(collections).items():
        masks = np.stack([_collision_mask(rebuild_positions(*tree)[0]) for tree in trees])
        per_graph = masks[:, 1:].mean(1)
        first = [int(np.argmax(mask)) + 1 for mask in masks if mask.any()]
        result[name] = {
            "mean_collision_fraction": float(per_graph.mean()),
            "collision_free_graph_fraction": float((per_graph == 0).mean()),
            "median_first_collision_node": float(np.median(first)) if first else None,
        }
        curves[name] = masks.mean(0)
    if show:
        print(f"{'collection':<14} {'collision fraction':>19} {'collision-free graphs':>22} "
              f"{'median first collision':>23}")
        for name, row in result.items():
            first = row["median_first_collision_node"]
            print(f"{name:<14} {row['mean_collision_fraction']:>19.4f} "
                  f"{row['collision_free_graph_fraction']:>22.3f} "
                  f"{'none' if first is None else f'{first:.0f}':>23}")
        figure, axis = plt.subplots(figsize=(6, 3.5))
        for name, curve in curves.items():
            axis.plot(np.arange(1, len(curve) + 1), curve, label=name, lw=1.5)
        axis.set_xlabel("arrival ID")
        axis.set_ylabel("collision rate")
        axis.set_title("Collision rate by arrival")
        axis.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    return result


def fractal_dimension_experiment(
    collections: Collections, *, fit_min: int = 10, show: bool = True
) -> dict[str, object]:
    """Estimate fractal dimension from the radius of gyration of arrival prefixes.

    For each prefix size k, the radius of gyration is the RMS distance of the first k
    nodes from their mean position. It grows like k^(1/D). D is about 1.71 for 2D DLA
    and is noisy below a few thousand nodes. Generated trees with collisions are
    measured as rebuilt, overlaps included.
    """
    result = {}
    curves = {}
    for name, trees in _nonempty(collections).items():
        radii = np.stack([_prefix_gyration_radius(rebuild_positions(*tree)[0]) for tree in trees])
        sizes = np.arange(1, radii.shape[1] + 1)
        mean_radius = radii.mean(0)
        fit = sizes >= fit_min
        if fit.sum() < 3:
            raise ValueError(f"need at least 3 prefix sizes >= fit_min={fit_min}")
        slope = np.polyfit(np.log(sizes[fit]), np.log(mean_radius[fit]), 1)[0]
        result[name] = {"slope": float(slope), "fractal_dimension": float(1 / slope)}
        curves[name] = (sizes, mean_radius)
    if show:
        print(f"{'collection':<14} {'slope':>7} {'dimension':>10}   (DLA reference ≈ 1.71)")
        for name, row in result.items():
            print(f"{name:<14} {row['slope']:>7.3f} {row['fractal_dimension']:>10.3f}")
        figure, axis = plt.subplots(figsize=(5.5, 4))
        for name, (sizes, radius) in curves.items():
            axis.loglog(sizes[1:], radius[1:], label=name, lw=1.5)
        axis.axvline(fit_min, color="0.6", lw=0.8, ls="--")
        for log_axis in (axis.xaxis, axis.yaxis):
            log_axis.set_major_formatter(ScalarFormatter())
            log_axis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("prefix size k")
        axis.set_ylabel("mean radius of gyration")
        axis.set_title("Radius of gyration vs prefix size")
        axis.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    return result


def tree_statistics_experiment(collections: Collections, *, show: bool = True) -> dict[str, object]:
    """Compare tree shape: depth, leaves, and branching, from parent arrays alone."""
    result = {}
    depth_curves = {}
    for name, trees in _nonempty(collections).items():
        rows = []
        depths = []
        for parents, _ in trees:
            depth = _depths(parents)
            children = np.bincount(parents[1:] - 1, minlength=len(parents))
            rows.append((
                depth.mean(), depth.max(), (children == 0).mean(),
                children.max(), children[children > 0].mean(),
            ))
            depths.append(depth)
        rows = np.array(rows)
        keys = ("mean_depth", "max_depth", "leaf_fraction", "max_children",
                "mean_children_of_internal_nodes")
        result[name] = {key: float(rows[:, index].mean()) for index, key in enumerate(keys)}
        depth_curves[name] = np.stack(depths).mean(0)
    if show:
        keys = list(next(iter(result.values())))
        print(f"{'collection':<14}" + "".join(f"{key:>34}" for key in keys))
        for name, row in result.items():
            print(f"{name:<14}" + "".join(f"{row[key]:>34.3f}" for key in keys))
        figure, axis = plt.subplots(figsize=(6, 3.5))
        for name, curve in depth_curves.items():
            axis.plot(np.arange(1, len(curve) + 1), curve, label=name, lw=1.5)
        axis.set_xlabel("arrival ID")
        axis.set_ylabel("mean depth")
        axis.set_title("Tree depth by arrival")
        axis.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    return result


def consistency_experiment(
    generator: DLAGenerator, *, seeds: Iterable[int] = range(5), show: bool = True
) -> dict[str, object]:
    """Check that answers do not depend on query order or on other seeds' queries."""
    seeds = tuple(seeds)
    nodes = list(range(1, generator.num_nodes + 1))
    shuffled = list(np.random.default_rng(0).permutation(nodes))
    checks = {"reverse_order": 0, "shuffled_order": 0, "interleaved_seeds": 0, "matches_arrays": 0}
    for seed in seeds:
        expected = [generator.query(seed, t) for t in nodes]
        checks["reverse_order"] += [generator.query(seed, t) for t in reversed(nodes)] == expected[::-1]
        checks["shuffled_order"] += (
            [generator.query(seed, int(t)) for t in shuffled] == [expected[t - 1] for t in shuffled]
        )
        interleaved = []
        for t in nodes:
            generator.query(seed + 1, t)
            interleaved.append(generator.query(seed, t))
        checks["interleaved_seeds"] += interleaved == expected
        parents, directions = generator.arrays(seed)
        checks["matches_arrays"] += (
            parents[1:].tolist() == [parent for parent, _ in expected[1:]]
            and directions[1:].tolist() == [direction for _, direction in expected[1:]]
        )
    result = {
        "seeds": len(seeds),
        "passed": {name: int(count) for name, count in checks.items()},
        "all_passed": all(count == len(seeds) for count in checks.values()),
    }
    if show:
        print(result)
    return result


def timing_experiment(
    generator: DLAGenerator,
    *,
    num_nodes: int,
    threads: int,
    arrivals: Sequence[int] | None = None,
    repeats: int = 10,
    show: bool = True,
) -> dict[str, object]:
    """Compare one model query for node t against simulating a cluster up to node t."""
    if arrivals is None:
        arrivals = sorted({max(2, num_nodes // 8), max(2, num_nodes // 2), num_nodes})
    simulate_dla(10, 0)  # compile before timing
    for _ in range(5):
        generator.query(123, num_nodes)

    rows = []
    for t in arrivals:
        query_times, simulate_times = [], []
        for repeat in range(repeats):
            started = time.perf_counter()
            generator.query(9_000_000 + repeat, t)
            query_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            simulate_dla(t, 8_000_000 + repeat)
            simulate_times.append(time.perf_counter() - started)
        rows.append({
            "arrival": int(t),
            "median_query_ms": 1e3 * float(np.median(query_times)),
            "median_simulate_to_arrival_ms": 1e3 * float(np.median(simulate_times)),
        })
    tree_times = []
    for repeat in range(repeats):
        started = time.perf_counter()
        generator.arrays(7_000_000 + repeat)
        tree_times.append(time.perf_counter() - started)

    result = {
        "per_arrival": rows,
        "median_full_tree_ms": 1e3 * float(np.median(tree_times)),
        "timing_environment": {
            "device": "cpu float32 public decoder, one query at a time",
            "num_nodes": num_nodes,
            "threads": threads,
        },
    }
    if show:
        print(f"{'arrival':>8} {'model query ms':>15} {'simulate to arrival ms':>23}")
        for row in rows:
            print(f"{row['arrival']:>8} {row['median_query_ms']:>15.3f} "
                  f"{row['median_simulate_to_arrival_ms']:>23.3f}")
        print(f"full tree through the query API: {result['median_full_tree_ms']:.2f} ms")
    return result


def _nonempty(collections: Collections) -> Collections:
    if not collections:
        raise ValueError("collections must contain at least one collection")
    for name, trees in collections.items():
        if not trees:
            raise ValueError(f"collection {name!r} must not be empty")
    return collections


def _collision_mask(positions: np.ndarray) -> np.ndarray:
    """Mark each node whose square was already occupied by an earlier node."""
    seen = set()
    mask = np.zeros(len(positions), dtype=bool)
    for index, (x, y) in enumerate(positions.tolist()):
        mask[index] = (x, y) in seen
        seen.add((x, y))
    return mask


def _prefix_gyration_radius(positions: np.ndarray) -> np.ndarray:
    """Radius of gyration of the first k nodes, for every k at once."""
    counts = np.arange(1, len(positions) + 1)[:, None]
    mean = np.cumsum(positions, axis=0) / counts
    mean_square = np.cumsum(positions.astype(np.float64) ** 2, axis=0) / counts
    return np.sqrt(np.clip((mean_square - mean**2).sum(1), 0, None))


def _depths(parents: np.ndarray) -> np.ndarray:
    """Hops from each node to the root. Parents are always older, so one pass suffices."""
    depth = np.zeros(len(parents), dtype=np.int64)
    for index in range(1, len(parents)):
        depth[index] = depth[parents[index] - 1] + 1
    return depth
