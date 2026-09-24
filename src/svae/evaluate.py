"""Notebook-facing experiments for evaluating seeded graph generators.

The first four experiments preserve the original demo notebook. The numbered
sections implement ``docs/evaluation-plan.md`` for arrival-labeled m=1 trees.
Numerical conventions and notebook examples are in ``docs/evaluation-api.md``.

Each experiment owns its calculation, printed summary, and visualization.  This
keeps notebook usage direct: call one function and receive both the displayed
analysis and its reusable result data.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.nn import functional
from scipy.special import logsumexp, xlogy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist, pdist
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .generate import ParentGenerator, SeededPAGenerator, keyed_rng, generators_from_run
from .simulate import parents_to_graph, simulate_graph, graph_to_parents
from .train import TrainingRun, train

GraphCollections = Mapping[str, Sequence[nx.Graph]]


def reconstruction_experiment(
    run: TrainingRun, *, batch_size: int = 256, show: bool = True
) -> dict[str, object]:
    """Report mean-latent reconstruction accuracy on the held-out test split."""
    model = run.model
    model.eval()
    # Batched: a whole large test split at once costs gigabytes of one-hot and
    # decoder activations for two scalar averages.
    queries = torch.arange(3, run.config.num_nodes + 1, device=run.device)
    cross_entropy_total = correct = predictions = 0.0
    with torch.no_grad():
        for start in range(0, len(run.data.test), batch_size):
            test_parents = run.data.test[start : start + batch_size].to(run.device)
            latent_mean, _ = model.encode(test_parents)
            _, masked_logits, _ = model.decode(latent_mean, queries)
            target_parents = test_parents[:, 2:] - 1
            cross_entropy_total += functional.cross_entropy(
                masked_logits.flatten(0, 1), target_parents.flatten(), reduction="sum"
            ).item()
            correct += (masked_logits.argmax(-1) == target_parents).sum().item()
            predictions += target_parents.numel()
    cross_entropy = cross_entropy_total / predictions
    accuracy = correct / predictions

    result = {
        "mean_latent_test_parent_accuracy": accuracy,
        "mean_latent_test_cross_entropy": cross_entropy,
        "note": "reconstruction diagnostic; not fresh generation or marginal likelihood",
    }
    if show:
        print(result)
    return result


def graph_samples_experiment(
    run: TrainingRun,
    generator: ParentGenerator,
    baseline: ParentGenerator,
    *,
    seeds: Iterable[int] | None = None,
    show: bool = True,
) -> dict[str, list[nx.Graph]]:
    """
    Generate the three original graph collections and display sample graphs.

    This function simulates graphs for each method, across all of the seeds provided.
    """
    if seeds is None:
        seeds = range(3_000_000, 3_000_000 + run.config.generated_graphs)
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one generation seed is required")

    collections = {
        "NetworkX": [parents_to_graph(parents.numpy()) for parents in run.data.test],
        "VAE": [generator.graph(seed) for seed in seeds],
        "Independent": [baseline.graph(seed) for seed in seeds],
    }
    if show:
        figure, axes = plt.subplots(3, 3, figsize=(10, 9))
        for row_index, (collection_name, graphs) in enumerate(collections.items()):
            for column_index, graph in enumerate(graphs[:3]):
                nx.draw(
                    graph,
                    nx.spring_layout(graph, seed=71),
                    ax=axes[row_index, column_index],
                    node_size=90,
                    with_labels=True,
                    font_size=6,
                )
                axes[row_index, column_index].set_title(
                    f"{collection_name} sample {column_index}"
                )
        plt.tight_layout()
        plt.show()
    return collections


def graph_statistics_experiment(
    collections: Mapping[str, Sequence[nx.Graph]],
    *,
    num_nodes: int,
    bootstrap_samples: int = 200,
    show: bool = True,
) -> dict[str, object]:
    """Display the original degree, arrival, and reinforcement diagnostics."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5)) if show else (None, None)
    summary = {}
    reinforcement_report = {}
    evaluation_rng = np.random.default_rng(8675309)

    for collection_name, graphs in collections.items():
        maximum_degrees = np.asarray(
            [max(dict(graph.degree()).values()) for graph in graphs], float
        )
        leaf_fractions = np.asarray(
            [
                sum(degree == 1 for _, degree in graph.degree()) / len(graph)
                for graph in graphs
            ],
            float,
        )
        degrees_by_arrival = np.asarray(
            [
                [graph.degree(node) for node in range(1, num_nodes + 1)]
                for graph in graphs
            ],
            float,
        )
        summary[collection_name] = {
            "max_degree_mean": maximum_degrees.mean(),
            "max_degree_sd": maximum_degrees.std(ddof=1),
            "leaf_fraction_mean": leaf_fractions.mean(),
            "leaf_fraction_sd": leaf_fractions.std(ddof=1),
        }

        if show:
            degrees = [degree for graph in graphs for _, degree in graph.degree()]
            values, counts = np.unique(degrees, return_counts=True)
            axes[0].plot(
                values, counts / counts.sum(), marker="o", label=collection_name
            )
            axes[1].plot(
                range(1, num_nodes + 1),
                degrees_by_arrival.mean(0),
                label=collection_name,
            )

        midpoint = num_nodes // 2
        midpoint_degrees = np.asarray(
            [graph.subgraph(range(1, midpoint + 1)).degree(1) for graph in graphs],
            float,
        )
        later_degree_gains = (
            np.asarray([graph.degree(1) for graph in graphs], float) - midpoint_degrees
        )
        pearson_correlation = _correlation(midpoint_degrees, later_degree_gains)
        bootstrap_correlations = []
        undefined = 0
        for _ in range(bootstrap_samples):
            sample_indices = evaluation_rng.integers(
                0, len(midpoint_degrees), len(midpoint_degrees)
            )
            correlation = _correlation(
                midpoint_degrees[sample_indices], later_degree_gains[sample_indices]
            )
            if correlation is None:
                undefined += 1
            else:
                bootstrap_correlations.append(correlation)
        confidence_interval = (
            [
                float(bound)
                for bound in np.quantile(bootstrap_correlations, [0.025, 0.975])
            ]
            if bootstrap_correlations
            else None
        )
        reinforcement_report[collection_name] = {
            "pearson": pearson_correlation,
            "bootstrap_95pct": confidence_interval,
            "undefined_resamples": undefined,
        }
        if show:
            axes[2].scatter(
                midpoint_degrees,
                later_degree_gains,
                s=12,
                alpha=0.45,
                label=collection_name,
            )

    if show:
        axes[0].set(
            xlabel="degree",
            ylabel="aggregate frequency",
            title="Degree distribution",
        )
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[1].set(
            xlabel="arrival ID", ylabel="mean final degree", title="Degree vs arrival"
        )
        axes[1].legend()
        print("Hub/leaf summary:", json.dumps(summary, indent=2))
        axes[2].set(
            xlabel="node 1 degree at N/2",
            ylabel="later degree gain",
            title="Reinforcement diagnostic",
        )
        axes[2].legend()
        plt.tight_layout()
        plt.show()
        print("Reinforcement:", json.dumps(reinforcement_report, indent=2))

    return {"hub_leaf_summary": summary, "reinforcement": reinforcement_report}


def timing_experiment(
    generator: ParentGenerator,
    *,
    num_nodes: int,
    threads: int,
    show: bool = True,
) -> dict[str, object]:
    """Run the original scalar-query, graph-generation, and NetworkX timings."""
    query = min(10, num_nodes)
    for _ in range(5):
        generator.parent(123, query)

    query_times = []
    graph_times = []
    for _ in range(50):
        started = time.perf_counter()
        generator.parent(123, query)
        query_times.append(time.perf_counter() - started)
    for seed_offset in range(10):
        started = time.perf_counter()
        generator.graph(9_000_000 + seed_offset)
        graph_times.append(time.perf_counter() - started)
    started = time.perf_counter()
    for seed_offset in range(10):
        simulate_graph(num_nodes, 8_000_000 + seed_offset)
    networkx_time = (time.perf_counter() - started) / 10

    result = {
        "median_scalar_query_ms_including_latent": 1e3 * np.median(query_times),
        "median_public_full_graph_ms": 1e3 * np.median(graph_times),
        "mean_networkx_full_graph_ms": 1e3 * networkx_time,
        "timing_environment": {
            "device": "cpu float32 public decoder",
            "num_nodes": num_nodes,
            "threads": threads,
        },
    }
    if show:
        print(result)
    return result


def _correlation(first_values: np.ndarray, second_values: np.ndarray) -> float | None:
    if first_values.std() == 0 or second_values.std() == 0:
        return None
    return float(np.corrcoef(first_values, second_values)[0, 1])


# ---------------------------------------------------------------------------
# 1. Exact structural invariants
# ---------------------------------------------------------------------------


def structural_invariants_experiment(
    collections: GraphCollections,
    *,
    num_nodes: int,
    show: bool = True,
) -> dict[str, object]:
    """Check the exact invariants implied by the current ``m=1`` model.

    For every graph, check node and edge counts, connectivity, acyclicity,
    self-loops, older-parent ordering, exact mean degree, and zero triangles.
    The display reports failure counts by collection and identifies
    offending graph indices so this experiment is useful as a correctness gate.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer of at least 2")
    if not collections:
        raise ValueError("collections must contain at least one graph collection")

    collection_results: dict[str, dict[str, object]] = {}
    all_passed = True
    for collection_name, graphs in collections.items():
        if not graphs:
            raise ValueError(f"collection {collection_name!r} must not be empty")

        graph_failures = []
        failure_counts = {invariant: 0 for invariant in _STRUCTURAL_INVARIANTS}
        for graph_index, graph in enumerate(graphs):
            failed_invariants = _structural_invariant_failures(graph, num_nodes)
            if failed_invariants:
                all_passed = False
                graph_failures.append(
                    {
                        "graph_index": graph_index,
                        "failed_invariants": failed_invariants,
                    }
                )
                for invariant in failed_invariants:
                    failure_counts[invariant] += 1

        collection_results[collection_name] = {
            "graphs": len(graphs),
            "passed": len(graphs) - len(graph_failures),
            "failed": len(graph_failures),
            "failure_counts": failure_counts,
            "failures": graph_failures,
        }

    result = {"all_passed": all_passed, "collections": collection_results}
    if show:
        print("Structural invariants (m=1)")
        print(f"{'collection':<20} {'graphs':>8} {'passed':>8} {'failed':>8}")
        for collection_name, summary in collection_results.items():
            print(
                f"{collection_name:<20} {summary['graphs']:>8} "
                f"{summary['passed']:>8} {summary['failed']:>8}"
            )
        if not all_passed:
            print("Failures:")
            for collection_name, summary in collection_results.items():
                for failure in summary["failures"]:
                    failed = ", ".join(failure["failed_invariants"])
                    print(f"  {collection_name}[{failure['graph_index']}]: {failed}")

    return result


_STRUCTURAL_INVARIANTS = (
    "node_count",
    "arrival_labels",
    "edge_count",
    "connected",
    "acyclic",
    "no_self_loops",
    "one_older_parent",
    "mean_degree",
    "zero_triangles",
)


def _structural_invariant_failures(graph: nx.Graph, num_nodes: int) -> tuple[str, ...]:
    """Return the names of all ``m=1`` invariants failed by one graph."""
    failures = []
    expected_nodes = set(range(1, num_nodes + 1))
    has_expected_nodes = set(graph) == expected_nodes
    has_no_self_loops = nx.number_of_selfloops(graph) == 0

    if graph.number_of_nodes() != num_nodes:
        failures.append("node_count")
    if not has_expected_nodes:
        failures.append("arrival_labels")
    if graph.number_of_edges() != num_nodes - 1:
        failures.append("edge_count")
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        failures.append("connected")
    if graph.number_of_nodes() == 0 or not nx.is_forest(graph):
        failures.append("acyclic")
    if not has_no_self_loops:
        failures.append("no_self_loops")

    # Arrival labels make parent direction recoverable from an undirected graph:
    # the unique neighbor with a smaller label is the node's parent.
    has_one_older_parent = has_expected_nodes and all(
        sum(neighbor < node for neighbor in graph.neighbors(node)) == 1
        for node in range(2, num_nodes + 1)
    )
    if not has_one_older_parent:
        failures.append("one_older_parent")

    observed_mean_degree = (
        sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        if graph.number_of_nodes()
        else float("nan")
    )
    expected_mean_degree = 2 * (num_nodes - 1) / num_nodes
    if not np.isclose(observed_mean_degree, expected_mean_degree):
        failures.append("mean_degree")

    if sum(nx.triangles(graph).values()) // 3 != 0:
        failures.append("zero_triangles")

    return tuple(failures)


# ---------------------------------------------------------------------------
# 2. Exact BA attachment-law evaluation
# ---------------------------------------------------------------------------


def ba_sequence_likelihood_experiment(
    collections: GraphCollections,
    *,
    normalize_per_arrival: bool = False,
    show: bool = True,
) -> dict[str, object]:
    """Compare complete parent sequences under the exact BA likelihood.

    Replay each arrival-labeled graph and sum the log probability of its observed parent
    at arrivals 3 through N.  Compare distributions, quantiles, and Wasserstein
    distance against the NetworkX reference.  Plot histograms or ECDFs in this
    function.  Normalized scores are required when comparing different N.
    """
    parents = _parent_collections(collections, same_size=not normalize_per_arrival)
    scores = {}
    for name, arrays in parents.items():
        values = []
        for array in arrays:
            score = sum(
                np.log(degrees[parent] / degrees.sum())
                for degrees, parent in _arrivals(array)
            )
            values.append(score / (len(array) - 2) if normalize_per_arrival else score)
        scores[name] = np.asarray(values)
    result = _scalar_comparison(scores)
    if show:
        plt.figure()
        for name, values in scores.items():
            plt.step(
                np.sort(values), np.arange(1, len(values) + 1) / len(values), label=name
            )
            print(name, result[name]["summary"])
            print(name, "Wasserstein distance to NetworkX:", result[name]["wasserstein"])
        plt.xlabel(
            "BA log probability / arrival"
            if normalize_per_arrival
            else "BA log probability"
        )
        plt.ylabel("Empirical CDF")
        plt.legend()
        plt.show()
    return result


def attachment_kernel_experiment(
    collections: GraphCollections,
    *,
    alpha_grid: Sequence[float] | None = None,
    bootstrap_samples: int = 1_000,
    show: bool = True,
) -> dict[str, object]:
    """Recover the empirical attachment exponent in ``A(k) = k**alpha``.

    Score each candidate alpha using replayed graph histories, retain the
    maximum-likelihood estimate for each collection, and bootstrap whole graphs
    for uncertainty.  The visualization should show the full likelihood curve,
    since peak width is as important as the winning alpha.
    """
    parents = _parent_collections(collections)
    grid = np.asarray(np.linspace(0, 2, 81) if alpha_grid is None else alpha_grid)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("alpha_grid must contain increasing finite values")
    result = {}
    for name, arrays in parents.items(): # for each completed candidate
        likelihood = np.zeros((len(arrays), len(grid)))
        for row, array in enumerate(arrays): # for each completeld graph
            for degrees, parent in _arrivals(array): # replay each graph one arrival at a time
                weights = grid[:, None] * np.log(degrees)
                likelihood[row] += weights[:, parent] - logsumexp(weights, axis=1)
        total = likelihood.sum(0)
        estimate = float(grid[total.argmax()])
        interval = _bootstrap(
            likelihood, lambda x: grid[x.sum(0).argmax()], bootstrap_samples
        ) # estimate uncertainty by repeatedly resampling whole graph rows.
        result[name] = {
            "alpha": estimate,
            "alpha_grid": grid,
            "log_likelihood": total,
            "alpha_interval": interval,
            "at_grid_boundary": estimate in (grid[0], grid[-1]),
        }
    if show:
        plt.figure()
        for name, row in result.items():
            plt.plot(grid, row["log_likelihood"], label=name)
            print(name, "alpha:", row["alpha"], "95% interval:", row["alpha_interval"])
        plt.xlabel("Attachment exponent alpha")
        plt.ylabel("Total log likelihood")
        plt.legend()
        plt.show()
    return result


def selected_parent_degree_experiment(
    collections: GraphCollections,
    *,
    show: bool = True,
) -> dict[str, object]:
    """Compare observed and BA-expected selections by parent degree class.

    During history replay, accumulate the BA probability mass assigned to each
    degree class and the class actually selected.  Report observed/expected
    ratios with opportunity counts, especially in sparse high-degree bins.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    result = {}
    for name, arrays in parents.items():
        observed, expected, opportunities = np.zeros((3, n))
        for array in arrays:
            for degrees, parent in _arrivals(array):
                counts = np.bincount(degrees, minlength=n)
                opportunities += counts
                expected += counts * np.arange(n) / degrees.sum()
                observed[degrees[parent]] += 1
        ratio = np.divide(
            observed, expected, out=np.full(n, np.nan), where=expected > 0
        )
        result[name] = {
            "observed": observed,
            "expected": expected,
            "opportunities": opportunities,
            "ratio": ratio,
        }
    if show:
        plt.figure()
        for name, row in result.items():
            plt.plot(np.arange(n), row["ratio"], ".-", label=name)
        plt.axhline(1, color="black", linestyle="--")
        plt.xlabel("Parent degree before arrival")
        plt.ylabel("Observed / BA expected selections")
        plt.legend()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 3. Degree-distribution laws
# ---------------------------------------------------------------------------


def degree_distribution_experiment(
    collections: GraphCollections,
    *,
    tail_thresholds: Sequence[int] = (5, 10),
    show: bool = True,
) -> dict[str, object]:
    """Compare empirical degree laws with NetworkX and the asymptotic BA law.

    Evaluate degree mass, complementary CDFs, total variation, selected tail
    masses, low-degree counts, and the complete maximum-degree distribution.
    Equal-sized NetworkX graphs are the primary finite-size reference.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    result = {}
    for name, arrays in parents.items():
        degrees = np.array([_degree_history(p)[-1] for p in arrays])
        counts = np.array([np.bincount(d, minlength=n) for d in degrees])
        pmf = counts.mean(0) / n
        result[name] = {
            "pmf": pmf,
            "ccdf": np.cumsum(pmf[::-1])[::-1],
            "low_degree_counts": counts[:, 1:4],
            "max_degrees": degrees.max(1),
            "tail_mass": {k: float(pmf[k:].sum()) for k in tail_thresholds},
        }
    reference = result.get("NetworkX")
    for row in result.values():
        row["total_variation"] = (
            None
            if reference is None
            else float(abs(row["pmf"] - reference["pmf"]).sum() / 2)
        )
    if show:
        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        for name, row in result.items():
            axes[0].plot(range(1, n), row["pmf"][1:], label=name)
            axes[1].plot(range(1, n), row["ccdf"][1:], label=name)
            axes[2].hist(
                row["max_degrees"],
                bins=np.arange(0.5, n + 0.5, 1),
                weights=np.full(len(row["max_degrees"]), 1 / len(row["max_degrees"])),
                histtype="step",
                label=name,
            )
            print(
                name,
                "PMF total variation vs NetworkX:",
                row["total_variation"],
                "node fractions with degree >= threshold:",
                row["tail_mass"],
            )
        k = np.arange(1, n)
        axes[0].plot(k, 4 / (k * (k + 1) * (k + 2)), "k--", label="Asymptotic BA")
        axes[0].set(xlabel="Node degree k", ylabel="Fraction of nodes with degree k")
        axes[1].set(xlabel="Degree threshold k", ylabel="Fraction of nodes with degree ≥ k")
        axes[2].set(xlabel="Maximum degree in a graph", ylabel="Fraction of graphs")
        largest_observed_degree = max(row["max_degrees"].max() for row in result.values())
        for axis in axes[:2]:
            axis.set_yscale("log")
            axis.set_xlim(1, largest_observed_degree)
        for axis, title in zip(axes, ("Degree PMF", "Degree CCDF", "Maximum degree")):
            axis.set_title(title)
            axis.legend()
        plt.tight_layout()
        plt.show()
    return result


def finite_size_degree_counts_experiment(
    collections: GraphCollections,
    *,
    num_nodes: int,
    bootstrap_samples: int = 1_000,
    show: bool = True,
) -> dict[str, object]:
    """Compare mean degree counts with the exact finite-size recurrence.

    Starting from two degree-one nodes, advance the expectation recurrence to N.
    Plot observed-minus-expected residuals by degree with graph-level bootstrap
    intervals; NetworkX should provide a check on the recurrence convention.
    """
    parents = _parent_collections(collections)
    if any(len(p) != num_nodes for arrays in parents.values() for p in arrays):
        raise ValueError("num_nodes must match every graph")
    expected = np.zeros(num_nodes)
    expected[1] = 2
    k = np.arange(num_nodes)
    for size in range(2, num_nodes):
        flow = k * expected / (2 * (size - 1))
        expected = expected - flow + np.roll(flow, 1)
        expected[1] += 1
    result = {}
    for name, arrays in parents.items():
        counts = np.array(
            [np.bincount(_degree_history(p)[-1], minlength=num_nodes) for p in arrays]
        )
        residual = counts.mean(0) - expected
        interval = _bootstrap(counts, lambda x: x.mean(0) - expected, bootstrap_samples)
        result[name] = {
            "expected": expected,
            "mean_counts": counts.mean(0),
            "residual": residual,
            "interval": interval,
        }
    if show:
        plt.figure()
        for name, row in result.items():
            plt.plot(k[1:], row["residual"][1:], label=name)
            plt.fill_between(k[1:], *row["interval"][:, 1:], alpha=0.15)
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Degree")
        plt.ylabel("Mean count minus exact BA expectation")
        # High-degree expected counts are tiny at this graph size; show the
        # low-degree bins where count residuals are interpretable.
        plt.xlim(1, min(num_nodes - 1, 15))
        plt.legend()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 4. Degree growth by node age
# ---------------------------------------------------------------------------


def degree_by_age_experiment(
    collections: GraphCollections,
    *,
    early_nodes: int = 10,
    show: bool = True,
) -> dict[str, object]:
    """Compare final and intermediate degree behavior by arrival ID.

    Report means, standard deviations, and quantiles for each arrival ID, with
    particular attention to early nodes.  Include degree trajectories and the
    rescaled value ``D_i(N) * sqrt(i/N)`` to expose age-scaling errors.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    if early_nodes < 1:
        raise ValueError("early_nodes must be positive")
    expected = np.zeros(n)
    expected[:2] = 1
    for size in range(2, n):
        expected[:size] *= 1 + 1 / (2 * (size - 1))
        expected[size] = 1
    result = {}
    for name, arrays in parents.items():
        histories = np.array([_degree_history(p) for p in arrays])
        degrees = histories[:, -1]
        result[name] = {
            "mean": degrees.mean(0),
            "sd": degrees.std(0),
            "quantiles": np.quantile(degrees, [0.025, 0.5, 0.975], axis=0),
            "early_degrees": degrees[:, :early_nodes],
            "mean_trajectories": histories.mean(0),
            "exact_expectation": expected,
            "rescaled_degrees": degrees * np.sqrt(np.arange(1, n + 1) / n),
        }
    if show:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
        selected_nodes = np.arange(1, min(early_nodes, n) + 1)
        offsets = np.linspace(-0.2, 0.2, len(result))
        for offset, (name, row) in zip(offsets, result.items()):
            # The interval describes uncertainty in the estimated mean, not
            # the much wider spread of degrees across individual graphs.
            sample_count = len(row["early_degrees"])
            mean_error = 1.96 * row["sd"][: len(selected_nodes)] / np.sqrt(sample_count)
            axes[0].errorbar(
                selected_nodes + offset,
                row["mean"][: len(selected_nodes)] - expected[: len(selected_nodes)],
                yerr=mean_error,
                fmt="o",
                capsize=2,
                label=name,
            )
            root_degrees = np.sort(row["early_degrees"][:, 0])
            axes[1].step(
                root_degrees,
                np.arange(1, sample_count + 1) / sample_count,
                where="post",
                label=name,
            )
        axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set(
            title="Early-node mean degree vs exact BA",
            xlabel="Node arrival ID",
            ylabel="Mean final degree minus BA expectation",
            xticks=selected_nodes,
        )
        axes[1].set(
            title="Final degree of node 1",
            xlabel="Node 1 final degree",
            ylabel="Fraction of graphs at or below degree",
            ylim=(0, 1),
        )
        for axis in axes:
            axis.legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 5. Reinforcement and temporal dependence
# ---------------------------------------------------------------------------


def reinforcement_experiment(
    collections: GraphCollections,
    *,
    node_ids: Sequence[int] = tuple(range(1, 11)),
    split_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    bootstrap_samples: int = 1_000,
    show: bool = True,
) -> dict[str, object]:
    """Measure early-degree versus later-gain correlations across nodes and splits.

    This generalizes the original node-1 midpoint diagnostic.  Compare complete
    node-by-split matrices with graph-level bootstrap intervals and visualize
    generator-minus-NetworkX errors as heatmaps.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    nodes = _node_indices(node_ids, n)
    splits = [_split_time(f, n) for f in split_fractions]

    def correlations(histories):
        matrix = np.full((len(nodes), len(splits)), np.nan)
        for column, split in enumerate(splits):
            for row, node in enumerate(nodes):
                if node + 1 <= split:
                    early = histories[:, split - 2, node]
                    correlation = _correlation(early, histories[:, -1, node] - early)
                    matrix[row, column] = np.nan if correlation is None else correlation
        return matrix

    result = {}
    for name, arrays in parents.items():
        histories = np.array([_degree_history(p) for p in arrays])
        result[name] = {
            "correlation": correlations(histories),
            "interval": _bootstrap(histories, correlations, bootstrap_samples),
            "node_ids": nodes + 1,
            "split_times": splits,
        }
    for row in result.values():
        row["reference_error"] = (
            row["correlation"] - result["NetworkX"]["correlation"]
            if "NetworkX" in result
            else None
        )
    if show:
        _, axes = plt.subplots(
            1, len(result), squeeze=False, figsize=(5 * len(result), 4)
        )
        for axis, (name, row) in zip(axes[0], result.items()):
            matrix = (
                row["reference_error"]
                if name != "NetworkX" and row["reference_error"] is not None
                else row["correlation"]
            )
            plot = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
            axis.set(
                title=("NetworkX correlation" if name == "NetworkX" else f"{name} − NetworkX"),
                xticks=range(len(splits)),
                xticklabels=splits,
                yticks=range(len(nodes)),
                yticklabels=nodes + 1,
                xlabel="Split after this many nodes",
                ylabel="Node arrival ID",
            )
            plt.colorbar(
                plot,
                ax=axis,
                label="Correlation" if name == "NetworkX" else "Correlation difference",
            )
        plt.suptitle("Early degree versus later degree gain")
        plt.tight_layout()
        plt.show()
    return result


def conditional_future_growth_experiment(
    collections: GraphCollections,
    *,
    node_ids: Sequence[int] = tuple(range(1, 11)),
    split_fraction: float = 0.5,
    minimum_bin_count: int = 20,
    bootstrap_samples: int = 200,
    show: bool = True,
) -> dict[str, object]:
    """Estimate future degree gain conditional on degree at an earlier split.

    Bin graphs by ``D_i(t)`` and compare mean later gain, uncertainty, and bin
    counts.  Keep low-count bins visible but clearly mark them as unreliable.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    split = _split_time(split_fraction, n)
    nodes = _node_indices(node_ids, n)
    if not np.any(nodes + 1 <= split):
        raise ValueError("at least one requested node must exist at the split")
    result = {}
    for name, arrays in parents.items():
        histories = np.array([_degree_history(p) for p in arrays])
        result[name] = {}
        for node in nodes:
            if node + 1 > split:
                continue
            early = histories[:, split - 2, node]
            gain = histories[:, -1, node] - early
            bins = {}
            for degree in np.unique(early):
                values = gain[early == degree]
                bins[int(degree)] = {
                    "mean_gain": values.mean(),
                    "count": len(values),
                    "reliable": len(values) >= minimum_bin_count,
                    "interval": _bootstrap(values, np.mean, bootstrap_samples),
                }
            result[name][int(node + 1)] = bins
    if show:
        eligible = nodes[nodes + 1 <= split] + 1
        columns = min(3, len(eligible))
        rows = (len(eligible) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows, columns, squeeze=False, figsize=(4.5 * columns, 3.5 * rows)
        )
        for axis, node in zip(axes.flat, eligible):
            for name, rows in result.items():
                bins = rows[node]
                means = np.array([v["mean_gain"] for v in bins.values()])
                bounds = np.array([v["interval"] for v in bins.values()])
                axis.plot(list(bins), means, ".-", label=name)
                axis.fill_between(list(bins), bounds[:, 0], bounds[:, 1], alpha=0.15)
                for degree, row in bins.items():
                    axis.annotate(
                        str(row["count"]) + ("*" if not row["reliable"] else ""),
                        (degree, row["mean_gain"]),
                        fontsize=6,
                    )
            axis.set(
                title=f"Node {node} (* sparse bin)",
                xlabel=f"Degree at {split}",
                ylabel="Future gain",
            )
        for axis in list(axes.flat)[len(eligible) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=len(labels))
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
    return result


def attachment_residuals_experiment(
    collections: GraphCollections,
    *,
    show: bool = True,
) -> dict[str, object]:
    """Diagnose selection bias after subtracting the exact BA probability.

    Replay each opportunity and compute ``observed - BA_probability``.  Aggregate
    residuals by current degree, arrival time, node age, leader status, and prior
    attachments; compare NetworkX as the finite-sample noise floor.
    """
    parents = _parent_collections(collections)
    result = {}
    for name, arrays in parents.items():
        groups = {
            key: {}
            for key in ("degree", "time", "age", "leader", "previous_attachment")
        }
        lag_pairs = []
        for array in arrays:
            previous_residual = None
            previous_parent = -1
            for step, (degrees, parent) in enumerate(_arrivals(array), start=3):
                residual = -degrees / degrees.sum()
                residual[parent] += 1
                labels = {
                    "degree": degrees,
                    "time": np.full(len(degrees), step),
                    "age": step - np.arange(1, step),
                    "leader": degrees == degrees.max(),
                    "previous_attachment": np.arange(len(degrees)) == previous_parent,
                }
                for key, labels_at_step in labels.items():
                    for label in np.unique(labels_at_step):
                        values = residual[labels_at_step == label]
                        total, count = groups[key].get(int(label), (0.0, 0))
                        groups[key][int(label)] = (
                            total + values.sum(),
                            count + len(values),
                        )
                if previous_residual is not None:
                    lag_pairs.extend(
                        zip(previous_residual, residual[: len(previous_residual)])
                    )
                previous_residual, previous_parent = residual, parent
        report = {
            key: {
                label: {"mean": total / count, "count": count}
                for label, (total, count) in sorted(bins.items())
            }
            for key, bins in groups.items()
        }
        pairs = np.asarray(lag_pairs)
        report["lag_one_correlation"] = (
            _correlation(pairs[:, 0], pairs[:, 1]) if len(pairs) else None
        )
        result[name] = report
    if show:
        _, axes = plt.subplots(1, 5, figsize=(18, 4))
        for axis, key in zip(axes, groups):
            for name, row in result.items():
                axis.plot(
                    list(row[key]),
                    [v["mean"] for v in row[key].values()],
                    ".-",
                    label=name,
                )
            axis.axhline(0, color="black", linestyle="--")
            axis.set(xlabel=key, ylabel="Mean BA residual")
            axis.legend()
        print(
            "Lag-one residual correlations:",
            {name: row["lag_one_correlation"] for name, row in result.items()},
        )
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 6. Hub formation and persistence
# ---------------------------------------------------------------------------


def hub_dynamics_experiment(
    collections: GraphCollections,
    *,
    split_fraction: float = 0.5,
    show: bool = True,
) -> dict[str, object]:
    """Compare hub size, leader identity, leader changes, and persistence.

    Use a documented tie policy while replaying graph growth.  Analyze ordered
    top degrees, gaps, final-leader arrival IDs, first-lead time, number of leader
    changes, and the transition from split-time leader to final leader.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    split = _split_time(split_fraction, n)
    result = {}
    for name, arrays in parents.items():
        histories = np.array([_degree_history(p) for p in arrays])
        # np.argmax resolves every tie to the smallest arrival ID consistently.
        leaders = histories.argmax(2) + 1
        final = leaders[:, -1]
        transition_counts = np.zeros((n, n), dtype=int)
        np.add.at(transition_counts, (leaders[:, split - 2] - 1, final - 1), 1)
        totals = transition_counts.sum(1, keepdims=True)
        transition = np.divide(
            transition_counts, totals, out=np.full((n, n), np.nan), where=totals > 0
        )
        ordered = np.sort(histories[:, -1], axis=1)[:, ::-1]
        result[name] = {
            "top_degrees": ordered[:, :10],
            "gap": ordered[:, 0] - ordered[:, 1],
            "leader_ids": final,
            "leader_pmf": np.bincount(final, minlength=n + 1)[1:] / len(final),
            "leader_changes": (np.diff(leaders, axis=1) != 0).sum(1),
            "first_lead_time": (leaders == final[:, None]).argmax(1) + 2,
            "persistence": float(np.mean(leaders[:, split - 2] == final)),
            "transition_counts": transition_counts,
            "transition_probabilities": transition,
            "tie_policy": "smallest arrival ID",
        }
    if show:
        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        for name, row in result.items():
            axes[0].plot(range(1, n + 1), row["leader_pmf"], label=name)
            axes[1].plot(
                range(1, row["top_degrees"].shape[1] + 1),
                row["top_degrees"].mean(0),
                label=name,
            )
            axes[2].hist(
                row["leader_changes"], histtype="step", density=True, label=name
            )
            print(name, "leader persistence:", row["persistence"])
        for axis, title in zip(
            axes, ("Final leader arrival ID", "Ranked hub degrees", "Leader changes")
        ):
            axis.set_title(title)
            axis.legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 7. Dependencies beyond node 1
# ---------------------------------------------------------------------------


def early_node_covariance_experiment(
    collections: GraphCollections,
    *,
    node_ids: Sequence[int] = tuple(range(1, 11)),
    bootstrap_samples: int = 1_000,
    show: bool = True,
) -> dict[str, object]:
    """Compare covariance and correlation among fixed early-node degrees.

    Display all collections on common color scales and report matrix-level
    distances from NetworkX alongside entrywise graph-bootstrap uncertainty.
    """
    parents = _parent_collections(collections)
    nodes = _node_indices(node_ids, len(next(iter(parents.values()))[0]))
    result = {}
    for name, arrays in parents.items():
        degrees = np.array([_degree_history(p)[-1, nodes] for p in arrays])
        result[name] = {
            "covariance": _covariance(degrees),
            "correlation": _correlation_matrix(degrees),
            "covariance_interval": _bootstrap(degrees, _covariance, bootstrap_samples),
            "correlation_interval": _bootstrap(
                degrees, _correlation_matrix, bootstrap_samples
            ),
        }
    for row in result.values():
        row["covariance_distance"] = (
            float(np.linalg.norm(row["covariance"] - result["NetworkX"]["covariance"]))
            if "NetworkX" in result
            else None
        )
    if show:
        _, axes = plt.subplots(
            2, len(result), squeeze=False, figsize=(5 * len(result), 8)
        )
        limit = max(np.nanmax(abs(row["covariance"])) for row in result.values())
        for column, (name, row) in enumerate(result.items()):
            for index, key in enumerate(("covariance", "correlation")):
                axis = axes[index, column]
                bound = limit if index == 0 else 1
                plot = axis.imshow(row[key], cmap="coolwarm", vmin=-bound, vmax=bound)
                axis.set(
                    title=f"{name}: {key}",
                    xticks=range(len(nodes)),
                    xticklabels=nodes + 1,
                    yticks=range(len(nodes)),
                    yticklabels=nodes + 1,
                )
                plt.colorbar(plot, ax=axis)
        plt.tight_layout()
        plt.show()
    return result


def joint_attachment_experiment(
    collections: GraphCollections,
    *,
    arrival_pairs: Sequence[tuple[int, int]],
    candidate_parents: Sequence[int] = (1, 2, 3, 4, 5),
    minimum_event_count: int = 20,
    bootstrap_samples: int = 200,
    show: bool = True,
) -> dict[str, object]:
    """Measure dependence between pairs of parent-query outcomes.

    Estimate joint-to-product-of-marginals ratios for declared arrival pairs and
    early candidate parents.  Include confidence intervals and suppress numeric
    interpretation when event counts fall below ``minimum_event_count``.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    events = [(s, t, i) for s, t in arrival_pairs for i in candidate_parents]
    if not events or any(not 1 <= i < s < t <= n or s < 3 for s, t, i in events):
        raise ValueError(
            "events require 1 <= parent < first arrival < second arrival <= N, with first arrival >= 3"
        )

    def ratio(values):
        denominator = values[:, 0].mean() * values[:, 1].mean()
        return (
            np.mean(values[:, 0] & values[:, 1]) / denominator
            if denominator
            else np.nan
        )

    result = {}
    for name, arrays in parents.items():
        array = np.asarray(arrays)
        result[name] = {}
        for s, t, i in events:
            values = np.column_stack((array[:, s - 1] == i, array[:, t - 1] == i))
            count = int(np.sum(values[:, 0] & values[:, 1]))
            result[name][s, t, i] = {
                "ratio": ratio(values),
                "joint_count": count,
                "marginal_counts": values.sum(0),
                "reliable": count >= minimum_event_count,
                "interval": _bootstrap(values, ratio, bootstrap_samples),
            }
    if show:
        plt.figure(figsize=(max(8, len(events)), 4))
        any_reliable = False
        for name, rows in result.items():
            # Sparse ratios remain in returned data but are omitted from inference plots.
            any_reliable |= any(v["reliable"] for v in rows.values())
            plt.plot(
                range(len(events)),
                [v["ratio"] if v["reliable"] else np.nan for v in rows.values()],
                ".-",
                label=name,
            )
            print(
                name,
                "joint counts:",
                {event: row["joint_count"] for event, row in rows.items()},
            )
        if not any_reliable:
            plt.text(
                0.5,
                0.5,
                f"No event reached {minimum_event_count} joint observations",
                transform=plt.gca().transAxes,
                ha="center",
                va="center",
            )
        plt.xticks(range(len(events)), [str(e) for e in events], rotation=60)
        plt.axhline(1, color="black", linestyle="--")
        plt.xlabel("Event (first arrival, second arrival, parent)")
        plt.ylabel("Joint / product of marginal probabilities")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return result


def hub_concentration_experiment(
    collections: GraphCollections,
    *,
    top_k: int = 5,
    show: bool = True,
) -> dict[str, object]:
    """Compare how total degree is concentrated among leading nodes.

    Measure largest-hub and top-k edge shares, Herfindahl concentration, and
    entropy of normalized degree mass as scalar distributions per collection.
    """
    _parent_collections(collections)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    metrics = {
        key: {}
        for key in (
            "largest_hub_edge_share",
            "top_k_edge_share",
            "herfindahl",
            "degree_entropy",
        )
    }
    for name, graphs in collections.items():
        rows = []
        for graph in graphs:
            degrees = dict(graph.degree())
            hubs = set(sorted(graph, key=lambda i: (-degrees[i], i))[:top_k])
            shares = np.array(list(degrees.values())) / (2 * graph.number_of_edges())
            # Count the union of incident edges: hub-hub edges count only once.
            rows.append(
                (
                    max(degrees.values()) / graph.number_of_edges(),
                    sum(u in hubs or v in hubs for u, v in graph.edges())
                    / graph.number_of_edges(),
                    np.sum(shares**2),
                    -np.sum(xlogy(shares, shares)),
                )
            )
        for column, key in enumerate(metrics):
            metrics[key][name] = np.asarray(rows)[:, column]
    result = {key: _scalar_comparison(values) for key, values in metrics.items()}
    if show:
        _, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axis_labels = {
            "largest_hub_edge_share": "Edges incident to largest hub / all edges",
            "top_k_edge_share": f"Edges incident to top {top_k} nodes / all edges",
            "herfindahl": "Herfindahl degree concentration",
            "degree_entropy": "Degree-share entropy (nats)",
        }
        for axis, (key, values) in zip(axes, metrics.items()):
            # Use the same bins for each method so their histogram heights
            # represent comparable fractions of graphs.
            bins = np.histogram_bin_edges(np.concatenate(list(values.values())), bins="auto")
            for name, samples in values.items():
                axis.hist(
                    samples,
                    bins=bins,
                    weights=np.full(len(samples), 1 / len(samples)),
                    histtype="step",
                    label=name,
                )
            axis.set(xlabel=axis_labels[key], ylabel="Fraction of graphs per bin")
            axis.legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 8. Tree-specific geometry
# ---------------------------------------------------------------------------


def tree_geometry_experiment(
    collections: GraphCollections,
    *,
    root: int = 1,
    show: bool = True,
) -> dict[str, object]:
    """Compare rooted-tree depth, distance, and branch geometry.

    Evaluate depth profiles, diameter, radius, pairwise distances, sorted root-
    branch sizes, descendant counts, branch balance, ancestor-age relationships,
    and lowest-common-ancestor depths.  Validate tree structure before analysis.
    """
    _parent_collections(collections)
    result = {}
    for name, graphs in collections.items():
        rows = [_tree_geometry(graph, root) for graph in graphs]
        result[name] = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    scalar_keys = (
        "mean_depth",
        "max_depth",
        "diameter",
        "radius",
        "branch_count",
        "largest_branch_fraction",
        "branch_balance",
    )
    for name, row in result.items():
        row["wasserstein"] = (
            {
                key: float(wasserstein_distance(row[key], result["NetworkX"][key]))
                for key in scalar_keys
            }
            if "NetworkX" in result
            else {}
        )
    if show:
        _, axes = plt.subplots(1, 4, figsize=(16, 4))
        for name, row in result.items():
            axes[0].plot(row["depths"].mean(0), label=name)
            axes[1].hist(row["diameter"], density=True, histtype="step", label=name)
            axes[2].plot(row["branch_sizes"].mean(0), label=name)
            axes[3].plot(row["distance_histogram"].mean(0), label=name)
            print(name, "geometry Wasserstein:", row["wasserstein"])
        for axis, title in zip(
            axes,
            (
                "Depth by arrival (index + 1)",
                "Diameter",
                "Root branch size by rank",
                "Pair distance counts",
            ),
        ):
            axis.set_title(title)
            axis.legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 9. Degree correlations along edges
# ---------------------------------------------------------------------------


def edge_degree_correlation_experiment(
    collections: GraphCollections,
    *,
    show: bool = True,
) -> dict[str, object]:
    """Compare the degree and arrival relationships at edge endpoints.

    Analyze parent-versus-child final degrees, joint endpoint-degree mass,
    average neighbor degree by degree, assortativity, and parent-versus-child
    arrival IDs.  Show joint distributions as well as scalar coefficients.
    """
    parents = _parent_collections(collections)
    n = len(next(iter(parents.values()))[0])
    result = {}
    for name, graphs in collections.items():
        joint, arrivals = np.zeros((n, n)), np.zeros((n, n))
        neighbor_sum, neighbor_count = np.zeros(n), np.zeros(n)
        assortativity = []
        for graph in graphs:
            degrees = dict(graph.degree())
            endpoint_pairs = []
            for u, v in graph.edges():
                parent, child = sorted((u, v))
                joint[degrees[parent], degrees[child]] += 1
                arrivals[parent - 1, child - 1] += 1
                endpoint_pairs.extend(
                    ((degrees[u], degrees[v]), (degrees[v], degrees[u]))
                )
            pairs = np.array(endpoint_pairs)
            assortativity.append(_correlation(pairs[:, 0], pairs[:, 1]))
            for node, value in nx.average_neighbor_degree(graph).items():
                neighbor_sum[degrees[node]] += value
                neighbor_count[degrees[node]] += 1
        result[name] = {
            "joint_degree_pmf": joint / joint.sum(),
            "arrival_pmf": arrivals / arrivals.sum(),
            "mean_neighbor_degree": np.divide(
                neighbor_sum,
                neighbor_count,
                out=np.full(n, np.nan),
                where=neighbor_count > 0,
            ),
            "assortativity": np.array(assortativity, dtype=float),
        }
    if show:
        _, axes = plt.subplots(
            2, len(result), squeeze=False, figsize=(5 * len(result), 8)
        )
        limit = max(row["joint_degree_pmf"].max() for row in result.values())
        for column, (name, row) in enumerate(result.items()):
            plot = axes[0, column].imshow(
                row["joint_degree_pmf"], origin="lower", vmin=0, vmax=limit
            )
            axes[0, column].set(
                title=name, xlabel="Child degree", ylabel="Parent degree"
            )
            plt.colorbar(plot, ax=axes[0, column])
            axes[1, column].plot(range(n), row["mean_neighbor_degree"], ".-")
            axes[1, column].set(xlabel="Degree", ylabel="Mean neighbor degree")
            print(name, "assortativity:", row["assortativity"])
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 10. Latent-specific diagnostics
# ---------------------------------------------------------------------------


def latent_ablation_experiment(
    generator: SeededPAGenerator,
    *,
    seeds: Iterable[int],
    samples_per_latent: int = 50,
    baseline: ParentGenerator | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Test whether shared latent variation coordinates graph-level behavior.

    Compare normal, zero, shuffled, latent-averaged, and independent
    generation.  The central variance decomposition requires separate control of
    the latent key and query-randomness key via ``diagnostic_graph``.
    Supply the fitted independent-parent generator as ``baseline`` to include it.
    """
    seeds = tuple(seeds)
    if len(seeds) < 2 or samples_per_latent < 2:
        raise ValueError(
            "variance decomposition needs at least two latents and two samples per latent"
        )
    latents = [generator.latent(seed) for seed in seeds]
    shuffled = np.random.default_rng(8675309).permutation(len(seeds))
    averaged = {
        q: np.mean([generator.probabilities(seed, q) for seed in seeds], axis=0)
        for q in range(3, generator.num_nodes + 1)
    }
    collections = {name: [] for name in ("Normal", "Zero", "Shuffled", "Averaged")}
    if baseline is not None:
        collections["Independent"] = []
    repetitions = []
    for index, seed in enumerate(seeds):
        collections["Normal"].append(generator.graph(seed))
        collections["Zero"].append(
            generator.diagnostic_graph(torch.zeros_like(latents[index]), seed)
        )
        # Whole-vector shuffling preserves within-graph sharing. It is a seed
        # reassignment control, not an intervention that removes coordination.
        collections["Shuffled"].append(
            generator.diagnostic_graph(latents[shuffled[index]], seed)
        )
        array = [-1, 1]
        for query, probabilities in averaged.items():
            cumulative = np.cumsum(probabilities)
            cumulative[-1] = 1
            array.append(
                int(
                    np.searchsorted(
                        cumulative,
                        keyed_rng(seed, "parent", query).random(),
                        side="right",
                    )
                )
                + 1
            )
        collections["Averaged"].append(parents_to_graph(array))
        if baseline is not None:
            collections["Independent"].append(baseline.graph(seed))
        repetitions.append(
            [
                _graph_features(
                    generator.diagnostic_graph(
                        latents[index], 10_000_000 + index * samples_per_latent + repeat
                    )
                )
                for repeat in range(samples_per_latent)
            ]
        )
    values = np.asarray(repetitions)
    within = values.var(axis=1, ddof=1).mean(0)
    observed_between = values.mean(1).var(axis=0, ddof=1)
    # Finite replication adds within/R noise to the variance of group means.
    between = observed_between - within / samples_per_latent
    diagnostics = graph_statistics_experiment(
        collections, num_nodes=generator.num_nodes, show=show
    )
    if show:
        _, axis = plt.subplots(figsize=(12, 4))
        x = np.arange(len(_FEATURE_NAMES))
        axis.bar(x - 0.2, between, 0.4, label="Between latent (noise corrected)")
        axis.bar(x + 0.2, within, 0.4, label="Within latent")
        axis.set_xticks(x, _FEATURE_NAMES, rotation=45, ha="right")
        axis.legend()
        plt.tight_layout()
        plt.show()
    return {
        "collections": collections,
        "diagnostics": diagnostics,
        "feature_names": _FEATURE_NAMES,
        "replicate_features": values,
        "between_latent": between,
        "within_latent": within,
        "observed_between_means": observed_between,
        "note": "Negative corrected between estimates reflect sampling noise; whole-latent shuffling preserves coordination.",
    }


def decoder_entropy_experiment(
    generator: SeededPAGenerator,
    *,
    seeds: Iterable[int],
    show: bool = True,
) -> dict[str, object]:
    """Measure decoder entropy and sensitivity to the shared latent by query.

    Compare entropy with ``log(t-1)``, variation across latents, argmax changes,
    and variance of parent probabilities.  This currently requires a generator
    exposing decoder probabilities, such as ``SeededPAGenerator``.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    result = {}
    for query in range(3, generator.num_nodes + 1):
        probabilities = np.array(
            [generator.probabilities(seed, query) for seed in seeds]
        )
        modes = probabilities.argmax(1)
        frequencies = np.bincount(modes, minlength=query - 1) / len(seeds)
        result[query] = {
            "entropy": -xlogy(probabilities, probabilities).sum(1),
            "probability_variance": probabilities.var(0),
            "mode_frequencies": frequencies,
            "mode_disagreement": 1 - np.sum(frequencies**2),
        }
    if show:
        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        queries = list(result)
        entropy = np.array([row["entropy"] for row in result.values()])
        axes[0].plot(queries, entropy.mean(1), label="Decoder")
        axes[0].fill_between(
            queries, *np.quantile(entropy, [0.025, 0.975], axis=1), alpha=0.2
        )
        axes[0].plot(queries, np.log(np.array(queries) - 1), "k--", label="Uniform")
        axes[0].legend()
        axes[1].plot(queries, [v["mode_disagreement"] for v in result.values()])
        axes[1].plot(
            queries,
            1 - 1 / (np.array(queries) - 1),
            "k--",
            label="Uniform modes (maximum)",
        )
        axes[1].legend()
        axes[2].plot(
            queries, [v["probability_variance"].sum() for v in result.values()]
        )
        for axis, title in zip(
            axes,
            (
                "Entropy (nats)",
                "Mode disagreement across latents",
                "Total probability variance",
            ),
        ):
            axis.set(title=title, xlabel="Arrival")
        axes[0].set_ylabel("Parent-choice entropy (nats)")
        axes[1].set_ylabel("Chance two seeds have different argmax")
        axes[2].set_ylabel("Sum of probability variances across seeds")
        plt.tight_layout()
        plt.show()
    return result


def latent_predictability_experiment(
    run: TrainingRun,
    *,
    seeds: Iterable[int] | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Fit held-out probes from latent vectors to global graph properties.

    Use simple regression for scalar properties and classification for leader
    identity.  Report held-out R-squared or accuracy against trivial baselines;
    keep probe fitting and its plots inside this experiment.
    """
    generator = SeededPAGenerator.from_model(run.model)
    seeds = tuple(
        range(4_000_000, 4_000_000 + run.config.generated_graphs)
        if seeds is None
        else seeds
    )
    if len(seeds) < 10 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least ten distinct seeds for held-out probes")
    latent = np.array([generator.latent(seed).numpy() for seed in seeds])
    properties = np.array([_graph_features(generator.graph(seed)) for seed in seeds])
    training, testing = train_test_split(
        np.arange(len(seeds)), test_size=0.3, random_state=8675309
    )
    result = {}
    for column, name in enumerate(_FEATURE_NAMES):
        y = properties[:, column]
        if name == "leader_id":
            estimator = (
                LogisticRegression(max_iter=1000)
                if len(np.unique(y[training])) > 1
                else DummyClassifier(strategy="most_frequent")
            )
            baseline = DummyClassifier(strategy="most_frequent")
            score = accuracy_score
        else:
            estimator, baseline, score = LinearRegression(), DummyRegressor(), r2_score
        probe = make_pipeline(StandardScaler(), estimator)
        probe.fit(latent[training], y[training])
        baseline.fit(latent[training], y[training])
        prediction = probe.predict(latent[testing])
        result[name] = {
            "score": score(y[testing], prediction),
            "baseline_score": score(y[testing], baseline.predict(latent[testing])),
            "metric": "accuracy" if name == "leader_id" else "R2",
            "truth": y[testing],
            "prediction": prediction,
        }
    if show:
        _, axes = plt.subplots(1, 2, figsize=(13, 4))
        regression = {name: row for name, row in result.items() if row["metric"] == "R2"}
        x = np.arange(len(regression))
        axes[0].bar(x - 0.2, [row["score"] for row in regression.values()], 0.4, label="Probe")
        axes[0].bar(
            x + 0.2,
            [row["baseline_score"] for row in regression.values()],
            0.4,
            label="Training-mean baseline",
        )
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set(
            title="Graph properties predicted from latent",
            ylabel="Held-out R² (higher is better; can be negative)",
        )
        axes[0].set_xticks(x, regression, rotation=45, ha="right")
        leader = result["leader_id"]
        axes[1].bar(
            ["Probe", "Most-frequent baseline"],
            [leader["score"], leader["baseline_score"]],
        )
        axes[1].set(
            title="Final leader predicted from latent",
            ylabel="Held-out accuracy (fraction correct)",
            ylim=(0, 1),
        )
        axes[0].legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 11. Distribution-level comparisons
# ---------------------------------------------------------------------------


def distribution_comparison_experiment(
    collections: GraphCollections,
    *,
    reference_name: str = "NetworkX",
    cross_validation_folds: int = 5,
    show: bool = True,
) -> dict[str, object]:
    """Run graph-level two-sample and diversity comparisons.

    Define a documented graph-feature vector, then use held-out classifier tests,
    feature-space distances, coverage/precision-like measures, and duplicate
    rates.  Generation source, seeds, and other artifacts must not be features.
    """
    parents = _parent_collections(collections)
    if reference_name not in collections:
        raise ValueError(f"missing reference collection {reference_name!r}")
    features = {
        name: np.array([_graph_features(g) for g in graphs])
        for name, graphs in collections.items()
    }
    reference = features[reference_name]
    if cross_validation_folds < 2 or any(
        len(x) < cross_validation_folds for x in features.values()
    ):
        raise ValueError(
            "every collection needs at least cross_validation_folds graphs; folds must be >= 2"
        )
    result = {}
    # Feature scaling for distances is fitted only on the reference. Classifier
    # scaling below is fitted separately inside each training fold.
    scaler = StandardScaler().fit(reference)
    ref_scaled = scaler.transform(reference)
    ref_distances = cdist(ref_scaled, ref_scaled)
    np.fill_diagonal(ref_distances, np.inf)
    radii = np.min(ref_distances, axis=1)
    for name, values in features.items():
        scaled = scaler.transform(values)
        cross_distances = cdist(ref_scaled, scaled)
        energy = (
            2 * cross_distances.mean()
            - cdist(ref_scaled, ref_scaled).mean()
            - cdist(scaled, scaled).mean()
        )
        row = {
            "features": values,
            "feature_names": _FEATURE_NAMES,
            "energy_distance_squared": max(0.0, float(energy)),
            "wasserstein": {
                key: float(wasserstein_distance(reference[:, i], values[:, i]))
                for i, key in enumerate(_FEATURE_NAMES)
                if key != "leader_id"
            },
            "coverage": float(np.mean(cross_distances.min(1) <= radii)),
            "precision": float(np.mean((cross_distances <= radii[:, None]).any(0))),
            "duplicate_rate": 1 - len({tuple(p) for p in parents[name]}) / len(values),
            "feature_diversity": float(pdist(scaled).mean()),
        }
        n = len(parents[name][0])
        ref_leaders = np.bincount(reference[:, 2].astype(int), minlength=n + 1) / len(
            reference
        )
        leaders = np.bincount(values[:, 2].astype(int), minlength=n + 1) / len(values)
        row["leader_total_variation"] = float(abs(leaders - ref_leaders).sum() / 2)
        if name != reference_name:
            # Balance source classes before CV so chance accuracy is 50%.
            rng = np.random.default_rng(8675309)
            count = min(len(reference), len(values))
            x = np.concatenate(
                (
                    reference[rng.choice(len(reference), count, replace=False)],
                    values[rng.choice(len(values), count, replace=False)],
                )
            )
            y = np.repeat([0, 1], count)
            scores, coefficients = [], []
            for training, testing in StratifiedKFold(
                cross_validation_folds, shuffle=True, random_state=8675309
            ).split(x, y):
                classifier = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000)
                )
                classifier.fit(x[training], y[training])
                scores.append(
                    accuracy_score(y[testing], classifier.predict(x[testing]))
                )
                coefficients.append(classifier[-1].coef_[0])
            row["classifier_accuracy"] = np.array(scores)
            row["standardized_coefficients"] = np.array(coefficients)
        result[name] = row
    if show:
        comparisons = {
            name: row for name, row in result.items() if name != reference_name
        }
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        for index, (name, row) in enumerate(comparisons.items()):
            axes[0].scatter(
                np.full(len(row["classifier_accuracy"]), index),
                row["classifier_accuracy"],
                label=name,
            )
            axes[1].plot(
                _FEATURE_NAMES,
                row["standardized_coefficients"].mean(0),
                ".-",
                label=name,
            )
            print(
                name,
                {
                    key: row[key]
                    for key in (
                        "energy_distance_squared",
                        "coverage",
                        "precision",
                        "duplicate_rate",
                        "leader_total_variation",
                    )
                },
            )
        axes[0].set_xticks(range(len(comparisons)), list(comparisons))
        axes[0].axhline(0.5, color="black", linestyle="--")
        axes[0].set_ylabel("Held-out classifier accuracy per fold")
        axes[1].tick_params(axis="x", rotation=60)
        axes[1].set_ylabel("Standardized logistic coefficient")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 12. Scaling laws
# ---------------------------------------------------------------------------


def scaling_experiment(
    configurations: Sequence[Config],
    *,
    training_seeds: Sequence[int],
    bootstrap_samples: int = 200,
    device: str | torch.device | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Coordinate evaluation across graph sizes, latent widths, and train seeds.

    Each configuration describes one N/D training condition. Train independent
    runs and apply the same metric suite,
    separating within-checkpoint sampling uncertainty from between-training-run
    variation. Also report reconstruction, KL, latency, parameters,
    and sample diversity alongside structural errors.

    Each ``Config`` supplies one graph-size/latent-width condition; its
    ``init_seed`` is replaced by each requested training seed. Checkpoints are
    not written; results retain configurations and metrics rather than models.
    """
    if not configurations or not training_seeds:
        raise ValueError("provide configurations and training seeds")
    records = []
    for condition, config in enumerate(configurations):
        for seed in training_seeds:
            actual = replace(config, init_seed=seed)
            run = train(
                actual, device=device, checkpoint_dir=None, show=False, report=None
            )
            generators = generators_from_run(run)
            collections = graph_samples_experiment(
                run, generators["vae"], generators["independent"], show=False
            )
            likelihood = ba_sequence_likelihood_experiment(
                collections, normalize_per_arrival=True, show=False
            )
            reinforcement = reinforcement_experiment(
                collections,
                node_ids=range(1, min(10, config.num_nodes) + 1),
                bootstrap_samples=bootstrap_samples,
                show=False,
            )
            degree = degree_distribution_experiment(collections, show=False)
            hubs = hub_dynamics_experiment(collections, show=False)
            geometry = tree_geometry_experiment(collections, show=False)
            generated_features = np.array(
                [_graph_features(g) for g in collections["VAE"]]
            )
            reference_features = np.array(
                [_graph_features(g) for g in collections["NetworkX"]]
            )
            error = reinforcement["VAE"]["reference_error"]
            # Evaluate KL on the restored model, not the last history row.
            with torch.no_grad():
                mean, log_variance = run.model.encode(run.data.test.to(run.device))
                kl = float(
                    (
                        0.5
                        * (mean.square() + log_variance.exp() - 1 - log_variance).sum(1)
                    ).mean()
                )
            output_layer = run.model.decoder[-1]
            metrics = {
                "ba_wasserstein": likelihood["VAE"]["wasserstein"],
                "reinforcement_rmse": (
                    float(np.sqrt(np.nanmean(error**2)))
                    if np.isfinite(error).any()
                    else np.nan
                ),
                "degree_tv": degree["VAE"]["total_variation"],
                "hub_wasserstein": float(
                    wasserstein_distance(
                        degree["VAE"]["max_degrees"], degree["NetworkX"]["max_degrees"]
                    )
                ),
                "leader_tv": float(
                    abs(
                        hubs["VAE"]["leader_pmf"] - hubs["NetworkX"]["leader_pmf"]
                    ).sum()
                    / 2
                ),
                "diameter_wasserstein": geometry["VAE"]["wasserstein"]["diameter"],
                "kl_per_graph": kl,
                "parameters": sum(p.numel() for p in run.model.parameters()),
                "decoder_parameters": sum(
                    p.numel() for p in run.model.decoder.parameters()
                ),
                "output_head_multiply_adds": output_layer.in_features
                * output_layer.out_features,
                "duplicate_rate": 1
                - len({tuple(graph_to_parents(g)) for g in collections["VAE"]})
                / len(collections["VAE"]),
            }
            timing = timing_experiment(
                generators["vae"],
                num_nodes=config.num_nodes,
                threads=config.threads,
                show=False,
            )
            metrics["query_ms"] = timing["median_scalar_query_ms_including_latent"]
            metrics["graph_ms"] = timing["median_public_full_graph_ms"]
            record = {
                "condition": condition,
                "config": actual,
                "metrics": metrics,
                "reconstruction": reconstruction_experiment(run, show=False),
                "reinforcement": reinforcement,
                "geometry": geometry,
                "feature_mean_intervals": {
                    "VAE": _bootstrap(
                        generated_features, lambda x: x.mean(0), bootstrap_samples
                    ),
                    "NetworkX": _bootstrap(
                        reference_features, lambda x: x.mean(0), bootstrap_samples
                    ),
                },
                "feature_names": _FEATURE_NAMES,
            }
            records.append(record)
            if show:
                print(
                    f"N={config.num_nodes}, D={config.latent_dim}, seed={seed}:",
                    metrics,
                )
    if show:
        keys = (
            "ba_wasserstein",
            "reinforcement_rmse",
            "degree_tv",
            "diameter_wasserstein",
            "query_ms",
            "parameters",
        )
        _, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, key in zip(axes.flat, keys):
            for width in sorted({c.latent_dim for c in configurations}):
                selected = [r for r in records if r["config"].latent_dim == width]
                sizes = sorted({r["config"].num_nodes for r in selected})
                groups = [
                    [
                        r["metrics"][key]
                        for r in selected
                        if r["config"].num_nodes == size
                    ]
                    for size in sizes
                ]
                axis.errorbar(
                    sizes,
                    [np.mean(v) for v in groups],
                    yerr=[np.std(v) for v in groups],
                    marker="o",
                    label=f"D={width}",
                )
            axis.set(title=key, xlabel="N")
            axis.legend()
        plt.suptitle("Mean and standard deviation across training runs")
        plt.tight_layout()
        plt.show()
    return {
        "runs": records,
        "uncertainty": "Feature intervals resample graphs; scaling error bars show variation across training runs.",
    }


# Shared numerical operations. Displays deliberately stay in each experiment.


def _parent_collections(collections: GraphCollections, *, same_size: bool = True):
    """Validate arrival trees once at the experiment boundary and recover parents."""
    if not collections or any(len(graphs) == 0 for graphs in collections.values()):
        raise ValueError("provide nonempty graph collections")
    result = {}
    sizes = set()
    for name, graphs in collections.items():
        result[name] = []
        for index, graph in enumerate(graphs):
            if graph.is_directed() or graph.is_multigraph():
                raise ValueError(
                    f"{name}[{index}] must be a simple undirected arrival tree"
                )
            array = graph_to_parents(graph)
            if len(array) < 3 or not graph.has_edge(1, 2):
                raise ValueError(
                    f"{name}[{index}] needs N >= 3 and initial edge (1, 2)"
                )
            sizes.add(len(array))
            result[name].append(array)
    if same_size and len(sizes) != 1:
        raise ValueError("collections must contain graphs with the same N")
    return result


def _arrivals(parents):
    """Yield pre-arrival degrees and zero-based selected parent, beginning at t=3."""
    degrees = np.zeros(len(parents), dtype=int)
    degrees[:2] = 1
    for index in range(2, len(parents)):
        parent = parents[index] - 1
        yield degrees[:index].copy(), parent
        degrees[parent] += 1
        degrees[index] = 1


def _degree_history(parents):
    """Rows correspond to sizes 2..N; unborn nodes have degree zero."""
    history = np.zeros((len(parents) - 1, len(parents)), dtype=int)
    history[0, :2] = 1
    for index in range(2, len(parents)):
        history[index - 1] = history[index - 2]
        history[index - 1, parents[index] - 1] += 1
        history[index - 1, index] = 1
    return history


def _bootstrap(values, statistic, samples):
    """Percentile 95% intervals from whole-graph resampling, reproducibly.

    Undefined resamples remain NaN; each interval uses its finite replicates.
    An entirely undefined statistic has a NaN interval, never a fabricated zero.
    """
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(8675309)
    estimates = np.asarray(
        [
            statistic(values[rng.integers(len(values), size=len(values))])
            for _ in range(samples)
        ]
    )
    flat = estimates.reshape(samples, -1)
    interval = np.full((2, flat.shape[1]), np.nan)
    for column in range(flat.shape[1]):
        finite = flat[:, column][np.isfinite(flat[:, column])]
        if len(finite):
            interval[:, column] = np.quantile(finite, [0.025, 0.975])
    return interval.reshape((2,) + estimates.shape[1:])


def _scalar_comparison(samples):
    """Retain raw scalar samples alongside summaries and reference distances."""
    result = {}
    for name, values in samples.items():
        result[name] = {
            "values": values,
            "summary": {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values)),
                "quantiles": np.quantile(values, [0.025, 0.25, 0.75, 0.975]),
            },
            "wasserstein": (
                float(wasserstein_distance(values, samples["NetworkX"]))
                if "NetworkX" in samples
                else None
            ),
        }
    return result


def _node_indices(node_ids, n):
    nodes = np.asarray(tuple(node_ids), dtype=int)
    if len(nodes) == 0 or np.any(nodes < 1) or np.any(nodes > n):
        raise ValueError("node_ids must be within 1..N")
    return nodes - 1


def _split_time(fraction, n):
    split = int(n * fraction)
    if not 2 <= split < n:
        raise ValueError("split fraction must give a split time in 2..N-1")
    return split


def _covariance(values):
    return (
        np.atleast_2d(np.cov(values, rowvar=False))
        if len(values) > 1
        else np.full((values.shape[1], values.shape[1]), np.nan)
    )


def _correlation_matrix(values):
    covariance = _covariance(values)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    return np.divide(
        covariance, scale, out=np.full_like(covariance, np.nan), where=scale > 0
    )


def _tree_geometry(graph, root=1):
    """Exact tree geometry; all unordered distinct vertex pairs are included.

    Histograms and branch profiles are padded to N for averaging across graphs.
    Pair distances determine LCA depth via (depth(u)+depth(v)-distance(u,v))/2.
    """
    n = len(graph)
    if root not in graph:
        raise ValueError("root must belong to the graph")
    tree = nx.bfs_tree(graph, root)
    depth = nx.single_source_shortest_path_length(graph, root)
    descendants = {node: 0 for node in graph}
    for node in reversed(list(nx.topological_sort(tree))):
        descendants[node] = sum(
            descendants[child] + 1 for child in tree.successors(node)
        )
    branches = sorted(
        (descendants[child] + 1 for child in tree.successors(root)), reverse=True
    )
    distances, lca_depths, eccentricity = np.zeros(n), np.zeros(n), []
    ancestor_ages = np.zeros((n, n), dtype=int)
    for u, lengths in nx.all_pairs_shortest_path_length(graph):
        eccentricity.append(max(lengths.values()))
        for v, distance in lengths.items():
            if u < v:
                distances[distance] += 1
                lca_depths[(depth[u] + depth[v] - distance) // 2] += 1
            if u != v and depth[v] == depth[u] + distance:
                ancestor_ages[u - 1, v - 1] += 1
    depths = np.array([depth[i] for i in range(1, n + 1)])
    return {
        "depths": depths,
        "mean_depth": depths.mean(),
        "max_depth": depths.max(),
        "diameter": max(eccentricity),
        "radius": min(eccentricity),
        "branch_count": len(branches),
        "branch_sizes": np.pad(branches, (0, n - len(branches))),
        "largest_branch_fraction": branches[0] / (n - 1),
        "branch_balance": branches[1] / branches[0] if len(branches) > 1 else 0.0,
        "descendants": np.array([descendants[i] for i in range(1, n + 1)]),
        "distance_histogram": distances,
        "lca_depth_histogram": lca_depths,
        "ancestor_age_counts": ancestor_ages,
    }


_FEATURE_NAMES = (
    "root_degree",
    "max_degree",
    "leader_id",
    "leaf_fraction",
    "diameter",
    "largest_branch_size",
)


def _graph_features(graph):
    """Declared six-property feature vector; contains no seeds or source labels."""
    degrees = dict(graph.degree())
    without_root = graph.copy()
    without_root.remove_node(1)
    return np.array(
        [
            degrees[1],
            max(degrees.values()),
            min(degrees, key=lambda i: (-degrees[i], i)),
            sum(d == 1 for d in degrees.values()) / len(graph),
            nx.diameter(graph),
            max(map(len, nx.connected_components(without_root))),
        ],
        dtype=float,
    )
