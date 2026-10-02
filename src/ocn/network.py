"""Thin helpers around PyOCN for fitting, inspecting, and timing single OCNs."""

import time
from collections.abc import Iterable

import networkx as nx
import numpy as np
import PyOCN as po

# Row/column offsets for the 8 neighbors, indexed clockwise from north. Index -1 marks a root.
DIRECTIONS = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=np.int64
)


def fit_ocn(
    size: int | tuple[int, int],
    seed: int,
    *,
    net_type: str = "V",
    wrap: bool = False,
    gamma: float = 0.5,
    cooling_rate: float = 1.0,
) -> po.OCN:
    """Build an OCN from a predefined initial network and anneal it with PyOCN's default schedule."""
    dims = (size, size) if isinstance(size, int) else tuple(size)
    ocn = po.OCN.from_net_type(net_type, dims=dims, gamma=gamma, random_state=seed, wrap=wrap)
    ocn.fit(cooling_rate=cooling_rate)
    return ocn


def rasters(ocn: po.OCN) -> dict[str, np.ndarray]:
    """Return the energy, drained-area, and elevation rasters as a dict of (rows, cols) arrays."""
    energy, area, elevation = ocn.to_numpy(unwrap=False)
    return {"energy": energy, "drained_area": area, "elevation": elevation}


def flow_directions(dag: nx.DiGraph, dims: tuple[int, int]) -> np.ndarray:
    """Return a (rows, cols) grid of downstream direction indices into DIRECTIONS, -1 for roots.

    This is the OCN written as a parent array over grid cells: each cell's parent is one of its
    8 neighbors. Periodic edges are resolved by wrapping the offset back into [-1, 1].
    """
    rows, cols = dims
    lookup = {tuple(offset): index for index, offset in enumerate(DIRECTIONS.tolist())}
    directions = np.full(dims, -1, dtype=np.int64)
    for node, child in dag.edges:
        r0, c0 = dag.nodes[node]["pos"]
        r1, c1 = dag.nodes[child]["pos"]
        dr = (r1 - r0 + 1) % rows - 1
        dc = (c1 - c0 + 1) % cols - 1
        directions[r0, c0] = lookup[(dr, dc)]
    return directions


def area_exceedance(area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted drained areas and P(A >= a), the standard OCN scaling check."""
    values = np.sort(area[np.isfinite(area)].ravel())
    probability = 1.0 - np.arange(len(values)) / len(values)
    return values, probability


def time_fit(
    sizes: Iterable[int], seeds: Iterable[int], *, net_type: str = "V", wrap: bool = False
) -> list[dict]:
    """Time construction, annealing, and graph export for each (size, seed) pair."""
    records = []
    for size in sizes:
        for seed in seeds:
            start = time.perf_counter()
            ocn = po.OCN.from_net_type(net_type, dims=(size, size), random_state=seed, wrap=wrap)
            built = time.perf_counter()
            ocn.fit()
            fitted = time.perf_counter()
            ocn.to_digraph()
            exported = time.perf_counter()
            records.append(
                {
                    "size": size,
                    "cells": size * size,
                    "seed": seed,
                    "iterations": int(ocn.history[-1, 0]),
                    "build_s": built - start,
                    "fit_s": fitted - built,
                    "export_s": exported - fitted,
                    "energy": ocn.energy,
                }
            )
    return records
