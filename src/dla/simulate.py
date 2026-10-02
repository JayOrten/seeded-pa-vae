import math
import os
from multiprocessing import Pool

import numpy as np
import torch
from numpy.typing import NDArray
from numba import njit
# Representations of the DLA graph:
# list[(x,y) coordinates on grid] - I don't know if we'll even need this one, except for display
# list[(parent, direction)] - direction is one of four directions, parent is the arrival id of the parent node
# so nodes are id'd by their arrival number, just like the BA graphs.

# unit step for each direction code: +x, -x, +y, -y
OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)


LAUNCH_MARGIN = 5  # walkers start on a circle of radius r_max + LAUNCH_MARGIN
KILL_FACTOR = 20  # relaunch walkers that get farther than KILL_FACTOR * launch radius


@njit(cache=True)  # cached so worker processes skip recompiling
def simulate_dla(
    num_nodes: int, simulator_seed: int
) -> tuple[NDArray[np.int32], NDArray[np.int8]]:
    """Simulate one DLA graph"""
    np.random.seed(simulator_seed)  # ensure reproducibility
    parents = np.zeros(num_nodes, dtype=np.int32)
    directions = np.zeros(num_nodes, dtype=np.int8)

    # generous approximation of the grid size needed to fit the cluster
    grid_size = 4 * math.ceil(num_nodes**0.6) + 64

    # place initial node at the center of the grid
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    center = grid_size // 2
    grid[center, center] = 1
    parents[0] = -1  # root node has no parent
    directions[0] = -1  # root node has no direction
    r_max = 0.0  # distance from the center to the farthest node

    # data structure here:
    # grid is 0 where there is no node, and the arrival ID of the node where there is one
    # parents is an array of length num_nodes, where parents[i] is the arrival ID of node i's parent
    # directions is an array of length num_nodes, where directions[i] is the direction from
    # node i's parent to node i, encoded as 0, 1, 2, 3 for +x, -x, +y, -y

    neighbor_directions = np.zeros(4, dtype=np.int8)
    for node_id in range(1, num_nodes):
        # launch a walker on a circle just outside the cluster
        launch_radius = r_max + LAUNCH_MARGIN
        kill_radius = KILL_FACTOR * launch_radius
        angle = 2 * math.pi * np.random.random()
        x = center + round(launch_radius * math.cos(angle))
        y = center + round(launch_radius * math.sin(angle))

        # random walk until the walker is next to the cluster
        while True:
            dist = math.hypot(x - center, y - center)
            if dist > kill_radius:
                angle = 2 * math.pi * np.random.random()
                x = center + round(launch_radius * math.cos(angle))
                y = center + round(launch_radius * math.sin(angle))
                continue

            # circle around the walker that is known to be empty: jump to its edge
            rho = dist - r_max - 2
            if rho > 1:
                angle = 2 * math.pi * np.random.random()
                x = round(x + rho * math.cos(angle))
                y = round(y + rho * math.sin(angle))
                continue

            # close to the cluster, so (x, y) and its neighbors are inside the grid
            num_neighbors = 0
            for k in range(4):
                if grid[x + OFFSETS[k, 0], y + OFFSETS[k, 1]] != 0:
                    neighbor_directions[num_neighbors] = k
                    num_neighbors += 1
            if num_neighbors > 0:
                break

            step = np.random.randint(0, 4)
            x += OFFSETS[step, 0]
            y += OFFSETS[step, 1]

        # stick: pick one occupied neighbor as the parent
        k = neighbor_directions[np.random.randint(0, num_neighbors)]
        parents[node_id] = grid[x + OFFSETS[k, 0], y + OFFSETS[k, 1]]
        directions[node_id] = k ^ 1  # the new node is on the opposite side of its parent
        grid[x, y] = node_id + 1  # arrival ID is node_id + 1

        r_max = max(r_max, math.hypot(x - center, y - center))
        if r_max + 8 >= center:
            raise ValueError("cluster outgrew the grid; increase grid_size")

    return parents, directions


def _simulate_one(task: tuple[int, int]) -> tuple[NDArray[np.int32], NDArray[np.int8]]:
    return simulate_dla(*task)


def make_dataset(
    num_nodes: int, num_graphs: int, *, first_seed: int, processes: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate consecutive seeds and return ``(parents, directions)`` tensors of shape (graphs, nodes).

    Seeds run in separate processes because Numba's random state is per thread.
    """
    if num_graphs < 1:
        raise ValueError("num_graphs must be positive")
    tasks = [(num_nodes, first_seed + offset) for offset in range(num_graphs)]
    processes = min(processes or os.cpu_count() or 1, num_graphs)
    if processes == 1:
        results = [_simulate_one(task) for task in tasks]
    else:
        with Pool(processes) as pool:
            results = pool.map(_simulate_one, tasks, chunksize=max(1, num_graphs // (4 * processes)))
    parents = np.stack([result[0] for result in results]).astype(np.int64)
    directions = np.stack([result[1] for result in results]).astype(np.int64)
    return torch.from_numpy(parents), torch.from_numpy(directions)


def validate_dla(parents, directions) -> tuple[np.ndarray, np.ndarray]:
    """Validate one ``(parents, directions)`` pair and return them as int64 arrays."""
    parents = np.asarray(parents)
    directions = np.asarray(directions)
    if parents.ndim != 1 or parents.shape != directions.shape or len(parents) < 2:
        raise ValueError("parents and directions must be equal-length 1D arrays of length >= 2")
    if parents[0] != -1 or directions[0] != -1:
        raise ValueError("the root must have parent -1 and direction -1")
    ids = np.arange(1, len(parents))
    if np.any(parents[1:] < 1) or np.any(parents[1:] > ids):
        raise ValueError("node at index i must have a parent in 1..i")
    if np.any(directions[1:] < 0) or np.any(directions[1:] > 3):
        raise ValueError("directions must be in 0..3")
    return parents.astype(np.int64, copy=False), directions.astype(np.int64, copy=False)


def rebuild_positions(parents, directions) -> tuple[np.ndarray, int]:
    """Place each node one step from its parent and count nodes landing on an occupied square.

    Returns positions of shape (nodes, 2) with the root at (0, 0), and the collision count. A
    colliding node keeps its position, so later nodes attached to it are placed as well.
    """
    parents, directions = validate_dla(parents, directions)
    positions = np.zeros((len(parents), 2), dtype=np.int64)
    occupied = {(0, 0)}
    collisions = 0
    for index in range(1, len(parents)):
        positions[index] = positions[parents[index] - 1] + OFFSETS[directions[index]]
        cell = (int(positions[index, 0]), int(positions[index, 1]))
        if cell in occupied:
            collisions += 1
        occupied.add(cell)
    return positions, collisions


def parents_to_graph(
    parents: NDArray[np.int32], directions: NDArray[np.int8]
) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
    """Place each node on the grid, labelled by arrival ID.

    Follows the BA parent-array convention: ``parents[i]`` is the 1-indexed
    arrival ID of node ``i + 1``'s parent, and ``parents[0]`` (the root) is unused.
    """
    coordinates, collisions = rebuild_positions(parents, directions)
    if collisions:
        raise ValueError("two nodes occupy the same grid cell")
    # shift so the cluster starts at (0, 0); walks can go negative
    coordinates -= coordinates.min(axis=0)
    width, height = coordinates.max(axis=0) + 1
    grid = np.zeros((width, height), dtype=np.int32)
    grid[coordinates[:, 0], coordinates[:, 1]] = np.arange(1, len(parents) + 1)
    return grid
