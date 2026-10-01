import numpy as np
from numpy.typing import NDArray
from numba import njit
import math
# Representations of the DLA graph:
# list[(x,y) coordinates on grid] - I don't know if we'll even need this one, except for display
# list[(parent, direction)] - direction is one of four directions, parent is the arrival id of the parent node
# so nodes are id'd by their arrival number, just like the BA graphs.

# unit step for each direction code: +x, -x, +y, -y
OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)


LAUNCH_MARGIN = 5  # walkers start on a circle of radius r_max + LAUNCH_MARGIN
KILL_FACTOR = 20  # relaunch walkers that get farther than KILL_FACTOR * launch radius


@njit
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


def parents_to_graph(
    parents: NDArray[np.int32], directions: NDArray[np.int8]
) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
    """Place each node on the grid, labelled by arrival ID.

    Follows the BA parent-array convention: ``parents[i]`` is the 1-indexed
    arrival ID of node ``i + 1``'s parent, and ``parents[0]`` (the root) is unused.
    """
    num_nodes = len(parents)
    coordinates = np.zeros((num_nodes, 2), dtype=np.int32)
    for index in range(1, num_nodes):
        direction = directions[index]
        if not 0 <= direction < 4:
            raise ValueError(f"Invalid direction {direction}")
        coordinates[index] = coordinates[parents[index] - 1] + OFFSETS[direction]

    # shift so the cluster starts at (0, 0); walks can go negative
    coordinates -= coordinates.min(axis=0)
    width, height = coordinates.max(axis=0) + 1
    grid = np.zeros((width, height), dtype=np.int32)
    grid[coordinates[:, 0], coordinates[:, 1]] = np.arange(1, num_nodes + 1)
    if np.count_nonzero(grid) != num_nodes:
        raise ValueError("two nodes occupy the same grid cell")
    return grid
