"""Two-level chunked river prototype: a coarse OCN routes chunks, local OCNs fill each chunk.

The coarse OCN has one cell per chunk. Its flow directions say which neighbor each chunk drains
into. Every chunk is then generated alone, from the world seed and its own coordinates:

1. Its outlet sits on the border facing its coarse downstream neighbor. The exact border cell is
   hashed from the shared edge, so the downstream chunk computes the same cell as its inlet.
2. A local OCN is annealed with a single root at that outlet.
3. Water entering from upstream chunks is added along the local path from each inlet to the
   outlet. Its amount comes from the coarse drained area, not from fitting the upstream chunk.

Only the coarse OCN is global. A chunk never needs its neighbors to be fitted first.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import PyOCN as po

from .network import DIRECTIONS, fit_ocn, flow_directions

def _hash_int(*values: int) -> int:
    return int(np.random.SeedSequence([int(v) & 0xFFFFFFFF for v in values]).generate_state(1)[0])


def border_cell(direction: int, offset: int, chunk_size: int) -> tuple[int, int]:
    """Return the local cell on the chunk border facing ``direction``.

    Cardinal borders use ``offset`` along the edge. Diagonal borders use the corner.
    """
    last = chunk_size - 1
    dr, dc = DIRECTIONS[direction]
    if dr != 0 and dc != 0:
        return (0 if dr < 0 else last, 0 if dc < 0 else last)
    if dr != 0:
        return (0 if dr < 0 else last, offset)
    return (offset, 0 if dc < 0 else last)


def bfs_tree(dims: tuple[int, int], root: tuple[int, int]) -> nx.DiGraph:
    """Build a valid initial spanning tree that drains every cell to ``root`` over 4-neighbors."""
    rows, cols = dims
    dag = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            dag.add_node(r * cols + c, pos=(r, c))
    seen = {root}
    queue = deque([root])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                seen.add((nr, nc))
                dag.add_edge(nr * cols + nc, r * cols + c)
                queue.append((nr, nc))
    return dag


@dataclass
class Chunk:
    coords: tuple[int, int]
    outlet: tuple[int, int]
    outlet_direction: int
    inlets: dict[tuple[int, int], tuple[int, int]]  # upstream chunk -> local inlet cell
    directions: np.ndarray  # local flow directions, -1 at the outlet
    local_area: np.ndarray  # drained area from this chunk's own cells only
    area: np.ndarray  # local area plus water routed in from upstream chunks
    fit_seconds: float = 0.0


@dataclass
class ChunkedWorld:
    world_seed: int
    world_chunks: int = 8  # coarse grid side, in chunks
    chunk_size: int = 32  # chunk side, in cells
    coarse: po.OCN = field(init=False, repr=False)
    coarse_directions: np.ndarray = field(init=False, repr=False)
    coarse_area: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.coarse = fit_ocn(self.world_chunks, _hash_int(self.world_seed, 0))
        dims = self.coarse.dims
        self.coarse_directions = flow_directions(self.coarse.to_digraph(), dims)
        # The coarse root sits on the map border. Send it off the map through a cardinal edge.
        (root_r, root_c), = np.argwhere(self.coarse_directions < 0)
        for direction in (0, 2, 4, 6):
            dr, dc = DIRECTIONS[direction]
            if not (0 <= root_r + dr < dims[0] and 0 <= root_c + dc < dims[1]):
                self.coarse_directions[root_r, root_c] = direction
                break
        self.coarse_area = self.coarse.to_numpy(unwrap=False)[1]

    def _edge_offset(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Shared border offset in 1..chunk_size-2, identical when computed from either side."""
        (r0, c0), (r1, c1) = sorted([a, b])
        return 1 + _hash_int(self.world_seed, 1, r0, c0, r1, c1) % (self.chunk_size - 2)

    def _upstream(self, coords: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        """Return (direction toward neighbor, neighbor) for each coarse neighbor draining here."""
        r, c = coords
        n = self.world_chunks
        upstream = []
        for direction, (dr, dc) in enumerate(DIRECTIONS.tolist()):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                back = DIRECTIONS[self.coarse_directions[nr, nc]]
                if (nr + back[0], nc + back[1]) == (r, c):
                    upstream.append((direction, (nr, nc)))
        return upstream

    def chunk(self, r: int, c: int) -> Chunk:
        """Generate one chunk from the world seed and its coordinates alone."""
        size = self.chunk_size
        out_dir = int(self.coarse_directions[r, c])
        dr, dc = DIRECTIONS[out_dir]
        outlet = border_cell(out_dir, self._edge_offset((r, c), (r + dr, c + dc)), size)

        start = time.perf_counter()
        local = po.OCN.from_digraph(
            bfs_tree((size, size), outlet), random_state=_hash_int(self.world_seed, 2, r, c)
        )
        local.fit()
        fit_seconds = time.perf_counter() - start

        directions = flow_directions(local.to_digraph(), (size, size))
        local_area = local.to_numpy(unwrap=False)[1]
        area = local_area.copy()
        inlets = {}
        for direction, neighbor in self._upstream((r, c)):
            cell = border_cell(direction, self._edge_offset((r, c), neighbor), size)
            inlets[neighbor] = cell
            inflow = self.coarse_area[neighbor] * size * size
            # Every local cell drains to the outlet, so this walk always terminates there.
            cr, cc = cell
            while True:
                area[cr, cc] += inflow
                step = directions[cr, cc]
                if step < 0:
                    break
                cr, cc = cr + DIRECTIONS[step][0], cc + DIRECTIONS[step][1]

        return Chunk((r, c), outlet, out_dir, inlets, directions, local_area, area, fit_seconds)

    def assemble(self, chunks: dict[tuple[int, int], Chunk]) -> tuple[np.ndarray, nx.DiGraph]:
        """Stitch generated chunks into one area raster and one cell-level flow graph."""
        size = self.chunk_size
        n = self.world_chunks
        area = np.full((n * size, n * size), np.nan)
        graph = nx.DiGraph()
        for (r, c), chunk in chunks.items():
            r0, c0 = r * size, c * size
            area[r0 : r0 + size, c0 : c0 + size] = chunk.area
            for lr in range(size):
                for lc in range(size):
                    node = (r0 + lr, c0 + lc)
                    graph.add_node(node, area=chunk.area[lr, lc])
                    step = chunk.directions[lr, lc]
                    if step < 0:
                        step = chunk.outlet_direction
                    target = (node[0] + DIRECTIONS[step][0], node[1] + DIRECTIONS[step][1])
                    if 0 <= target[0] < n * size and 0 <= target[1] < n * size:
                        graph.add_edge(node, target)
        return area, graph
