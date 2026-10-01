# Diffusion-limited aggregation as a second target

## Why DLA

Barabási–Albert (BA) trees show the architecture works, but BA is cheap to simulate, so random access
into a BA graph has little practical value. Diffusion-limited aggregation (DLA) is a better second
target:

- It grows one step at a time, like BA: particles arrive one at a time, wander randomly, and stick to
  the existing cluster when they touch it.
- Each particle attaches to exactly one earlier particle, so the result is a tree labeled by arrival
  order. The existing parent-pointer decoder applies directly.
- It is expensive. Each arrival is a random walk whose outcome depends on the entire cluster built so
  far. There is a known theoretical result that computing a DLA cluster from its random numbers
  cannot be meaningfully sped up by running parts of it in parallel. That is motivation for a learned
  shortcut, not a result the model beats (see below).
- Its shape statistics are well known, giving a ready-made check. The number of particles within
  distance `r` of the center grows like `r^1.71` (for a filled disk it would be `r^2`). This exponent
  is the cluster's fractal dimension.

## Representation

Use 2D DLA on a square grid. Each particle occupies one square and attaches to an occupied neighbor
directly up, down, left, or right of it. Each arrival `t` is fully described by

$$
(\text{parent}_t,\ \text{direction}_t), \qquad \text{direction}_t \in \{\text{up}, \text{down}, \text{left}, \text{right}\}
$$

where direction is the side of the parent the new particle sits on. Positions are recovered by
following parent pointers back to the first particle and adding up the unit steps. Parents always
come earlier, so rebuilding in arrival order computes every position in one pass.

When a walker touches more than one occupied square, the simulator picks the parent at random among
them, using the seeded random generator.

Model changes relative to the BA baseline: the existing parent head stays, and one 4-way direction
head is added. Unlike the parent head, which can rule out future nodes from the query index alone,
the direction head cannot rule out already-occupied squares. Knowing which squares are occupied
requires the cluster state, which a single random-access query does not have. Particles landing on
occupied squares are therefore possible, and their rate is the main evaluation metric.

Known tradeoff: very large grid clusters take on the grid's shape, growing in a cross or diamond
pattern along the axes. At a few thousand particles the effect is small.

## Simulator

Write a custom Numba simulator rather than depend on an existing package.

- `dla-ideal-solver` is not classic DLA: it moves many walkers simultaneously, records no parents or
  exact arrival order, has no seed parameter, and wraps around the grid edges.
- `fogleman/dlaf` (C++) uses the correct one-particle-at-a-time algorithm and records parents, but
  places particles in continuous space rather than on a grid, so it is a design reference only.

Numba is not yet a project dependency: `uv add numba`.

### How the algorithm works

Plain DLA is simple:

1. Put one particle, the seed particle, at the center of the grid.
2. Release a walker somewhere far away.
3. Move the walker one square at a time, up, down, left, or right, chosen at random.
4. As soon as the walker is directly next to (up, down, left, or right of) an occupied square, it
   stops and becomes a particle. Its parent is that neighbor.
5. Repeat from step 2 until there are `N` particles.

That version is correct but very slow, because a walker spends almost all its time wandering in empty
space far from the cluster. Everything else in this section is about skipping that wandering without
changing the result.

Two facts about random walks make the skipping possible:

- **Starting on a circle.** A walker coming from very far away first touches any circle around the
  cluster at a uniformly random point on it. So instead of starting "infinitely far away", start the
  walker at a uniformly random point on a circle just outside the cluster. This gives the same result.
- **Jumping across empty circles.** A walker that starts at the center of a circle and wanders until
  it reaches the edge ends up at a uniformly random point on that edge. If the whole circle is known
  to be empty, nothing could have happened inside it, so the walker can jump straight to a random point
  on the edge in one move. This is exact in continuous space and a close approximation on a grid.

### Conventions

- **Grid.** A square `int32` array `grid[x, y]` of side `L`, filled with -1. An occupied square holds
  its particle's arrival index. The center square is `(c, c)` with `c = L // 2`.
- **Directions.** Encode as integers with lookup arrays:

  | code | name  | `DX` | `DY` |
  |------|-------|------|------|
  | 0    | right | +1   | 0    |
  | 1    | left  | -1   | 0    |
  | 2    | up    | 0    | +1   |
  | 3    | down  | 0    | -1   |

  With this ordering the opposite direction of `d` is `d ^ 1` (0↔1, 2↔3).
- **Direction label.** `direction[t]` is the side of the parent the new particle sits on:
  `(x_t, y_t) = (x_parent + DX[d], y_parent + DY[d])`.
- **Arrival index.** Particle 0 is the seed, with `parent[0] = -1` and `direction[0] = -1`. Particles
  are numbered in the order they stick.
- **Cluster radius.** `r_max` is the largest distance from the center to any particle, as a float.
  Starts at 0. Update it every time a particle sticks.

### Parameters

| name            | suggested default | meaning |
|-----------------|-------------------|---------|
| `n`             | —                 | number of particles, including the seed particle |
| `seed`          | —                 | random seed |
| `launch_margin` | 5                 | walkers start on a circle of radius `r_max + launch_margin` |
| `kill_factor`   | 20                | relaunch a walker that gets farther than `kill_factor * (r_max + launch_margin)` |
| `L`             | see below         | grid side length |

The walker's position is kept as two integers `(x, y)`, not stored in the grid. Grid lookups only
happen when the walker is close to the cluster (within `r_max + 3` of the center), so the grid only
has to cover the cluster plus a small border, not the kill circle. Pick `L` from a generous guess, such
as `L = 4 * ceil(n ** 0.6) + 64`, and have the simulator stop with an error if
`r_max + 8 >= L // 2`. Adjust the guess once you see real cluster sizes.

### Main loop

```
seed the random generator with `seed`   (inside the Numba function)
place particle 0 at (c, c); r_max = 0

for i in 1 .. n-1:
    launch walker
    loop:
        dist = distance from (x, y) to (c, c)

        if dist > r_kill:
            launch walker again; continue

        rho = dist - r_max - 2              # radius of a circle around the walker known to be empty
        if rho > 1:
            jump: pick angle uniformly in [0, 2π),
                  x = round(x + rho cos(angle)), y = round(y + rho sin(angle))
            continue

        # close to the cluster: grid lookups are safe here
        if any of the 4 neighbors of (x, y) is occupied:
            stick; break
        take one step in a random direction (0..3)

    record particle i; update r_max; check r_max against the grid size
```

Details for each part:

- **Launch.** Pick an angle uniformly in `[0, 2π)`. Set `R = r_max + launch_margin`. Start at
  `x = c + round(R cos(angle))`, `y = c + round(R sin(angle))`. Recompute `r_kill` here too, since
  `r_max` changes between particles.
- **Why `rho = dist - r_max - 2`.** Every particle is within `r_max` of the center, so every particle
  is at least `dist - r_max` from the walker. A circle of radius `dist - r_max - 2` around the walker
  therefore stays at least 2 squares from every particle. The walker can't stick anywhere inside it, and
  rounding the landing point to a square (off by at most about 0.7) can't put it on or next to a
  particle.
- **Stick.** Collect the occupied neighbors: for each direction `k`, look at
  `(x + DX[k], y + DY[k])`. If there is more than one, pick one uniformly at random. Call its direction
  from the walker `k`. Then `parent[i] = grid[x + DX[k], y + DY[k]]` and `direction[i] = k ^ 1`, since
  the new particle is on the opposite side of the parent. Set `grid[x, y] = i`, store `xs[i] = x`,
  `ys[i] = y`, and set `r_max = max(r_max, distance from (x, y) to center)`.
- **Walkers never land on a particle.** A walker next to a particle sticks before it can step onto it,
  and jumps never land within 2 of a particle. So no occupied-square check is needed when moving.
- **Why the kill circle is large.** Relaunching a walker on the launch circle assumes it came back from
  very far away. That is only accurate if it really had wandered far. A kill circle close to the
  cluster biases where particles land. Jumps make far-away walkers cheap, so a large kill circle costs
  little.

### Random numbers in Numba

- Call `np.random.seed(seed)` as the first line inside the `@njit` simulation function. Calling it from
  regular Python seeds NumPy's generator, not Numba's, and runs silently won't repeat.
- Inside `@njit`, use `np.random.random()` for angles (times `2π`) and `np.random.randint(0, 4)` for
  steps. For choosing among `m` occupied neighbors, use `np.random.randint(0, m)`.
- The seed-to-cluster mapping depends on the exact order of random draws. Changing the code, such as
  adding the optional speedup below, changes which cluster a given seed produces. That's fine, but
  record the simulator version alongside generated data.
- Numba's random state is per thread. When generating many clusters in parallel, use separate
  processes (for example `multiprocessing.Pool` or `joblib`) and pass each task its own seed.

### Suggested functions

- `simulate(n, seed, L, launch_margin, kill_factor)`: the `@njit` core. Returns
  `parent (int32[n])`, `direction (int8[n])`, `xs (int32[n])`, `ys (int32[n])`. Positions are relative
  to the grid; subtract `c` so the seed particle is at `(0, 0)`.
- `generate(n, seed, ...)`: plain Python wrapper that picks `L` and calls `simulate`.
- `rebuild_positions(parent, direction)`: plain NumPy, no simulation. Walks `t = 0..n-1`, setting
  `pos[t] = pos[parent[t]] + (DX[direction[t]], DY[direction[t]])`. This is what model outputs will go
  through later, so it's worth having early.
- `generate_dataset(n, seeds, path)`: runs many seeds in parallel and saves `parent` and `direction`
  (and optionally positions) per cluster to an `.npz` file along with the seeds.

### Optional speedup: medium jumps near the cluster

Once the basic version is validated, most remaining time is spent in single steps near the cluster,
especially in the empty gaps between branches. A coarse grid lets walkers jump there too.

- Keep a second array `blocks[bx, by]` over blocks of `B x B` squares (`B = 8` to start), counting
  particles per block. Increment the block's count whenever a particle sticks.
- In the close-to-cluster branch, before checking neighbors, look at the walker's block
  `(x // B, y // B)` and the 8 blocks around it. If all 9 are empty, every particle is at least `B`
  squares away, so jump on a circle of radius `B - 2` exactly as in the far-away case.
- Validate this against the version without it (see below). It should not change the statistics.

## Validation of the simulator

Before training on its output:

1. **Repeatability.** Same seed gives identical arrays. Different seeds give different arrays.
2. **Structure.** `parent[0] == -1`. For `t >= 1`, `0 <= parent[t] < t`. `rebuild_positions` matches
   `xs, ys` exactly. No two particles share a square. Every particle is exactly one square from its
   parent.
3. **Fractal dimension.** Take the first `k` particles of a cluster for many `k` (for example 50 values
   spaced evenly on a log scale from 100 to `n`). The first `k` particles of a DLA cluster are
   themselves a DLA cluster of size `k`. For each, compute the root-mean-square distance of those
   particles from their average position (the radius of gyration). Fit a straight line to
   `log(radius)` versus `log(k)`. The slope should be about `1 / 1.71 ≈ 0.585`, so `1 / slope` gives
   the fractal dimension. Average over about 20 clusters of a few thousand particles. Grid clusters at
   this size typically give roughly 1.65 to 1.72.
4. **Jumps don't change results.** Add a flag that disables all jumps (every move is a single step) and
   compare against the normal version on small clusters (say 500 particles, 50 seeds each). Fractal
   dimension and average radius should agree within noise. Do the same for the optional block jumps.
5. **Kill circle doesn't change results.** Compare `kill_factor` 20 against 100 the same way.
6. **Look at it.** Scatter-plot a cluster's positions, colored by arrival index. It should look like a
   branching, feathery shape with early particles near the center and no compact blobs.
7. **Speed.** Time `generate` for `n` = 1,000, 10,000, and 100,000. This is the baseline the model is
   later compared against.

## Evaluation of the model

- **Collision rate.** Fraction of rebuilt particles that land on an already-occupied square. The
  simulator produces none; this is the most direct measure of whether independent queries agree on
  one geometry.
- **Distribution match.** Fractal dimension and tree statistics (depth, branching) against simulator
  clusters. Include the independent-parent baseline, as with BA.
- **Query consistency.** Same checks as BA: answers do not depend on query order or on other queries.
- **Speed.** Time per query versus simulating up to arrival `t`, measured against the optimized
  simulator, not a slow Python loop. Report where the crossover falls.

## Staged plan

1. Simulator plus validation.
2. Tree only: train on parent arrays, reusing the BA pipeline unchanged. A valid intermediate result,
   but not sufficient alone, since it drops the geometry that makes DLA hard.
3. Add the direction head.
4. Measure collision rate, fractal dimension, and speed.

## Framing

Closest prior work and how this differs, in brief:

- Exact random-access generation (Even et al. 2017; Biswas, Rubinfeld and Yodpinyanee 2020): same goal,
  but hand-derived algorithms for simple processes only.
- Networks that map a shared random code plus a query coordinate to an output (GASP, Functa, Neural
  Processes): same architecture shape, but applied to images and shapes, not step-by-step processes.
- Networks that output an edge probability from two node coordinates: edges are independent of each
  other, so no dependence on history.
- GraphRNN, NetGAN: learned graph generators, but they must build the graph in order, with no random
  access.

The architecture alone reads as a conditional VAE. The contribution is the problem (learned random
access into step-by-step random processes), the evaluation (consistency, distribution match, speed),
and the demonstration on a process with no known exact shortcut.
