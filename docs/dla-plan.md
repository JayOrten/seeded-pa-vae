# DLA implementation plan

This plan adapts the BA prototype in `src/svae/` to DLA trees. The DLA code lives in its own package,
`src/dla/`, with no imports from `svae`. Code is copied from `svae` and changed where DLA differs.

The query interface is `query(seed, t) -> (parent, direction)`: given an arrival index, return the
node's parent and which side of the parent it sits on. See `docs/dla.md` for the representation and
the simulator.

Start with small graphs. Expand only after the small ones work.

## Layout

```
src/dla/
  simulate.py   exists; add dataset building and position rebuilding
  config.py     copy of svae/config.py with DLA profiles
  train.py      copy of svae/train.py; the model gains a direction head
  generate.py   copy of svae/generate.py; queries return (parent, direction)
  evaluate.py   new and small; only experiments that apply to DLA
notebooks/dla-training.ipynb
tests/test_dla_simulate.py
tests/test_dla_train.py
tests/test_dla_generate.py
tests/test_dla_evaluate.py
```

## Steps

Each step ends with a check-in before the next one starts.

### 1. Dataset helpers (`simulate.py`)

- Add `make_dataset(num_nodes, num_graphs, first_seed)`. It returns two tensors, `parents` with shape
  `(graphs, nodes)` and `directions` with the same shape. It runs seeds in separate processes, because
  Numba's random state is per thread.
- Add `rebuild_positions(parents, directions)`. It returns each node's position and the number of
  nodes that land on an occupied square. `parents_to_graph` raises on the first collision; evaluation
  needs to count them instead.
- Tests: the same seed gives the same arrays, simulator output has no collisions, and every node is
  one step from its parent.

### 2. Config (`config.py`)

- Same `Config` fields as `svae`.
- Profiles: `smoke` (16 nodes) first, then 64, 128, and 256 nodes.

### 3. Model and training (`train.py`)

- Encoder input: a one-hot parent and a one-hot direction for each node.
- Decoder: the existing parent head plus a 4-way direction head.
- Loss: parent cross-entropy plus direction cross-entropy, plus the KL term as before.
- Query ranges: node 2's parent is always node 1, but its direction is random. The parent head
  covers nodes 3..N, as in BA. The direction head covers nodes 2..N.
- Checkpoints use a `dla_` prefix so they do not overwrite BA checkpoints in `artifacts/`.
- First version: the parent and direction heads are independent given the latent and the query.
  A second version, where the direction head also receives the parent, comes later as a comparison
  (see Later work).

### 4. Generators (`generate.py`)

- `SeededDLAGenerator.query(seed, t) -> (parent, direction)`. Parent and direction each use their own
  seeded random stream.
- `IndependentDLAGenerator`: the baseline. It samples each node's parent and direction from their
  per-node frequencies in the training data, with no shared latent.

### 5. Evaluation (`evaluate.py`)

Written new rather than copied. Most of `svae/evaluate.py` measures degree reinforcement, which does
not apply to DLA. Copy only reconstruction, timing, and query-order consistency. Add:

- **Collision rate:** the fraction of rebuilt nodes that land on an occupied square. The simulator
  produces none.
- **Fractal dimension:** from the radius of gyration over arrival prefixes, as in `docs/dla.md`.
- **Tree depth and branching:** compared against simulator trees.
- **Speed:** time per query compared with simulating up to node `t`.

Each metric is reported for the VAE, the independent baseline, and the simulator.

### 6. Notebook (`notebooks/dla-training.ipynb`)

Train at each size, run the evaluation, and compare against the baseline.

## Known limit: encoder size

The copied encoder takes a flattened one-hot of every parent, about `N²` inputs. Its first layer then
has about `N² × hidden` weights:

| Nodes | First-layer weights (hidden 512) |
|---|---|
| 256 | ~33M |
| 512 | ~134M |
| 1024 | ~535M |

The copied architecture is practical up to about 256–512 nodes. Fractal dimension needs a few thousand
particles to measure well, so at 256 it will be noisy. This round keeps the architecture unchanged.
Replacing the encoder is the first change needed to go larger.

## Later work

- **Direction conditioned on parent.** Feed the chosen parent into the direction head, so the model
  learns p(direction | parent). During training, feed it the true parent. Compare collision rate
  against the independent version.
- **A smaller encoder.** Needed for graphs beyond about 512 nodes.
- **A hierarchical model.** After the flat model has been measured at several sizes.
