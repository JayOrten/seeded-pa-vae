# Seeded preferential-attachment VAE

An experimental PyTorch baseline for learning a deterministic, random-access interface to sampled
preferential-attachment trees:

```python
generator.parent(seed=123, query=10)
```

Start with the self-contained [`seeded_pa_vae_demo.ipynb`](notebooks/seeded_pa_vae_demo.ipynb). The notebook
simulates arrival-labeled trees with NetworkX, trains a query-conditioned VAE, exposes the frozen
seed-and-query API, compares it with an independent-parent baseline, and evaluates reconstruction,
distributional behavior, reinforcement, consistency, and timing.

## Development setup

Install the project and its notebook dependencies into a managed `.venv`:

```bash
uv sync
```

Run Python or project commands inside the environment with `uv run`, without activating it:

```bash
uv run python
```

Point your notebook editor at `.venv/bin/python`. If it requires a registered Jupyter kernel, run:

```bash
uv run python -m ipykernel install --user \
    --name seeded-pa-vae \
    --display-name "Python (seeded-pa-vae)"
```

Notebooks can live anywhere in the repository (or outside it) and import package modules without
modifying `sys.path`:

```python
from svae import Config, PROFILES
from svae import simulate
```

The intended notebook workflow is deliberately small:

```python
from svae import PROFILES, generators_from_run, train
from svae.evaluate import (
    graph_samples_experiment,
    graph_statistics_experiment,
    reconstruction_experiment,
    timing_experiment,
)

run = train(PROFILES["smoke"])
generators = generators_from_run(run)
reconstruction = reconstruction_experiment(run)
collections = graph_samples_experiment(
    run, generators["vae"], generators["independent"]
)
statistics = graph_statistics_experiment(
    collections, num_nodes=run.config.num_nodes
)
timings = timing_experiment(
    generators["vae"],
    num_nodes=run.config.num_nodes,
    threads=run.config.threads,
)
```

The experiments above reproduce the original demo notebook and return their underlying results.
The additional evaluation-plan experiments are available through `from svae import evaluate as ev`.
See [the evaluation API guide](docs/evaluation-api.md) for individual calls, returned measurements,
statistical conventions, latent ablations, and training sweeps. Plotting and printing are enabled
by default; pass `show=False` in automated tests.

The project is installed editably, so changes under `src/svae/` are available the next time a module
is imported. Restart the notebook kernel or use IPython's autoreload extension for already-imported
modules. Commit `uv.lock` so collaborators and notebook runs resolve the same dependency versions.

The [research documentation](docs/README.md) records the feasibility argument, important theoretical
corrections, architecture, experiment design, prior art, and novelty boundary. The detailed original
implementation handoff is in [`seeded_pa_vae_notebook_plan.md`](seeded_pa_vae_notebook_plan.md).

The checked-in notebook defaults to the GPU-oriented `large` configuration: 32,768 training graphs,
4,096 validation and test graphs, a 128-dimensional latent, width 512, and 100 epochs. Change
`RUN_PROFILE` to `smoke` when validating a new environment. The selected model is saved under
`artifacts/`; use repeated runs before interpreting model quality.
