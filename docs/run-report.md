# Smoke run report

The notebook's code cells were executed top-to-bottom on 2026-09-09 using the `smoke`
configuration: `N=16`, latent width 8, hidden width 64, 128 training graphs, 32 validation graphs,
64 test graphs, batch size 32, and two epochs. The environment used CPU float32 with two PyTorch
threads, Python on Linux, NumPy 2.4.2, PyTorch 2.10.0+cu128, NetworkX 3.6.1, and Matplotlib 3.10.8.

All simulator round trips, tensor/mask checks, finite-loss and finite-gradient checks, optimizer-change
check, deterministic query schedules, interleaved seeds, state reload, graph reassembly, and generated
tree assertions passed.

Observed smoke metrics (not research conclusions):

| Metric | Result |
|---|---:|
| Mean-latent test parent accuracy | 0.2533 |
| Mean-latent test cross-entropy/query | 1.9793 |
| NetworkX reinforcement correlation | 0.3870 |
| VAE reinforcement correlation | 0.1982 |
| Independent-parent reinforcement correlation | 0.0582 |
| Median scalar VAE query, including latent | 0.081 ms |
| Median full graph through scalar API | 1.195 ms |
| Mean NetworkX full graph | 0.088 ms |

The VAE smoke samples had lower hub degree and leaf fraction than NetworkX references. Two epochs are
only an execution check, so this is weak fidelity rather than evidence for or against the research
hypothesis. The full demo configuration remains unexecuted.

The notebook now defaults to the subsequently added `large` profile. That profile has not been run in
this environment because its GPU is not exposed here; this report remains the historical smoke result.

The environment lacks Jupyter/nbformat, so validation used a lightweight executor over the notebook's
code cells with Matplotlib's noninteractive backend. Consequently the checked-in notebook has clean
cells rather than persisted rich outputs. Running it in Jupyter will render the prescribed figures.
