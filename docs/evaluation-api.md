# Running the evaluation plan

The numbered sections of `src/svae/evaluate.py` follow `evaluation-plan.md`.
Every public experiment computes and returns data, and displays its own results
when `show=True` (the default). Use `show=False` to collect results silently.
The original four demo functions still provide the original notebook plots.

```python
from svae import evaluate as ev
from svae import PROFILES, train, generators_from_run

run = train(PROFILES["demo"])
generators = generators_from_run(run)
collections = ev.graph_samples_experiment(
    run, generators["vae"], generators["independent"]
)
likelihood = ev.ba_sequence_likelihood_experiment(collections)
kernel = ev.attachment_kernel_experiment(collections)
reinforcement = ev.reinforcement_experiment(collections)
```

## Common conventions

- Collections map method names to lists of graphs. Use `"NetworkX"` for the
  reference; reference distances are absent (`None` or an empty mapping) when it
  is missing. Distribution comparison accepts a different `reference_name`.
- Graphs must be simple, undirected arrival-labeled trees on `1..N`, with initial
  edge `(1, 2)` and exactly one older neighbor per arrival. These experiments
  implement **m=1**. The common input check recovers parent arrays once per
  experiment. Numerical helpers assume that validated representation.
- Except for normalized sequence scores, comparison collections must have the
  same N. The new experiments require N >= 3. Default early-node selections
  assume N >= 10; pass explicit node IDs for smaller graphs.
- Degree history row `t-2` contains degrees at size t. Columns are node ID minus
  one; unborn nodes have degree zero and are excluded from temporal correlations.
- Bootstrap intervals are percentile 95% intervals, resampling entire graphs.
  The deterministic evaluation RNG seed is 8675309. Undefined correlations and
  bootstrap estimates are NaN; interval bounds use finite replicates. Sparse
  bins can produce unreliable or degenerate intervals and are marked explicitly.
- Plot bands in degree-by-age and decoder-entropy experiments describe sample
  variation (2.5–97.5 percentiles), not uncertainty in an estimated mean.
- Results contain NumPy arrays, including individual graph measurements where
  useful. They are intended for notebook analysis, not direct JSON serialization.

## Experiments and returned data

| Function | Main returned measurements |
| --- | --- |
| `structural_invariants_experiment` | Pass counts and indexed invariant failures |
| `ba_sequence_likelihood_experiment` | Log scores, quantiles, SD, Wasserstein distance |
| `attachment_kernel_experiment` | Grid likelihood curve, estimated alpha, bootstrap interval, boundary flag |
| `selected_parent_degree_experiment` | Observed/expected selections, eligible opportunity counts, ratios |
| `degree_distribution_experiment` | PMF, CCDF, low-degree counts, maxima, tail mass, reference TV |
| `finite_size_degree_counts_experiment` | Exact expected counts, residuals, bootstrap intervals |
| `degree_by_age_experiment` | Means, SDs, quantiles, early-node samples, trajectories, rescaled degrees |
| `reinforcement_experiment` | Node-by-split correlations, intervals, reference errors |
| `conditional_future_growth_experiment` | Per-node degree bins with counts, mean gain, intervals, reliability flags |
| `attachment_residuals_experiment` | Opportunity-weighted residual means/counts by state variable; pooled lag-one correlation |
| `hub_dynamics_experiment` | Top degrees, gaps, leader identities/changes, first lead time, persistence, transition counts/probabilities |
| `early_node_covariance_experiment` | Covariance/correlation matrices and intervals; covariance Frobenius distance |
| `joint_attachment_experiment` | Joint-to-marginal-product ratios, counts, intervals, reliability flags |
| `hub_concentration_experiment` | Incident edge shares, Herfindahl measure, entropy, scalar distances |
| `tree_geometry_experiment` | Depths, distances, diameter/radius, branch profiles, descendants, ancestor ages, LCA depth counts |
| `edge_degree_correlation_experiment` | Parent/child degree and arrival PMFs, neighbor-degree curve, assortativity |
| `latent_ablation_experiment` | Ablated graph collections, original reinforcement diagnostic, repeated feature samples, variance components |
| `decoder_entropy_experiment` | Per-query entropy samples, mode frequencies/disagreement, probability variance |
| `latent_predictability_experiment` | Held-out predictions, scores, trivial baseline scores |
| `distribution_comparison_experiment` | CV classifier scores/coefficients, energy distance, scalar distances, leader TV, coverage/precision, duplicates/diversity |
| `scaling_experiment` | Per-training-run configurations, fidelity/cost metrics, reconstruction, graph bootstrap intervals |

## Statistical choices to review

Sequence likelihood starts at arrival 3 and uses pre-arrival degrees. The
six-node example in the plan scores `log(1/64)`. Higher likelihood is not itself
a success criterion: compare the complete distribution and diversity. Use
`normalize_per_arrival=True` when comparing different sizes.

Kernel recovery defaults to 81 alpha values from 0 through 2. Intervals inherit
the grid resolution; a boundary estimate means the optimum may be outside it.
The finite-size count recurrence updates all degree bins simultaneously from
the previous state, beginning with two degree-one nodes.

Leader ties always choose the smallest arrival ID. This policy also defines
leader targets for probes. Top-k incident edge share counts each edge once,
including edges between two hubs. Herfindahl and entropy instead use normalized
degree mass `degree/(2E)`.

Tree geometry includes every unordered distinct vertex pair. Ancestor-age
counts exclude self relationships. Branch sizes are padded with zeros to N;
branch balance is second-largest/largest branch size (zero for one branch).
Geometry takes quadratic work per graph for pair measurements. Distribution
distances also materialize pairwise sample distances; use smaller collections
for exploratory runs on very large datasets.

The residual lag-one correlation pools same-node adjacent-time opportunities.
It is descriptive, not a significance test. Joint-event ratios below the minimum
count remain in returned data but are omitted from the inference plot.

## Latent experiments

```python
ablation = ev.latent_ablation_experiment(
    generators["vae"], seeds=range(100), samples_per_latent=50,
    baseline=generators["independent"],
)
entropy = ev.decoder_entropy_experiment(generators["vae"], seeds=range(100))
probes = ev.latent_predictability_experiment(run)

# The returned ablation collections can be evaluated independently.
ev.ba_sequence_likelihood_experiment({
    "NetworkX": collections["NetworkX"], **ablation["collections"]
})
```

`SeededPAGenerator.diagnostic_graph(latent, query_seed, argmax=False)` separates
the latent from query randomness. Normal public generation remains reproducible.
Pass the empirical independent generator explicitly to include that baseline.
Averaged decoding uses the supplied latent sample to estimate marginal decoder
probabilities. Whole-vector shuffling preserves latent sharing inside each
graph; it is a seed reassignment control, not a removal of dependence.

Variance decomposition uses sample variance within each latent group, and
subtracts `within / samples_per_latent` from the variance of group means to
estimate between-latent variance. Negative corrected estimates are retained as
evidence of finite-sample noise. Mode disagreement is the probability that two
draws from the empirical latent population have different argmax parents.

Probes use fresh prior latent/generated-property pairs, not encoded training
graphs. A fixed 70/30 seed split holds out probe examples. Linear regression
predicts scalar properties and logistic regression predicts leader identity;
the latter falls back to a majority classifier if training has only one class.
These are predictive diagnostics, not causal evidence.

## Distribution tests and scaling

The declared feature vector is root degree, maximum degree, leader arrival ID,
leaf fraction, diameter, and largest root-branch size. Logistic coefficients
describe this particular feature representation; leader ID is a numeric arrival
age here. The classifier has balanced source classes, stratified cross-validation,
and scaling fitted only on each training fold. Chance accuracy is 50%; a weak
classifier's failure to distinguish samples does not prove equivalence.

Distances use reference-standardized features. The reported squared energy
distance is the empirical V-statistic `2 mean cross - mean within_ref - mean
within_candidate`. Coverage is the fraction of reference points with a generated
point inside their nearest-reference-neighbor radius. Precision is the fraction
of generated points inside any such reference neighborhood. These are explicit
descriptive feature-space measures, not universal fidelity scores. Duplicate
rate compares exact arrival-labeled parent arrays, not unlabeled isomorphism.

```python
from dataclasses import replace

configs = [replace(PROFILES["demo"], num_nodes=n, latent_dim=d)
           for n in (16, 32, 64) for d in (8, 32)]
scaling = ev.scaling_experiment(configs, training_seeds=[11, 22, 33])
```

Scaling actually trains every requested model. It does not write checkpoints;
returned records contain metrics rather than models. Run it intentionally after
choosing budgets. Graph bootstrap intervals and variation across training runs
are separate: plotted error bars are standard deviations across training runs,
not graph-bootstrap confidence intervals. Keep other hyperparameters fixed when
using the default N/D plots. Query timings include latent derivation on each
call but are process-warm timings, not cold-start import/model-loading costs.
