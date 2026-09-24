# Baseline architecture and inference contract

## Training model

Each graph is an integer parent array. The encoder sees all learned parent choices and returns the
parameters of a diagonal Gaussian:

$$
(\mu, \log \sigma^2)=E_\phi(P), \qquad
z=\mu+\sigma\odot\epsilon, \quad \epsilon\sim\mathcal N(0,I).
$$

One sampled `z` is reused for every arrival query in that graph. The decoder receives only `z` and a
one-hot arrival ID and emits parent logits. Invalid parents—current or future nodes—are masked.
There is no attention or communication across the query dimension.

```mermaid
flowchart LR
    P[complete training parent array] --> E[MLP encoder]
    E --> M[mu, log variance]
    X[epsilon ~ N(0,I)] --> Z[reparameterized z]
    M --> Z
    Z --> D[shared MLP decoder]
    Q[arrival ID] --> D
    D --> O[masked older-parent logits]
```

For `Q=N-2` learned arrivals, the loss per graph is

$$
\mathcal L = \sum_{t=3}^{N}-\log p_\theta(P_t\mid z,t)
+\beta\,\frac12\sum_d(\mu_d^2+e^{\log\sigma_d^2}-1-\log\sigma_d^2).
$$

KL is charged once per graph, not once per query. With beta one this is a Monte Carlo estimate of the
negative ELBO for the categorical decoder, not exact marginal negative log likelihood.

## Generation model

Fresh generation bypasses the encoder. A canonical JSON record is hashed with SHA-256 and converted
to a PCG64 seed. Role `latent` gives one standard-normal vector shared by all queries for a graph.
Role `parent` plus the arrival ID gives a uniform variate used for inverse-CDF sampling.

```python
generator.parent(seed=123, query=50)
```

For a fixed checkpoint and supported environment, this call requires no input graph, history,
caller-supplied latent, encoder, or cache. Query order and intervening seeds do not affect its answer.
Node 1 returns `None`; node 2 returns parent 1. The wrapper deliberately uses identical scalar CPU
float32 decoder calls to avoid batch-shape-dependent floating-point differences.

This reproducibility promise is not cross-version bit identity. Retraining, changing configuration,
PyTorch versions, or hardware can change results.

## What coherence means here

The latent is a coordination channel, but categorical sampling also uses query-specific noise. If the
decoder leaves incompatible alternatives unresolved, independent query draws can assemble an
implausible graph. The experiment therefore evaluates whole sampled collections, not only decoder
accuracy. A later variant could use deterministic argmax so all diversity resides in `z`, but that is
a different generative model.

## Model selection and early stopping

Validation uses fixed latent noise so changes between epochs reflect model changes rather than a new
Monte Carlo draw. Checkpoint selection begins after KL warmup and minimizes validation reconstruction
plus KL loss. Training stops when both validation components are worse than the selected checkpoint
for `early_stopping_patience` consecutive epochs (10 by default). An epoch where either component is
within `early_stopping_min_delta` of the selected value resets the counter. The selected checkpoint is
restored before evaluation, and the run records its epoch and whether training stopped early.

## Computational accounting

The public method recomputes a small latent and evaluates a dense `N`-class head. It avoids replaying
prior attachments, but its work is not constant with graph size because the output head grows with
`N`. Full-graph generation calls the scalar method for every node. Any future claim about efficiency
must include latent construction, candidate scoring, hierarchy expansion, local refinement, and cache
warmup.
