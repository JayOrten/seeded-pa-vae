# Literature map and novelty boundary

The search summarized in the originating conversation found substantial overlap. Its conclusion was
provisional: no exact match was identified for learning from growth-graph samples while exposing a
stateless `(seed, arrival_id) -> parent_id` interface and explicitly measuring joint-distribution
fidelity against query cost. That is not proof that the combination is unpublished.

## Closest work

- [Guzmán et al., *Recovering Barabási–Albert Parameters of Graphs through Disentanglement*](https://arxiv.org/abs/2105.00997)
  trains latent models on BA graphs and reports difficulty with a one-shot VAE before using a
  sequential decoder. Its emphasis is generator-parameter recovery, not realization-specific
  stateless parent queries. Follow backward to Stoehr et al. on disentangling generative parameters.
- [Even et al., *Sublinear Random Access Generators for Preferential Attachment Graphs*](https://arxiv.org/abs/1602.06159)
  establishes algorithmic random access and is an essential baseline. Its state, commitment, and
  initialization conventions must be checked before direct comparison.
- Garavaglia, Hazra, van der Hofstad, and Ray, *Universality of the local limit of preferential
  attachment models*, develops Pólya-urn representations where attachments become conditionally
  independent given shared variables. Those variables grow with the graph; this does not imply a
  fixed 64-dimensional latent is sufficient.
- [Simonovsky and Komodakis, *GraphVAE*](https://arxiv.org/abs/1802.03480) establishes graph-level
  latents with conditionally independent output variables for bounded-size graphs.
- [Dupont, Teh, and Doucet, *Generative Models as Distributions of Functions*](https://arxiv.org/abs/2102.04776)
  is the closest general precedent for sampling a latent-conditioned function and querying arbitrary
  coordinates. Follow with *From data to functa*.
- Graphon autoencoders and implicit graphon representations are close neighbors for claims about
  locally queryable neural graph generators. Relevant examples include Xu et al. (2021), Xia,
  Mishne, and Wang (2023), and Azizpour, Zilberstein, and Segarra (2025). Their exchangeable/node
  alignment framing differs from arrival-indexed, exactly-one-older-parent growth.
- GraphRNN is a useful learned sequential comparison but does not satisfy the target inference
  interface.

## Foundation reading order

1. [Doersch, *Tutorial on Variational Autoencoders*](https://arxiv.org/abs/1606.05908).
2. [Kingma and Welling, *Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114).
3. Barabási, *Network Science*, Chapter 5.
4. Krapivsky and Redner, *Organization of Growing Random Networks*.
5. [O'Bray et al., *Evaluation Metrics for Graph Generative Models*](https://arxiv.org/abs/2106.01098).

Then read Guzmán et al., GraphVAE, distributions of functions, Even et al., and the Pólya-urn
construction before settling a contribution claim.

## Current novelty assessment

| Claim | Assessment |
|---|---|
| Train a VAE on simulated BA graphs | Established |
| Coordinate non-autoregressive graph outputs with a shared latent | Established |
| Represent samples as queryable neural functions | Established |
| Random access to preferential-attachment graphs | Established algorithmically |
| Preserve realization-specific PA dependence with a learned stateless parent-query generator under measured capacity and cost | Plausible research target; novelty unconfirmed |

The contribution must therefore rest on demonstrated dependence fidelity, capacity/cost scaling, and
architectural advantage over both simple latent and independent-parent baselines plus relevant exact
constructions—not on the interface or VAE alone.

