# Research question and feasibility

## Target interface

The desired external behavior is

$$
f_\theta(\text{seed},\text{query},\text{controls}) \rightarrow \text{value}.
$$

Queries may arrive in any order and do not receive earlier outputs. Answers for one seed should
nevertheless describe one coherent sampled object. For this first experiment, the object is an
arrival-labeled preferential-attachment tree and the value is the queried node's parent.

This is possible in a meaningful sense. Independently evaluating decoder calls does not imply that
their outputs are statistically independent. Different calls can recover shared decisions from the
same seed-derived latent. The open empirical question is how much dependency a bounded
representation and bounded query computation can preserve.

## Important corrections to the exploratory discussion

- A proposed entropy lower bound equating seed entropy with total correlation is not valid in
  general. One random bit copied to N positions has one bit of seed entropy and N-1 bits of total
  correlation. A shared decision can coordinate many outputs.
- Adding an index to a model does not make an arbitrary sequential process exchangeable, and de
  Finetti's theorem does not promise a compact or efficiently queried latent representation.
- Consistency can be learned through a loss when outputs have a shared coordination route. No
  argument in the conversation proves novel-length or graph-scale consistency impossible.
- A finite seed limits the number of reproducible outputs, but does not by itself imply dataset
  memorization or a bytes-per-object storage bound.
- A hierarchy or shared latent creates an opportunity for agreement, not a guarantee. Learned child
  states can still violate parent commitments.

The real constraint is a rate–distortion–computation tradeoff: some object distributions have compact,
locally accessible descriptions; others require growing information, growing computation, or an
approximation that loses dependencies. That tradeoff must be measured rather than assumed.

## Why preferential attachment

Preferential attachment is a useful controlled target because arrival labels are meaningful, the
sequential simulator is simple, and reinforcement produces measurable dependencies across time.
It is not evidence of novelty: exact sublinear random-access algorithms and shared-randomness
representations already exist for relevant PA variants.

Version one fixes `N` and `m=1`. Nodes 1 and 2 begin with one edge. Every node `t >= 3` chooses one
older parent proportional to current undirected degree. The result is a tree whose parent array is
easy to validate exactly.

## Longer-term design space

```mermaid
flowchart TD
    I[seed + query interface] --> G[one global latent]
    I --> H[hierarchical latent path]
    I --> W[bounded query neighborhood]
    G --> T[fixed-size PA baseline]
    H --> L[O(log N) path expansion if states/splits are fixed cost]
    W --> B[local refinement with canonical keyed noise]
```

The global latent is the simplest falsifiable baseline. A hierarchy could recursively carry shared
commitments; for example, an exact-count binary sequence can pass a count down a tree by jointly
splitting the remaining count. Variable-length language adds a routing problem because phrase lengths
must be committed consistently before token positions have stable paths. Bounded local refinement is
another option, but the public interface alone does not make its hidden computation constant or cheap.

## Claims this project should and should not make

The baseline can test whether a learned parent-query generator preserves graph-specific reinforcement
under explicit capacity and query-cost budgets. It cannot claim that VAEs on BA graphs, conditionally
independent graph decoding, distributions over neural functions, or random access to PA graphs are new.

