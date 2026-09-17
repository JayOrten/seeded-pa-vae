"""Deterministic random-access generators built from trained models and baselines."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
import torch
from torch.nn import functional

from .simulate import parents_to_graph, validate_parents
from .train import QueryVAE, TrainingRun


@runtime_checkable
class ParentGenerator(Protocol):
    """Common interface consumed by evaluation experiments."""

    num_nodes: int

    def parent(self, seed: int, query: int) -> int | None: ...
    def parent_array(self, seed: int) -> np.ndarray: ...
    def graph(self, seed: int) -> nx.Graph: ...


def keyed_rng(seed: int, role: str, query: int | None = None) -> np.random.Generator:
    """Derive independent, stable random streams from a public integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    payload = json.dumps(
        ["pa-demo-v1", str(int(seed)), role, None if query is None else int(query)],
        separators=(",", ":"),
    ).encode()
    key = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(key))


class SeededPAGenerator:
    """Expose a frozen decoder through deterministic random-access parent queries."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        num_nodes: int,
        latent_dim: int,
    ):
        self.num_nodes = int(num_nodes)
        self.latent_dim = int(latent_dim)
        self.decoder = copy.deepcopy(decoder).cpu().float().eval()
        self.decoder.requires_grad_(False)

    @classmethod
    def from_model(cls, model: QueryVAE):
        """Create a deployable generator without retaining the encoder."""
        return cls(model.decoder, model.num_nodes, model.latent_dim)

    def latent(self, seed: int) -> torch.Tensor:
        """Return the shared latent deterministically selected by a public seed."""
        values = (
            keyed_rng(seed, "latent")
            .standard_normal(self.latent_dim)
            .astype(np.float32)
        )
        return torch.from_numpy(values)

    def probabilities(
        self, seed: int, query: int, *, latent: torch.Tensor | None = None
    ) -> np.ndarray:
        """Return decoder probabilities over the valid parents for one query."""
        self._validate_query(query)
        if query < 3:
            return np.ones(max(query - 1, 0), dtype=np.float64)
        latent = (
            self.latent(seed) if latent is None else latent.detach().cpu().float()
        )[None, :]
        query_encoding = functional.one_hot(
            torch.tensor([query - 1]), num_classes=self.num_nodes
        ).float()
        with torch.no_grad():
            logits = self.decoder(torch.cat([latent, query_encoding], 1))[
                0, : query - 1
            ]
        return torch.softmax(logits, 0).double().numpy()

    def diagnostic_graph(
        self, latent: torch.Tensor, query_seed: int, *, argmax: bool = False
    ) -> nx.Graph:
        """Generate with explicit latent and independent query-noise controls.

        This supports fixed-latent replication and zero-latent ablations without
        changing the public seed-to-graph convention. Argmax ties use older IDs.
        """
        parents = [-1, 1]
        for query in range(3, self.num_nodes + 1):
            probabilities = self.probabilities(query_seed, query, latent=latent)
            cumulative = np.cumsum(probabilities)
            cumulative[-1] = 1
            parent = (
                probabilities.argmax()
                if argmax
                else np.searchsorted(
                    cumulative,
                    keyed_rng(query_seed, "parent", query).random(),
                    side="right",
                )
            )
            parents.append(int(parent) + 1)
        return parents_to_graph(parents)

    def parent(self, seed: int, query: int) -> int | None:
        self._validate_query(query)
        query = int(query)
        if query == 1:
            return None
        if query == 2:
            return 1
        probabilities = self.probabilities(seed, query)
        cumulative = np.cumsum(probabilities, dtype=np.float64)
        cumulative[-1] = 1.0
        sample = keyed_rng(seed, "parent", query).random()
        return int(np.searchsorted(cumulative, sample, side="right") + 1)

    def parents(self, seed: int, queries: Iterable[int]) -> list[int | None]:
        return [self.parent(seed, query) for query in queries]

    def parent_array(self, seed: int) -> np.ndarray:
        parents = np.asarray(
            [-1, *(self.parent(seed, query) for query in range(2, self.num_nodes + 1))],
            dtype=np.int64,
        )
        return validate_parents(parents)

    def graph(self, seed: int) -> nx.Graph:
        return parents_to_graph(self.parent_array(seed))

    def _validate_query(self, query: int) -> None:
        if (
            isinstance(query, bool)
            or not isinstance(query, (int, np.integer))
            or not 1 <= int(query) <= self.num_nodes
        ):
            raise ValueError(f"query must be an integer in 1..{self.num_nodes}")


class IndependentParentGenerator:
    """Baseline that preserves parent marginals but removes cross-query dependence."""

    def __init__(
        self, training_parents: torch.Tensor | np.ndarray, *, smoothing: float = 0.5
    ):
        training = np.asarray(training_parents)
        if training.ndim != 2 or training.shape[1] < 2:
            raise ValueError("training_parents must have shape (graphs, nodes)")
        if smoothing < 0:
            raise ValueError("smoothing must be nonnegative")
        self.num_nodes = training.shape[1]
        self.probabilities: dict[int, np.ndarray] = {}
        for query in range(3, self.num_nodes + 1):
            counts = np.full(query - 1, smoothing, dtype=np.float64)
            np.add.at(counts, training[:, query - 1].astype(int) - 1, 1)
            self.probabilities[query] = counts / counts.sum()

    def parent(self, seed: int, query: int) -> int | None:
        if isinstance(query, bool) or not isinstance(query, (int, np.integer)):
            raise ValueError(f"query must be an integer in 1..{self.num_nodes}")
        query = int(query)
        if not 1 <= query <= self.num_nodes:
            raise ValueError(f"query must be an integer in 1..{self.num_nodes}")
        if query == 1:
            return None
        if query == 2:
            return 1
        cumulative = np.cumsum(self.probabilities[query], dtype=np.float64)
        cumulative[-1] = 1.0
        sample = keyed_rng(seed, "parent", query).random()
        return int(np.searchsorted(cumulative, sample, side="right") + 1)

    def parent_array(self, seed: int) -> np.ndarray:
        return validate_parents(
            np.asarray(
                [
                    -1,
                    *(
                        self.parent(seed, query)
                        for query in range(2, self.num_nodes + 1)
                    ),
                ],
                dtype=np.int64,
            )
        )

    def graph(self, seed: int) -> nx.Graph:
        return parents_to_graph(self.parent_array(seed))


def generators_from_run(run: TrainingRun) -> dict[str, ParentGenerator]:
    """Build the standard learned and baseline generators for an experiment run."""
    return {
        "vae": SeededPAGenerator.from_model(run.model),
        "independent": IndependentParentGenerator(run.data.train),
    }
