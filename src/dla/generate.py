"""Deterministic random-access DLA generators built from trained models and baselines."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.nn import functional

from .simulate import rebuild_positions, validate_dla
from .train import NUM_DIRECTIONS, DLAQueryVAE, TrainingRun

Answer = tuple[int, int] | tuple[None, None]


@runtime_checkable
class DLAGenerator(Protocol):
    """Common interface consumed by evaluation experiments."""

    num_nodes: int

    def query(self, seed: int, t: int) -> Answer: ...
    def arrays(self, seed: int) -> tuple[np.ndarray, np.ndarray]: ...
    def positions(self, seed: int) -> tuple[np.ndarray, int]: ...


def keyed_rng(seed: int, role: str, query: int | None = None) -> np.random.Generator:
    """Derive independent, stable random streams from a public integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    payload = json.dumps(
        ["dla-v1", str(int(seed)), role, None if query is None else int(query)],
        separators=(",", ":"),
    ).encode()
    key = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(key))


def _sample(probabilities: np.ndarray, seed: int, role: str, query: int) -> int:
    """Draw an index from ``probabilities`` using the keyed stream for (seed, role, query)."""
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, keyed_rng(seed, role, query).random(), side="right"))


class _ArraysFromQueries:
    """Shared whole-tree helpers for generators that implement ``query``."""

    num_nodes: int

    def query(self, seed: int, t: int) -> Answer:
        raise NotImplementedError

    def queries(self, seed: int, ts: Iterable[int]) -> list[Answer]:
        return [self.query(seed, t) for t in ts]

    def arrays(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Return validated ``(parents, directions)`` built from one query per node."""
        answers = self.queries(seed, range(2, self.num_nodes + 1))
        parents = np.array([-1, *(parent for parent, _ in answers)], dtype=np.int64)
        directions = np.array([-1, *(direction for _, direction in answers)], dtype=np.int64)
        return validate_dla(parents, directions)

    def positions(self, seed: int) -> tuple[np.ndarray, int]:
        """Return rebuilt positions and the number of nodes landing on an occupied square."""
        return rebuild_positions(*self.arrays(seed))

    def _validate_query(self, t: int) -> None:
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, np.integer))
            or not 1 <= int(t) <= self.num_nodes
        ):
            raise ValueError(f"query must be an integer in 1..{self.num_nodes}")


class SeededDLAGenerator(_ArraysFromQueries):
    """Expose a frozen decoder through deterministic random-access (parent, direction) queries."""

    def __init__(self, decoder: torch.nn.Module, num_nodes: int, latent_dim: int):
        self.num_nodes = int(num_nodes)
        self.latent_dim = int(latent_dim)
        self.decoder = copy.deepcopy(decoder).cpu().float().eval()
        self.decoder.requires_grad_(False)

    @classmethod
    def from_model(cls, model: DLAQueryVAE):
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
        self, seed: int, t: int, *, latent: torch.Tensor | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return decoder probabilities over parents 1..t-1 and over the 4 directions."""
        self._validate_query(t)
        if t == 1:
            return np.ones(0), np.ones(0)
        latent = (
            self.latent(seed) if latent is None else latent.detach().cpu().float()
        )[None, :]
        query_encoding = functional.one_hot(
            torch.tensor([t - 1]), num_classes=self.num_nodes
        ).float()
        with torch.no_grad():
            parent_logits, direction_logits = self.decoder(torch.cat([latent, query_encoding], 1))
        return (
            torch.softmax(parent_logits[0, : t - 1], 0).double().numpy(),
            torch.softmax(direction_logits[0], 0).double().numpy(),
        )

    def query(self, seed: int, t: int) -> Answer:
        parent_probabilities, direction_probabilities = self.probabilities(seed, t)  # validates t
        t = int(t)
        if t == 1:
            return None, None
        parent = 1 if t == 2 else _sample(parent_probabilities, seed, "parent", t) + 1
        return parent, _sample(direction_probabilities, seed, "direction", t)


class IndependentDLAGenerator(_ArraysFromQueries):
    """Baseline that keeps per-node parent and direction frequencies but no shared latent."""

    def __init__(
        self,
        training_parents: torch.Tensor | np.ndarray,
        training_directions: torch.Tensor | np.ndarray,
        *,
        smoothing: float = 0.5,
    ):
        parents = np.asarray(training_parents)
        directions = np.asarray(training_directions)
        if parents.ndim != 2 or parents.shape[1] < 2 or parents.shape != directions.shape:
            raise ValueError("training arrays must share a shape of (graphs, nodes)")
        if smoothing < 0:
            raise ValueError("smoothing must be nonnegative")
        self.num_nodes = parents.shape[1]
        self.parent_probabilities: dict[int, np.ndarray] = {}
        self.direction_probabilities: dict[int, np.ndarray] = {}
        for t in range(2, self.num_nodes + 1):
            parent_counts = np.full(t - 1, smoothing, dtype=np.float64)
            np.add.at(parent_counts, parents[:, t - 1].astype(int) - 1, 1)
            self.parent_probabilities[t] = parent_counts / parent_counts.sum()
            direction_counts = np.full(NUM_DIRECTIONS, smoothing, dtype=np.float64)
            np.add.at(direction_counts, directions[:, t - 1].astype(int), 1)
            self.direction_probabilities[t] = direction_counts / direction_counts.sum()

    def query(self, seed: int, t: int) -> Answer:
        self._validate_query(t)
        t = int(t)
        if t == 1:
            return None, None
        parent = 1 if t == 2 else _sample(self.parent_probabilities[t], seed, "parent", t) + 1
        return parent, _sample(self.direction_probabilities[t], seed, "direction", t)


def generators_from_run(run: TrainingRun) -> dict[str, DLAGenerator]:
    """Build the standard learned and baseline generators for an experiment run."""
    return {
        "vae": SeededDLAGenerator.from_model(run.model),
        "independent": IndependentDLAGenerator(*run.data.train),
    }
