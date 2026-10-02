"""Query-conditioned VAE for DLA trees: a parent head and a direction head per query."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS_DIR, Config
from .simulate import make_dataset, rebuild_positions

NUM_DIRECTIONS = 4


@dataclass(frozen=True)
class DataSplits:
    """``(parents, directions)`` tensor pairs used for one training and evaluation run."""

    train: tuple[torch.Tensor, torch.Tensor]
    validation: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class TrainingRun:
    """State needed by downstream generation and evaluation."""

    config: Config
    model: DLAQueryVAE
    data: DataSplits
    history: tuple[dict[str, float], ...]
    best_validation_objective: float
    selected_epoch: int
    stopped_early: bool
    elapsed_seconds: float
    device: torch.device
    checkpoint_path: Path | None


class QueryDecoder(nn.Module):
    """Shared trunk over ``[latent, query one-hot]`` with separate parent and direction heads.

    The two heads are independent given the latent and the query.
    """

    def __init__(self, num_nodes: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + num_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.parent_head = nn.Linear(hidden_dim, num_nodes)
        self.direction_head = nn.Linear(hidden_dim, NUM_DIRECTIONS)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        return self.parent_head(hidden), self.direction_head(hidden)


class DLAQueryVAE(nn.Module):
    """Encode complete DLA trees and decode (parent, direction) for arbitrary queries."""

    def __init__(self, num_nodes: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        encoder_inputs = (num_nodes - 2) * num_nodes + (num_nodes - 1) * NUM_DIRECTIONS
        self.encoder = nn.Sequential(
            nn.Linear(encoder_inputs, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = QueryDecoder(num_nodes, latent_dim, hidden_dim)

    def encode(
        self, parents: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Node 2's parent is always node 1, so parents start at node 3. Directions start at node 2.
        encoded = torch.cat([
            functional.one_hot(parents[:, 2:] - 1, num_classes=self.num_nodes).float().flatten(1),
            functional.one_hot(directions[:, 1:], num_classes=NUM_DIRECTIONS).float().flatten(1),
        ], 1)
        hidden = self.encoder(encoded)
        return self.mean_head(hidden), self.log_variance_head(hidden).clamp(-10, 10)

    def decode(self, latent: torch.Tensor, queries: torch.Tensor):
        """Return raw parent logits, masked parent logits, the valid-parent mask, and direction logits."""
        query_encoding = functional.one_hot(
            queries - 1, num_classes=self.num_nodes
        ).float().unsqueeze(0).expand(latent.shape[0], -1, -1)
        repeated_latent = latent[:, None, :].expand(-1, queries.numel(), -1)
        raw_parent_logits, direction_logits = self.decoder(
            torch.cat([repeated_latent, query_encoding], -1)
        )
        valid_parent_mask = (
            torch.arange(1, self.num_nodes + 1, device=raw_parent_logits.device)[None, None, :]
            < queries[None, :, None]
        )
        masked_parent_logits = raw_parent_logits.masked_fill(~valid_parent_mask, float("-inf"))
        return raw_parent_logits, masked_parent_logits, valid_parent_mask, direction_logits

    def forward(
        self,
        parents: torch.Tensor,
        directions: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ):
        latent_mean, latent_log_variance = self.encode(parents, directions)
        latent_std = torch.exp(0.5 * latent_log_variance)
        noise = torch.randn_like(latent_std) if epsilon is None else epsilon
        latent = latent_mean + latent_std * noise
        # Decode nodes 2..N. Node 2's parent is fixed, so the parent loss skips it.
        decoded = self.decode(
            latent, torch.arange(2, self.num_nodes + 1, device=parents.device)
        )
        return *decoded, latent_mean, latent_log_variance, latent


def loss_parts(
    model: DLAQueryVAE,
    parents: torch.Tensor,
    directions: torch.Tensor,
    epsilon: torch.Tensor | None = None,
):
    """Return per-graph reconstruction/KL losses, per-query cross-entropies, and model outputs.

    Reconstruction is the parent cross-entropy over nodes 3..N plus the direction
    cross-entropy over nodes 2..N.
    """
    outputs = model(parents, directions, epsilon)
    masked_parent_logits, direction_logits = outputs[1], outputs[3]
    latent_mean, latent_log_variance = outputs[4:6]

    parent_targets = parents[:, 2:] - 1
    batch_size, num_parent_queries = parent_targets.shape
    parent_cross_entropy = functional.cross_entropy(
        masked_parent_logits[:, 1:].reshape(-1, model.num_nodes),
        parent_targets.reshape(-1), reduction="none",
    ).reshape(batch_size, num_parent_queries)

    direction_targets = directions[:, 1:]
    direction_cross_entropy = functional.cross_entropy(
        direction_logits.reshape(-1, NUM_DIRECTIONS),
        direction_targets.reshape(-1), reduction="none",
    ).reshape(batch_size, -1)

    reconstruction = parent_cross_entropy.sum(1) + direction_cross_entropy.sum(1)
    kl_divergence = 0.5 * (
        latent_mean.square() + latent_log_variance.exp() - 1 - latent_log_variance
    ).sum(1)
    return reconstruction, kl_divergence, parent_cross_entropy, direction_cross_entropy, outputs


def make_data_splits(config: Config) -> DataSplits:
    """Create non-overlapping deterministic simulation splits."""
    return DataSplits(
        train=make_dataset(config.num_nodes, config.train_graphs, first_seed=0),
        validation=make_dataset(config.num_nodes, config.val_graphs, first_seed=1_000_000),
        test=make_dataset(config.num_nodes, config.test_graphs, first_seed=2_000_000),
    )


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Use an explicitly requested device, otherwise prefer CUDA when available."""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def train(
    config: Config,
    *,
    data: DataSplits | None = None,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path | None = ARTIFACTS_DIR,
    report: Callable[[dict[str, float]], None] | None = print,
    show: bool = True,
) -> TrainingRun:
    """Train one model and return all state needed by notebook experiments."""
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    torch.set_num_threads(config.threads)
    random.seed(config.init_seed)
    np.random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    selected_device = resolve_device(device)
    data = data or make_data_splits(config)
    if show:
        print(
            {
                "train": tuple(data.train[0].shape),
                "validation": tuple(data.validation[0].shape),
                "test": tuple(data.test[0].shape),
            }
        )
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        for sample_index, axis in enumerate(axes):
            positions, _ = rebuild_positions(
                data.train[0][sample_index].numpy(), data.train[1][sample_index].numpy()
            )
            axis.scatter(
                positions[:, 0], positions[:, 1], c=np.arange(config.num_nodes),
                cmap="viridis", s=12, marker="s",
            )
            axis.set_aspect("equal")
            axis.set_title(f"train sample {sample_index}")
        plt.tight_layout()
        plt.show()
    loader_options = {"num_workers": 0, "pin_memory": selected_device.type == "cuda"}
    train_loader = DataLoader(
        TensorDataset(*data.train), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.init_seed), **loader_options,
    )
    validation_loader = DataLoader(
        TensorDataset(*data.validation), batch_size=config.batch_size, shuffle=False,
        **loader_options,
    )

    model = DLAQueryVAE(config.num_nodes, config.latent_dim, config.hidden).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    validation_noise = torch.randn(
        config.val_graphs, config.latent_dim,
        generator=torch.Generator().manual_seed(314159),
    )
    history: list[dict[str, float]] = []
    best_state = None
    best_objective = float("inf")
    best_validation_parts: tuple[float, float] | None = None
    selected_epoch = 0
    worsening_epochs = 0
    stopped_early = False
    started = time.perf_counter()
    for epoch in range(config.epochs):
        beta = config.beta * min(1.0, (epoch + 1) / config.warmup_epochs)
        training_metrics = _run_epoch(model, train_loader, selected_device, beta, optimizer)
        validation_metrics = _run_epoch(
            model, validation_loader, selected_device, 1.0,
            validation_noise=validation_noise,
        )
        row = {
            "epoch": float(epoch + 1), "beta": beta,
            "train_parent_query": training_metrics["parent"] / (config.num_nodes - 2),
            "train_direction_query": training_metrics["direction"] / (config.num_nodes - 1),
            "train_kl_graph": training_metrics["kl"],
            "val_parent_query": validation_metrics["parent"] / (config.num_nodes - 2),
            "val_direction_query": validation_metrics["direction"] / (config.num_nodes - 1),
            "val_kl_graph": validation_metrics["kl"],
        }
        objective = validation_metrics["reconstruction"] + validation_metrics["kl"]
        if epoch + 1 >= config.warmup_epochs and objective < best_objective:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            best_validation_parts = (
                validation_metrics["reconstruction"], validation_metrics["kl"]
            )
            selected_epoch = epoch + 1
            worsening_epochs = 0
        elif epoch + 1 >= config.warmup_epochs:
            assert best_validation_parts is not None
            both_worsened = all(
                current > selected + config.early_stopping_min_delta
                for current, selected in zip(
                    (validation_metrics["reconstruction"], validation_metrics["kl"]),
                    best_validation_parts,
                )
            )
            worsening_epochs = worsening_epochs + 1 if both_worsened else 0
        row["early_stopping_bad_epochs"] = float(worsening_epochs)
        history.append(row)
        if report is not None:
            report(row)
        if worsening_epochs >= config.early_stopping_patience:
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("training ended before a model could be selected")
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - started
    checkpoint_path = _save_checkpoint(
        model, config, history, best_objective, selected_epoch, stopped_early,
        checkpoint_dir,
    )
    if show:
        print(
            f"training seconds: {elapsed:.3f}; "
            f"selected epoch: {selected_epoch}; "
            f"selected validation objective/graph: {best_objective:.4f}; "
            f"stopped early: {stopped_early}"
        )
        if checkpoint_path is not None:
            print("saved checkpoint:", checkpoint_path.resolve())
        figure, axes = plt.subplots(1, 4, figsize=(14, 3))
        for axis, key, title in (
            (axes[0], "parent_query", "parent cross-entropy/query"),
            (axes[1], "direction_query", "direction cross-entropy/query"),
            (axes[2], "kl_graph", "KL/graph"),
        ):
            axis.plot([record[f"train_{key}"] for record in history], label="train")
            axis.plot([record[f"val_{key}"] for record in history], label="validation")
            axis.set_title(title)
            axis.legend()
        axes[3].plot([record["beta"] for record in history])
        axes[3].set_title("beta")
        plt.tight_layout()
        plt.show()
    return TrainingRun(
        config, model, data, tuple(history), best_objective, selected_epoch,
        stopped_early, elapsed,
        selected_device, checkpoint_path,
    )


def _run_epoch(
    model: DLAQueryVAE, loader: DataLoader, device: torch.device, beta: float,
    optimizer: torch.optim.Optimizer | None = None,
    validation_noise: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    parent_total = direction_total = kl_total = 0.0
    graphs_seen = noise_offset = 0
    with torch.set_grad_enabled(training):
        for parents, directions in loader:
            batch_size = len(parents)
            epsilon = None
            if validation_noise is not None:
                epsilon = validation_noise[noise_offset : noise_offset + batch_size].to(device)
                noise_offset += batch_size
            parents = parents.to(device, non_blocking=True)
            directions = directions.to(device, non_blocking=True)
            reconstruction, kl_divergence, parent_ce, direction_ce, _ = loss_parts(
                model, parents, directions, epsilon
            )
            if training:
                optimizer.zero_grad()
                (reconstruction + beta * kl_divergence).mean().backward()
                optimizer.step()
            parent_total += parent_ce.sum().item()
            direction_total += direction_ce.sum().item()
            kl_total += kl_divergence.sum().item()
            graphs_seen += batch_size
    return {
        "reconstruction": (parent_total + direction_total) / graphs_seen,
        "parent": parent_total / graphs_seen,
        "direction": direction_total / graphs_seen,
        "kl": kl_total / graphs_seen,
    }


def _save_checkpoint(
    model, config, history, best_objective, selected_epoch, stopped_early,
    checkpoint_dir,
):
    if checkpoint_dir is None:
        return None
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"dla_vae_{config.profile}_best.pt"
    torch.save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": asdict(config), "history": history,
        "best_validation_objective_per_graph": best_objective,
        "selected_epoch": selected_epoch,
        "stopped_early": stopped_early,
    }, path)
    return path
