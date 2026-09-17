"""Query-conditioned VAE definition and training workflow."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .simulate import make_parent_dataset, parents_to_graph


@dataclass(frozen=True)
class DataSplits:
    """Parent-array tensors used for one training and evaluation run."""

    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


@dataclass(frozen=True)
class TrainingRun:
    """State needed by downstream generation and evaluation."""

    config: Config
    model: QueryVAE
    data: DataSplits
    history: tuple[dict[str, float], ...]
    best_validation_objective: float
    elapsed_seconds: float
    device: torch.device
    checkpoint_path: Path | None


class QueryVAE(nn.Module):
    """Encode complete parent arrays and decode parents for arbitrary queries."""

    def __init__(self, num_nodes: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear((num_nodes - 2) * num_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )

    def encode(self, parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = functional.one_hot(
            parents[:, 2:] - 1, num_classes=self.num_nodes
        ).float().flatten(1)
        hidden = self.encoder(encoded)
        return self.mean_head(hidden), self.log_variance_head(hidden).clamp(-10, 10)

    def decode(self, latent: torch.Tensor, queries: torch.Tensor):
        query_encoding = functional.one_hot(
            queries - 1, num_classes=self.num_nodes
        ).float().unsqueeze(0).expand(latent.shape[0], -1, -1)
        repeated_latent = latent[:, None, :].expand(-1, queries.numel(), -1)
        raw_logits = self.decoder(torch.cat([repeated_latent, query_encoding], -1))
        valid_parent_mask = (
            torch.arange(1, self.num_nodes + 1, device=raw_logits.device)[None, None, :]
            < queries[None, :, None]
        )
        return raw_logits, raw_logits.masked_fill(~valid_parent_mask, float("-inf")), valid_parent_mask

    def forward(self, parents: torch.Tensor, epsilon: torch.Tensor | None = None):
        latent_mean, latent_log_variance = self.encode(parents)
        latent_std = torch.exp(0.5 * latent_log_variance)
        noise = torch.randn_like(latent_std) if epsilon is None else epsilon
        latent = latent_mean + latent_std * noise
        decoded = self.decode(
            latent, torch.arange(3, self.num_nodes + 1, device=parents.device)
        )
        return *decoded, latent_mean, latent_log_variance, latent


def loss_parts(model: QueryVAE, parents: torch.Tensor, epsilon: torch.Tensor | None = None):
    """Return per-graph reconstruction/KL losses and diagnostic tensors."""
    outputs = model(parents, epsilon)
    masked_logits = outputs[1]
    latent_mean, latent_log_variance = outputs[3:5]
    targets = parents[:, 2:] - 1
    batch_size, num_queries = targets.shape
    cross_entropy = functional.cross_entropy(
        masked_logits.reshape(batch_size * num_queries, model.num_nodes),
        targets.reshape(-1), reduction="none",
    ).reshape(batch_size, num_queries)
    reconstruction = cross_entropy.sum(1)
    kl_divergence = 0.5 * (
        latent_mean.square() + latent_log_variance.exp() - 1 - latent_log_variance
    ).sum(1)
    return reconstruction, kl_divergence, cross_entropy, outputs


def make_data_splits(config: Config) -> DataSplits:
    """Create non-overlapping deterministic simulation splits."""
    return DataSplits(
        train=make_parent_dataset(config.num_nodes, config.train_graphs, first_seed=0),
        validation=make_parent_dataset(config.num_nodes, config.val_graphs, first_seed=1_000_000),
        test=make_parent_dataset(config.num_nodes, config.test_graphs, first_seed=2_000_000),
    )


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Use an explicitly requested device, otherwise prefer CUDA when available."""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def train(
    config: Config,
    *,
    data: DataSplits | None = None,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path | None = "artifacts",
    report: Callable[[dict[str, float]], None] | None = print,
    show: bool = True,
) -> TrainingRun:
    """Train one model and return all state needed by notebook experiments."""
    torch.set_num_threads(config.threads)
    random.seed(config.init_seed)
    np.random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    selected_device = resolve_device(device)
    data = data or make_data_splits(config)
    if show:
        print(
            {
                "train": tuple(data.train.shape),
                "validation": tuple(data.validation.shape),
                "test": tuple(data.test.shape),
            }
        )
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        for sample_index, axis in enumerate(axes):
            graph = parents_to_graph(data.train[sample_index].numpy())
            nx.draw(
                graph,
                nx.spring_layout(graph, seed=17),
                ax=axis,
                node_size=140,
                with_labels=True,
                font_size=7,
            )
            axis.set_title(f"train sample {sample_index}")
        plt.tight_layout()
        plt.show()
    loader_options = {"num_workers": 0, "pin_memory": selected_device.type == "cuda"}
    train_loader = DataLoader(
        TensorDataset(data.train), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.init_seed), **loader_options,
    )
    validation_loader = DataLoader(
        TensorDataset(data.validation), batch_size=config.batch_size, shuffle=False,
        **loader_options,
    )

    model = QueryVAE(config.num_nodes, config.latent_dim, config.hidden).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    validation_noise = torch.randn(
        config.val_graphs, config.latent_dim,
        generator=torch.Generator().manual_seed(314159),
    )
    history: list[dict[str, float]] = []
    best_state = None
    best_objective = float("inf")
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
            "train_recon_query": training_metrics["reconstruction"] / (config.num_nodes - 2),
            "train_kl_graph": training_metrics["kl"],
            "val_recon_query": validation_metrics["reconstruction"] / (config.num_nodes - 2),
            "val_kl_graph": validation_metrics["kl"],
        }
        history.append(row)
        objective = validation_metrics["reconstruction"] + validation_metrics["kl"]
        if epoch + 1 >= config.warmup_epochs and objective < best_objective:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
        if report is not None:
            report(row)

    if best_state is None:
        raise RuntimeError("training ended before a model could be selected")
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - started
    checkpoint_path = _save_checkpoint(
        model, config, history, best_objective, checkpoint_dir
    )
    if show:
        print(
            f"training seconds: {elapsed:.3f}; "
            f"selected validation objective/graph: {best_objective:.4f}"
        )
        if checkpoint_path is not None:
            print("saved checkpoint:", checkpoint_path.resolve())
        figure, axes = plt.subplots(1, 3, figsize=(11, 3))
        axes[0].plot(
            [record["train_recon_query"] for record in history], label="train"
        )
        axes[0].plot(
            [record["val_recon_query"] for record in history], label="validation"
        )
        axes[0].set_title("reconstruction/query")
        axes[0].legend()
        axes[1].plot([record["train_kl_graph"] for record in history], label="train")
        axes[1].plot(
            [record["val_kl_graph"] for record in history], label="validation"
        )
        axes[1].set_title("KL/graph")
        axes[2].plot([record["beta"] for record in history])
        axes[2].set_title("beta")
        plt.tight_layout()
        plt.show()
    return TrainingRun(
        config, model, data, tuple(history), best_objective, elapsed,
        selected_device, checkpoint_path,
    )


def _run_epoch(
    model: QueryVAE, loader: DataLoader, device: torch.device, beta: float,
    optimizer: torch.optim.Optimizer | None = None,
    validation_noise: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    reconstruction_total = kl_total = 0.0
    graphs_seen = noise_offset = 0
    with torch.set_grad_enabled(training):
        for (parents,) in loader:
            batch_size = len(parents)
            epsilon = None
            if validation_noise is not None:
                epsilon = validation_noise[noise_offset : noise_offset + batch_size].to(device)
                noise_offset += batch_size
            parents = parents.to(device, non_blocking=True)
            reconstruction, kl_divergence, _, _ = loss_parts(model, parents, epsilon)
            if training:
                optimizer.zero_grad()
                (reconstruction + beta * kl_divergence).mean().backward()
                optimizer.step()
            reconstruction_total += reconstruction.sum().item()
            kl_total += kl_divergence.sum().item()
            graphs_seen += batch_size
    return {"reconstruction": reconstruction_total / graphs_seen, "kl": kl_total / graphs_seen}


def _save_checkpoint(model, config, history, best_objective, checkpoint_dir):
    if checkpoint_dir is None:
        return None
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"seeded_pa_vae_{config.profile}_best.pt"
    torch.save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": asdict(config), "history": history,
        "best_validation_objective_per_graph": best_objective,
    }, path)
    return path
