"""Experiment configuration profiles."""

from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass(frozen=True)
class Config:
    """Settings shared by simulation, training, and evaluation."""

    profile: str
    num_nodes: int
    latent_dim: int
    hidden: int
    train_graphs: int
    val_graphs: int
    test_graphs: int
    generated_graphs: int
    batch_size: int
    epochs: int
    lr: float = 1e-3
    beta: float = 1.0
    warmup_epochs: int = 10
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    init_seed: int = 20260909
    threads: int = 8


PROFILES = {
    "smoke": Config(
        profile="smoke",
        num_nodes=16,
        latent_dim=8,
        hidden=64,
        train_graphs=128,
        val_graphs=32,
        test_graphs=64,
        generated_graphs=64,
        batch_size=32,
        epochs=2,
        warmup_epochs=1,
        threads=2,
    ),
    "demo": Config(
        profile="demo",
        num_nodes=64,
        latent_dim=64,
        hidden=256,
        train_graphs=2_048,
        val_graphs=256,
        test_graphs=512,
        generated_graphs=512,
        batch_size=64,
        epochs=30,
    ),
    # 16x the demo training data, a wider model, and over 3x the epochs.
    "large": Config(
        profile="large",
        num_nodes=256,
        latent_dim=128,
        hidden=512,
        train_graphs=65_536,
        val_graphs=4_096,
        test_graphs=4_096,
        generated_graphs=2_048,
        batch_size=256,
        epochs=100,
        warmup_epochs=15,
        threads=8,
    ),
}
