import importlib

import torch

from dla.config import Config
from dla.train import DLAQueryVAE, loss_parts, train

train_module = importlib.import_module("dla.train")


def test_model_shapes_masking_and_gradients():
    model = DLAQueryVAE(num_nodes=6, latent_dim=3, hidden_dim=8)
    parents = torch.tensor([
        [-1, 1, 1, 2, 2, 4],
        [-1, 1, 2, 2, 1, 3],
    ])
    directions = torch.tensor([
        [-1, 0, 1, 2, 3, 0],
        [-1, 2, 2, 0, 3, 1],
    ])
    reconstruction, kl_divergence, parent_ce, direction_ce, outputs = loss_parts(
        model, parents, directions
    )
    raw_logits, masked_logits, valid_mask, direction_logits, latent_mean, _, latent = outputs

    # Queries are nodes 2..6; the parent loss skips node 2.
    assert raw_logits.shape == (2, 5, 6)
    assert valid_mask.shape == (1, 5, 6)
    assert direction_logits.shape == (2, 5, 4)
    assert latent_mean.shape == latent.shape == (2, 3)
    assert parent_ce.shape == (2, 4)
    assert direction_ce.shape == (2, 5)
    assert torch.isfinite(masked_logits[valid_mask.expand_as(masked_logits)]).all()
    assert valid_mask[0, 0].tolist() == [True, False, False, False, False, False]
    assert torch.allclose(reconstruction, parent_ce.sum(1) + direction_ce.sum(1))

    (reconstruction.mean() + kl_divergence.mean()).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _config(**overrides):
    settings = dict(
        profile="test", num_nodes=8, latent_dim=3, hidden=12,
        train_graphs=16, val_graphs=8, test_graphs=8, generated_graphs=4,
        batch_size=8, epochs=1, warmup_epochs=1, threads=1,
    )
    return Config(**{**settings, **overrides})


def test_training_runs_and_saves_a_checkpoint(tmp_path):
    run = train(_config(), device="cpu", checkpoint_dir=tmp_path, report=None, show=False)

    assert run.model.training is False
    assert run.checkpoint_path is not None and run.checkpoint_path.exists()
    assert run.checkpoint_path.name == "dla_vae_test_best.pt"
    assert {"train_parent_query", "val_direction_query"} <= set(run.history[0])


def test_training_stops_after_both_validation_losses_worsen(monkeypatch):
    validation_metrics = iter(
        (
            {"reconstruction": 10.0, "parent": 6.0, "direction": 4.0, "kl": 1.0},
            {"reconstruction": 11.0, "parent": 7.0, "direction": 4.0, "kl": 2.0},
            {"reconstruction": 12.0, "parent": 8.0, "direction": 4.0, "kl": 3.0},
        )
    )

    def fake_epoch(model, loader, device, beta, optimizer=None, validation_noise=None):
        if optimizer is not None:
            return {"reconstruction": 10.0, "parent": 6.0, "direction": 4.0, "kl": 1.0}
        return next(validation_metrics)

    monkeypatch.setattr(train_module, "_run_epoch", fake_epoch)
    run = train(
        _config(epochs=8, early_stopping_patience=2),
        device="cpu", checkpoint_dir=None, report=None, show=False,
    )

    assert run.stopped_early is True
    assert run.selected_epoch == 1
    assert len(run.history) == 3
    assert run.history[-1]["early_stopping_bad_epochs"] == 2
