import importlib

from svae import (
    Config,
    generators_from_run,
    graph_samples_experiment,
    graph_statistics_experiment,
    reconstruction_experiment,
    train,
)

train_module = importlib.import_module("svae.train")


def test_training_and_evaluation_workflow(tmp_path):
    config = Config(
        profile="test", num_nodes=8, latent_dim=3, hidden=12,
        train_graphs=16, val_graphs=8, test_graphs=8, generated_graphs=4,
        batch_size=8, epochs=1, warmup_epochs=1, threads=1,
    )
    run = train(
        config,
        device="cpu",
        checkpoint_dir=tmp_path,
        report=None,
        show=False,
    )
    generators = generators_from_run(run)
    reconstruction = reconstruction_experiment(run, show=False)
    collections = graph_samples_experiment(
        run,
        generators["vae"],
        generators["independent"],
        seeds=range(4),
        show=False,
    )
    statistics = graph_statistics_experiment(
        collections, num_nodes=config.num_nodes, bootstrap_samples=5, show=False
    )

    assert run.model.training is False
    assert run.checkpoint_path is not None and run.checkpoint_path.exists()
    assert set(generators) == {"vae", "independent"}
    assert "mean_latent_test_parent_accuracy" in reconstruction
    assert "reinforcement" in statistics


def test_training_stops_after_both_validation_losses_worsen(monkeypatch):
    config = Config(
        profile="early-stop-test", num_nodes=8, latent_dim=3, hidden=12,
        train_graphs=16, val_graphs=8, test_graphs=8, generated_graphs=4,
        batch_size=8, epochs=8, warmup_epochs=1, early_stopping_patience=2,
        threads=1,
    )
    validation_metrics = iter(
        (
            {"reconstruction": 10.0, "kl": 1.0},
            {"reconstruction": 11.0, "kl": 2.0},
            {"reconstruction": 12.0, "kl": 3.0},
        )
    )

    def fake_epoch(model, loader, device, beta, optimizer=None, validation_noise=None):
        if optimizer is not None:
            return {"reconstruction": 10.0, "kl": 1.0}
        return next(validation_metrics)

    monkeypatch.setattr(train_module, "_run_epoch", fake_epoch)
    run = train(
        config, device="cpu", checkpoint_dir=None, report=None, show=False
    )

    assert run.stopped_early is True
    assert run.selected_epoch == 1
    assert len(run.history) == 3
    assert run.history[-1]["early_stopping_bad_epochs"] == 2
