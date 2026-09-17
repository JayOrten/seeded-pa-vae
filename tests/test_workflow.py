from svae import (
    Config,
    generators_from_run,
    graph_samples_experiment,
    graph_statistics_experiment,
    reconstruction_experiment,
    train,
)


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
