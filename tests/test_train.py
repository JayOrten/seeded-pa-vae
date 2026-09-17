import torch

from svae.train import QueryVAE, loss_parts


def test_model_shapes_masking_and_gradients():
    model = QueryVAE(num_nodes=6, latent_dim=3, hidden_dim=8)
    parents = torch.tensor([
        [-1, 1, 1, 2, 2, 4],
        [-1, 1, 2, 2, 1, 3],
    ])
    reconstruction, kl_divergence, cross_entropy, outputs = loss_parts(model, parents)
    raw_logits, masked_logits, valid_mask, latent_mean, _, latent = outputs

    assert raw_logits.shape == (2, 4, 6)
    assert valid_mask.shape == (1, 4, 6)
    assert latent_mean.shape == latent.shape == (2, 3)
    assert cross_entropy.shape == (2, 4)
    assert torch.isfinite(masked_logits[valid_mask.expand_as(masked_logits)]).all()
    assert not valid_mask[..., -1].any()

    (reconstruction.mean() + kl_divergence.mean()).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
