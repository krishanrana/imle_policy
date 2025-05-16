

import torch
import wandb
import pdb

def rs_imle_loss(real_samples, fake_samples, epsilon=0.03):
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, 1, -1)
    fake_flat = fake_samples.reshape(B, n_samples, -1)

    distances = torch.cdist(real_flat, fake_flat).squeeze(1)

    valid_samples = (distances > epsilon).float()
    # wandb.log({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon})
    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)

    wandb_log = ({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon, "loss": loss.item()})
    return loss, wandb_log
