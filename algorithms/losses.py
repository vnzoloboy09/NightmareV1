import torch
from torch.distributions import Categorical, kl_divergence


def reconstruction_loss(reconstructed_obs, obs):
    return torch.nn.functional.mse_loss(reconstructed_obs, obs)


def reward_loss(predicted_rewards, rewards):
    predicted_rewards = predicted_rewards[:, 1:, 0]

    return torch.nn.functional.mse_loss(predicted_rewards, rewards)


def kl_loss(prior_logits, posterior_logits, balance=0.8, free_nats=1.0):
    prior_dist = Categorical(logits=prior_logits)
    posterior_dist = Categorical(logits=posterior_logits)

    dynamics_kl = kl_divergence(
        Categorical(logits=posterior_logits.detach()),
        prior_dist
    )

    representation_kl = kl_divergence(
        posterior_dist,
        Categorical(logits=prior_logits.detach())
    )

    kl = balance * dynamics_kl + (1.0 - balance) * representation_kl

    kl = torch.clamp(kl, min=free_nats)

    return kl.mean()


def world_model_loss(outputs, obs, rewards):
    recon_loss = reconstruction_loss(
        outputs["reconstructed_obs"],
        obs,
    )

    rew_loss = reward_loss(
        outputs["predicted_rewards"],
        rewards,
    )

    kl = kl_loss(
        outputs["prior_logits"],
        outputs["posterior_logits"],
    )

    total_loss = recon_loss + rew_loss + kl

    return {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "reward_loss": rew_loss,
        "kl_loss": kl,
    }