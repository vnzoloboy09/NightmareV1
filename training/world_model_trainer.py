import torch
import torch.nn as nn

from models.world_model import WorldModel
from algorithms.losses import world_model_loss


class WorldModelTrainer:
    def __init__(
        self,
        world_model: WorldModel,
        learning_rate=1e-4,
        max_grad_norm=100.0
    ):
        self.world_model = world_model
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=learning_rate,
        )

    def train_step(self, obs, actions, rewards):
        self.world_model.train()

        device = next(self.world_model.parameters()).device

        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        self.optimizer.zero_grad()

        output = self.world_model.observe(obs, actions)

        losses = world_model_loss(output, obs, actions)

        losses["total_loss"].backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()

        return {
            "total_loss": losses["total_loss"].item(),
            "reconstruction_loss": losses["reconstruction_loss"].item(),
            "reward_loss": losses["reward_loss"].item(),
            "kl_loss": losses["kl_loss"].item(),
            "grad_norm": float(grad_norm),
        }