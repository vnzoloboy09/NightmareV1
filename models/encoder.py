import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
