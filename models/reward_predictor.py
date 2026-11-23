import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    def __init__(
        self,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=128,
    ):
        super().__init__()

        input_dim = deter_dim + stoch_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, deter, stoch):
        latent = torch.cat([deter, stoch], dim=-1)

        return self.network(latent)