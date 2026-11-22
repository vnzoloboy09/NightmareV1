import torch
import torch.nn as nn


class DeterministicDynamics(nn.Module):
    def __init__(
        self,
        stoch_dim: int = 32,
        action_dim: int = 2,
        deter_dim: int = 256,
    ):
        super().__init__()

        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim

        input_dim = stoch_dim + action_dim

        self.gru = nn.GRUCell(
            input_size=input_dim,
            hidden_size=deter_dim
        )

    def forward(
        self,
        prev_z: torch.Tensor,
        prev_actionn: torch.Tensor,
        prev_h: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([prev_z, prev_actionn], dim=-1)

        h = self.gru(x, prev_h)

        return h


class PriorNetwork(nn.Module):
    def __init__(
        self,
        deter_dim: int = 256,
        hidden_dim: int = 128,
        stoch_dim: int = 32,
    ):
        super().__init__()

        self.stoch_dim = stoch_dim

        self.network = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, stoch_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)

    def get_dist(self, h: torch.Tensor):
        logits = self.forward(h)

        return torch.distributions.Categorical(logits=logits)
