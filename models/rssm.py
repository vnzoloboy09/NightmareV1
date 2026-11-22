import torch
import torch.nn as nn

from models.encoder import ObservationEncoder
from dataclasses import dataclass


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


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
        prev_action: torch.Tensor,
        prev_h: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([prev_z, prev_action], dim=-1)

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


class PosteriorNetwork(nn.Module):
    def __init__(
        self,
        deter_dim: int = 256,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        stoch_dim: int = 32,
    ):
        super().__init__()

        self.stoch_dim = stoch_dim

        input_dim = deter_dim + embed_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, stoch_dim),
        )

    def forward(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, embed], dim = -1)

        return self.network(x)

    def get_dist(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h, embed)

        return torch.distributions.Categorical(logits=logits)


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim = 4,
        action_dim = 2,
        embed_dim = 128,
        deter_dim = 256,
        stoch_dim = 32,
        hidden_dim = 128
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.encoder = ObservationEncoder(
            obs_dim = obs_dim,
            hidden_dim = hidden_dim,    
            embed_dim = embed_dim
        )

        self.dynamics = DeterministicDynamics(
            stoch_dim = stoch_dim,
            action_dim = action_dim,
            deter_dim = deter_dim,
        )

        self.prior = PriorNetwork(
            deter_dim = deter_dim,
            hidden_dim = hidden_dim,
            stoch_dim = stoch_dim
        )

        self.posterior = PosteriorNetwork(
            deter_dim = deter_dim,
            embed_dim = embed_dim,
            hidden_dim = hidden_dim,
            stoch_dim = stoch_dim
        )

    def initial(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        deter = torch.zeros(
            batch_size,
            self.deter_dim,
            device=device,
        )

        stoch = torch.zeros(
            batch_size,
            self.stoch_dim,
            device=device,
        )

        return RSSMState(
            deter=deter,
            stoch=stoch,
        )

    def get_stoch(self, logits):
        probs = torch.softmax(logits, dim = -1)

        dist = torch.distributions.Categorical(probs=probs)

        indicies = dist.sample()

        z_hard = torch.nn.functional.one_hot(
            indicies,
            num_classes=self.stoch_dim
        ).float()

        z = z_hard + probs - probs.detach()

        return z

    def observe(self, obs, actions):
        batch_size, num_obs, _ = obs.shape
        num_actions = num_obs - 1

        if actions.ndim == 2:
            actions = torch.nn.functional.one_hot(
                actions.long(),
                num_classes=self.action_dim
            )

        if actions.shape != (
            batch_size,
            num_actions,
            self.action_dim,
        ):
            raise ValueError("Invalid action shape")

        embeds = self.encoder(obs)

        state = self.initial(batch_size, device=obs.device)

        deter_states = []
        stoch_states = []
        prior_logits_list = []
        posterior_logits_list = []

        for t in range(num_obs):
            if t > 0:
                state.deter = self.dynamics(
                    prev_z=state.stoch,
                    prev_action=actions[:, t-1],
                    prev_h=state.deter
                )

            prior_logits = self.prior(state.deter)

            posterior_logits = self.posterior(state.deter, embeds[:, t])

            state.stoch = self.get_stoch(posterior_logits)

            deter_states.append(state.deter)
            stoch_states.append(state.stoch)

            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

        post_states = RSSMState(
            deter=torch.stack(deter_states, dim=1),
            stoch=torch.stack(stoch_states, dim=1),
        )

        prior_logits = torch.stack(prior_logits_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)

        return post_states, prior_logits, posterior_logits

    def imagine(self, start_state: RSSMState, actions):
        batch_size, horizon = actions.shape[:2]

        if actions.ndim == 2:
            actions = torch.nn.functional.one_hot(
            actions.long(),
            num_classes=self.action_dim,
        ).float()

        if actions.shape != (batch_size, horizon, self.action_dim):
            raise ValueError("Invalid action shape")

        state = RSSMState(
            deter=start_state.deter,
            stoch=start_state.stoch
        )

        deter_states = [state.deter]
        stoch_states = [state.stoch]
        prior_logits_list = []

        for t in range(horizon):
            next_deter = self.dynamics(
                prev_z=state.stoch,
                prev_action=actions[:, t],
                prevh=state.deter
            )

            prior_logits = self.prior(next_deter)
            next_stoch = self.get_stoch(prior_logits)

            state = RSSMState(
                deter=next_deter,
                stoch=next_stoch
            )

            deter_states.append(next_deter)
            stoch_states.append(next_stoch)
            prior_logits_list.append(prior_logits)

        imagined_states = RSSMState(
            deter=torch.stack(deter_states, dim=1),
            stoch=torch.stack(stoch_states, dim=1)
        )

        prior_logits = torch.stack(prior_logits_list, dim=1)

        return imagined_states, prior_logits