import torch
import torch.nn as nn

from models.rssm import RSSM
from models.decoder import ObservationDecoder
from models.reward_predictor import RewardPredictor


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim=4,
        action_dim=2,
        embed_dim=128,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=128,
    ):
        super().__init__()

        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )

        self.decoder = ObservationDecoder(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
        )

        self.reward_predictor = RewardPredictor(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )

    def observe(self, obs, actions):
        posterior_states, prior_logits, posterior_logits = self.rssm.observe(obs, actions)

        reconstructed_obs = self.decoder(
            posterior_states.deter,
            posterior_states.stoch,
        )

        predicted_rewards = self.reward_predictor(
            posterior_states.deter,
            posterior_states.stoch,
        )

        return {
            "posterior_states": posterior_states,
            "prior_logits": prior_logits,
            "posterior_logits": posterior_logits,
            "reconstructed_obs": reconstructed_obs,
            "predicted_rewards": predicted_rewards,
        }