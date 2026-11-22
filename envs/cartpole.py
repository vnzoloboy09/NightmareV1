import gymnasium as gym
import numpy as np


class CartPoleEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make(
            "CartPole-v1",
            render_mode=render_mode
        )

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(int(action))
        )

        done = terminated or truncated

        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self):
        self.env.close()