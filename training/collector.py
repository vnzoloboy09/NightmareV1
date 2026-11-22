import numpy as np

from envs.cartpole import CartPoleEnv


class RandomCollector:
    def __init__(self, env: CartPoleEnv):
        self.env = env

    def collect_episode(self, seed=None):
        obs = self.env.reset(seed=seed)

        observations = [obs]
        actions = []
        rewards = []
        terminateds = []
        truncateds = []

        while True:
            action = self.env.sample_random_action()

            next_obs, reward, terminated, truncated, _ = (
                self.env.step(action)
            )

            observations.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)

            if terminated or truncated:
                break

        return {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "terminated": np.asarray(terminateds, dtype=np.bool_),
            "truncated": np.asarray(truncateds, dtype=np.bool_),
        }