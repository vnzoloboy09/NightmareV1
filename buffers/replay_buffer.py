from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.episodes = deque()
        self.capacity = capacity
        self.num_steps = 0

    def add_episode(self, episode):
        episode_size = len(episode["actions"])

        if episode_size == 0:
            return

        self.episodes.append(episode)
        self.num_steps += episode_size

        while (
            self.num_steps > self.capacity
            and len(self.episodes) > 1
        ):
            removed = self.episodes.popleft()
            self.num_steps -= len(removed["actions"])

    def sample(self, batch_size, sequence_length):
        valid_episodes = [
            ep for ep in self.episodes
            if len(ep["actions"]) >= sequence_length
        ]

        if not valid_episodes:
            raise ValueError("Not enough data to sample")

        batch = []

        for _ in range(batch_size):
            ep = valid_episodes[np.random.randint(len(valid_episodes))]

            max_start = (len(ep["actions"]) - sequence_length)
    
            start = np.random.randint(max_start + 1)
            end = start + sequence_length

            batch.append({
                "obs": ep["obs"][start:end + 1],
                "actions": ep["actions"][start:end],
                "rewards": ep["rewards"][start:end],
                "terminated": ep["terminated"][start:end],
                "truncated": ep["truncated"][start:end],
            })

        return {
            key: np.stack([item[key] for item in batch])
            for key in batch[0]
        }

    def __len__(self):
        return self.num_steps