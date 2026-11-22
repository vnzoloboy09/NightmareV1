from envs.cartpole import CartPoleEnv
from buffers.replay_buffer import ReplayBuffer
from training.collector import RandomCollector


def main():
    env = CartPoleEnv()

    buffer = ReplayBuffer(capacity=100_000)
    collector = RandomCollector(env)

    num_episodes = 100

    for episode_idx in range(num_episodes):
        episode = collector.collect_episode()

        buffer.add_episode(episode)

        episode_reward = episode["rewards"].sum()

        print(
            f"Episode: {episode_idx + 1:03d} | "
            f"Length: {len(episode['actions']):03d} | "
            f"Reward: {episode_reward:.1f} | "
            f"Buffer: {len(buffer)}"
        )

    batch = buffer.sample(
        batch_size=16,
        sequence_length=32
    )

    print("\nSampled batch:")

    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    env.close()


if __name__ == "__main__":
    main()