from envs.cartpole import CartPoleEnv
from models.world_model import WorldModel
from training.world_model_trainer import WorldModelTrainer
from training.collector import RandomCollector
from buffers.replay_buffer import ReplayBuffer


def train_world_model(
    env,
    num_episodes=10,
    update_per_episodes=20,
    batch_size=16,
    sequence_length=5
):
    collector = RandomCollector(env)
    replay_buffer = ReplayBuffer(capacity=100_000)

    world_model = WorldModel()
    trainer = WorldModelTrainer(world_model=world_model)

    num_updates = 0

    for episode_idx in range(num_episodes):
        episode = collector.collect_episode()

        replay_buffer.add_episode(episode=episode)

        episode_losses = []

        for _ in range(update_per_episodes):
            try:
                batch = replay_buffer.sample(
                    batch_size=batch_size,
                    sequence_length=sequence_length
                )
            except ValueError:
                print("ahh fuck")
                break

            metrix = trainer.train_step(
                obs=batch["obs"],
                actions=batch["actions"],
                rewards=batch["rewards"]
            )

            episode_losses.append(metrix["total_loss"])
            num_updates += 1

        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)

            print(
                f"Episode {episode_idx + 1}/{num_episodes} | "
                f"Buffer steps: {replay_buffer.num_steps} | "
                f"Updates: {num_updates} | "
                f"Average loss: {avg_loss:.4f}"
            )

    return world_model, trainer


def main():
    env = CartPoleEnv()

    world_model, trainer = train_world_model(env=env)

    env.close()


if __name__ == "__main__":
    main()