
import numpy as np
import matplotlib.pyplot as plt


def plot_imagined_predictions(
    obs,
    predicted_obs,
    rewards,
    predicted_rewards,
    batch_index=0,
):
    # Convert tensors to NumPy if necessary
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    obs = to_numpy(obs)[batch_index]
    predicted_obs = to_numpy(predicted_obs)[batch_index]
    rewards = to_numpy(rewards)[batch_index]
    predicted_rewards = to_numpy(predicted_rewards)[batch_index, 1:, 0]

    observation_names = [
        "Cart Position",
        "Cart Velocity",
        "Pole Angle",
        "Pole Angular Velocity",
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14))

    timesteps = np.arange(len(obs))

    # Plot observation reconstruction
    for i, name in enumerate(observation_names):
        axes[i].plot(
            timesteps,
            obs[:, i],
            label="Actual",
        )

        axes[i].plot(
            timesteps,
            predicted_obs[:, i],
            label="Imagined",
            linestyle="--",
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Timestep")
        axes[i].legend()
        axes[i].grid(True)

    # Plot reward prediction
    reward_timesteps = np.arange(1, len(rewards) + 1)

    axes[4].plot(
        reward_timesteps,
        rewards,
        label="Actual Reward",
    )

    axes[4].plot(
        reward_timesteps,
        predicted_rewards,
        label="Predicted Reward",
        linestyle="--",
    )

    axes[4].set_title("Reward Prediction")
    axes[4].set_xlabel("Timestep")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()
