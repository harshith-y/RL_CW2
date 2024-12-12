import pandas as pd
import matplotlib.pyplot as plt
import json
import subprocess
import os
import numpy as np
import torch
from stable_baselines3 import SAC
import gymnasium as gym

# Paths
monitor_logs_zip = "./Train500k/monitor_logs.zip"
tensorboard_logs_zip = "./Train500k/sac_tensorboard.zip"

# Unzip monitor logs
if os.path.exists(monitor_logs_zip):
    subprocess.run(["unzip", "-o", monitor_logs_zip, "-d", "./Train500k/monitor_logs/"])
    print("Monitor logs unzipped successfully.")
else:
    print(f"File not found: {monitor_logs_zip}")

# Unzip TensorBoard logs
if os.path.exists(tensorboard_logs_zip):
    subprocess.run(["unzip", "-o", tensorboard_logs_zip, "-d", "./Train500k/sac_tensorboard/"])
    print("TensorBoard logs unzipped successfully.")
else:
    print(f"File not found: {tensorboard_logs_zip}")




# Load the trained model
model = SAC.load("./Train500k/sac_halfcheetah_500k_model.zip")

env_name = "HalfCheetah-v4"
env = gym.make(env_name, render_mode="rgb_array")

########## MONITOR LOGS

# Load monitor logs
monitor_data = pd.read_csv("./Train500k/monitor_logs/monitor.csv", skiprows=1)

# Load evaluation data
evaluation_data = np.load("./Train500k/evaluations-2.npz")

# Extract timesteps, mean rewards, and standard deviations
timesteps = evaluation_data["timesteps"]
mean_rewards = evaluation_data["results"].mean(axis=1)  # Mean over evaluation episodes
std_rewards = evaluation_data["results"].std(axis=1)  # Standard deviation over evaluation episodes

# Plot the learning curve with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, label='Reward per Evaluation (Mean)')
plt.fill_between(
    timesteps,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    color='blue',
    alpha=0.2,
    label='Standard Deviation'
)
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.title('Learning Curve (SAC)')
plt.legend()
plt.grid()
plt.show()



########## TRAINING LOGS

# Load training logs
with open("./Train500k/training_logs.json", "r") as f:
    training_logs = json.load(f)

# Plot policy loss
plt.figure(figsize=(10, 6))
plt.plot(training_logs["policy_losses"], label='Policy Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Policy Loss During Training')
plt.legend()
plt.grid()
plt.show()

# Plot Q-loss
plt.figure(figsize=(10, 6))
plt.plot(training_logs["q_losses"], label='Q Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Q Loss During Training')
plt.legend()
plt.grid()
plt.show()



########## EVALUATION LOGS

# Load evaluation data
evaluation_data = np.load("./Train500k/evaluations-2.npz")
timesteps = evaluation_data["timesteps"]
mean_rewards = evaluation_data["results"].mean(axis=1)
std_rewards = evaluation_data["results"].std(axis=1)

# Plot evaluation results
plt.figure(figsize=(10, 6))
plt.errorbar(timesteps, mean_rewards, yerr=std_rewards, fmt='-o', label='Mean Reward ± Std')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Evaluation Results')
plt.legend()
plt.grid()
plt.show()



########## HEATMAP

def plot_q_values_all_pairs_final(model, env, n_points=50):
    """
    Plots Q-value heatmaps for all pairs of action dimensions.
    Ensures labels for all dimensions, including `Dim 1`.
    """
    obs, _ = env.reset()  # Get a representative observation
    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32).to(model.device)

    n_dims = env.action_space.shape[0]
    action_ranges = np.linspace(-1, 1, n_points)
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(10, 10), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                # Turn off diagonal subplots but ensure labels appear
                axes[i, j].axis("off")
                continue

            # Create grid for two action dimensions
            action_grid = np.meshgrid(action_ranges, action_ranges)
            q_values = np.zeros((n_points, n_points))

            for x in range(n_points):
                for y in range(n_points):
                    action = np.zeros(n_dims)
                    action[i] = action_grid[0][x, y]
                    action[j] = action_grid[1][x, y]

                    # Convert action to tensor
                    action_tensor = torch.tensor(action.reshape(1, -1), dtype=torch.float32).to(model.device)
                    with torch.no_grad():
                        q_value = model.critic_target(obs_tensor, action_tensor)[0].cpu().numpy()
                    q_values[x, y] = q_value

            # Plot heatmap for the pair (i, j)
            im = axes[i, j].imshow(
                q_values,
                extent=[-1, 1, -1, 1],
                origin="lower",
                aspect="auto",
                cmap="viridis"
            )

    # Add global labels for rows and columns
    for i in range(n_dims):
        # Add column titles on the top row
        axes[0, i].set_title(f"Dim {i+1}", fontsize=6)
        # Add row labels on the first column
        axes[i, 0].set_ylabel(f"Dim {i+1}", fontsize=6)

    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Q-Value", fontsize=8)  # Label for the colorbar

    fig.suptitle("Q-Value Heatmaps for Action Dimension Pairs", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for colorbar
    plt.show()

# Call the function
plot_q_values_all_pairs_final(model, env)



########## ACTION DISTRIBUTION (POLICY)

def visualize_action_distribution_all_dims(model, env, n_samples=1000):
    """
    Visualizes the distribution of actions for all action dimensions.
    """
    actions = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)  # Sample stochastic actions
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    actions = np.array(actions)
    n_dims = actions.shape[1]

    # Create subplots for each action dimension
    fig, axes = plt.subplots(1, n_dims, figsize=(15, 5), sharey=True)
    for i in range(n_dims):
        axes[i].hist(actions[:, i], bins=50, density=True, alpha=0.7, label=f'Dim {i+1}')
        axes[i].set_title(f'Action Dim {i+1}')
        axes[i].set_xlabel('Action Value')
        axes[i].grid(True)

    plt.suptitle('Policy Action Distribution for All Dimensions')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# Call the function
visualize_action_distribution_all_dims(model, env)
