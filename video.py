import os
import gymnasium as gym
from stable_baselines3 import SAC

# ====== 1. Load the Model ======
# Replace "sac_halfcheetah_model.zip" with the path to your trained model
model = SAC.load("./Train100k/sac_halfcheetah_optimal_model.zip")
# model = SAC.load("./sac_halfcheetah_100_model.zip")

# ====== 2. Set Up the Environment ======
# Use the same environment used for training
env_name = "HalfCheetah-v4"
env = gym.make(env_name, render_mode="rgb_array")

# Wrap the environment with RecordVideo to record videos of the agent's behavior
video_dir = "./Train100k/videos/"
# video_dir = "./videos/"
os.makedirs(video_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda episode_id: True)

# ====== 3. Run the Agent ======
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

print(f"Videos saved to: {video_dir}")



