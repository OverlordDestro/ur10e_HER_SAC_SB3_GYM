import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
import sys
import os
sys.path.append(os.path.dirname(__file__))
import ur10e
print("ur10e imported")

env = gym.make("UR10E-v0", render_mode="human", max_episode_steps=500, no_early_termination=True)
model = SAC.load("ur10e_checkpoint", env=env, device="cpu")
#model = PPO.load("ur10e_checkpoint", env=env, device="cpu")
observation, info = env.reset(seed=40) 

component_names = [
    "xy_reward", "z_reward", "orientation_reward", "gripper_reward", "distance_reward","potential_reward"
]

episode_rewards = []
episode_count = 0
num_steps = 5000

summed_reward = 0.0
summed_components = {k: 0.0 for k in component_names}
all_episode_components = {k: [] for k in component_names}

for step in range(num_steps):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    summed_reward += reward

    components = info.get("reward_components", {})
    for k in component_names:
        summed_components[k] += components.get(k, 0.0)

    if terminated or truncated:
        episode_rewards.append(summed_reward)
        for k in component_names:
            all_episode_components[k].append(summed_components[k])

        episode_count += 1
        ag_pos        = observation['achieved_goal'][:3]
        dg_pos        = observation['desired_goal'][:3]
        obj_pos       = observation['observation'][15:18]
        current_stage = int(observation['observation'][-1])
        dist_to_goal  = np.linalg.norm(ag_pos - dg_pos)
        dist_to_box   = np.linalg.norm(ag_pos - obj_pos)
        print(f"Episode {episode_count}: Reward={summed_reward:.2f}, "
              f"Stage={current_stage}, "
              f"Dist to goal={dist_to_goal:.4f}m, "
              f"Dist to box={dist_to_box:.4f}m")

        summed_reward = 0.0
        summed_components = {k: 0.0 for k in component_names}
        observation, info = env.reset()

env.close()

episodes = list(range(1, episode_count + 1))
colors = plt.cm.tab10.colors

cols = 2
rows = (len(component_names) + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
axes = axes.flatten()

for idx, k in enumerate(component_names):
    ax = axes[idx]
    ax.plot(episodes, all_episode_components[k],
            marker='o', color=colors[idx % len(colors)], linewidth=2)
    ax.set_xticks(episodes)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(k, fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative per episode")
    ax.grid(True)

for idx in range(len(component_names), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Reward Components per Episode — UR10e Eval", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig("eval_rewards.png", dpi=150)
print("Graph saved to eval_rewards.png")
plt.show()