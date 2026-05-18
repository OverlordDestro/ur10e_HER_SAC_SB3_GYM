import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, HerReplayBuffer, PPO
from stable_baselines3.common.monitor import Monitor
import os

from stable_baselines3.common.callbacks import BaseCallback
import ur10e

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
MAX_EPISODE_STEPS = 1000 #1000 simulation steps = 500 episode steps * 2 frame skip
TOTAL_TIMESTEPS   = MAX_EPISODE_STEPS * 2000 #episodes * max_episode_steps
LOG_EVERY         = 10        # episodes between logged mean rewards
CHECKPOINT_EVERY  = 40        # episodes between checkpoint saves

def make_env():
    return gym.make("UR10E-v0", render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

class RewardLoggerCallback(BaseCallback): 
    def __init__(self, log_every=LOG_EVERY, checkpoint_every=CHECKPOINT_EVERY, n_envs=1):
        super().__init__()
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.n_envs = n_envs
        self.ep_rew_means = []
        self.episode_rewards = []
        self.current_ep_rewards = [0.0] * n_envs
        self.total_episode_count = 0

        # Component tracking
        self.component_names = [
            "xy_reward", "z_reward", "orientation_reward", "gripper_reward","distance_reward"
        ]
        self.current_ep_components = [{k: 0.0 for k in self.component_names} for _ in range(n_envs)]
        self.ep_component_means = {k: [] for k in self.component_names}

    def _on_step(self):
        infos = self.locals.get("infos", [{}] * self.n_envs)

        for i in range(self.n_envs):
            self.current_ep_rewards[i] += self.locals["rewards"][i]

            # Accumulate reward components
            components = infos[i].get("reward_components", {})
            for k in self.component_names:
                self.current_ep_components[i][k] += components.get(k, 0.0)

            if self.locals["dones"][i]:
                self.total_episode_count += 1
                self.episode_rewards.append(self.current_ep_rewards[i])

                # Store per-episode component totals
                for k in self.component_names:
                    self.ep_component_means[k].append(self.current_ep_components[i][k])

                # Reset accumulators
                self.current_ep_rewards[i] = 0.0
                self.current_ep_components[i] = {k: 0.0 for k in self.component_names}

                if len(self.episode_rewards) % self.log_every == 0:
                    mean = np.mean(self.episode_rewards[-self.log_every:])
                    self.ep_rew_means.append(mean)
                    print(f"Episode {self.total_episode_count}: Mean reward (last {self.log_every}): {mean:.2f}")

                if self.total_episode_count % self.checkpoint_every == 0:
                    self.model.save("ur10e_checkpoint")
                    print(f"💾 Checkpoint saved")

        return True

    def plot(self):
        colors = plt.cm.tab10.colors  # 10 distinct colors

        # ── Total reward plot ──────────────────────────────────────────────
        x = [(i + 1) * self.log_every for i in range(len(self.ep_rew_means))]
        plt.figure(figsize=(10, 5))
        plt.plot(x, self.ep_rew_means, color='steelblue')
        plt.xlabel("Episode")
        plt.ylabel(f"Mean reward (last {self.log_every} eps)")
        plt.title("Total Reward — SAC+HER on UR10e")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_reward_total.png", dpi=150)
        plt.show()

        # ── Per-component plots ────────────────────────────────────────────
        n = len(self.component_names)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
        axes = axes.flatten()

        episodes = list(range(1, len(next(iter(self.ep_component_means.values()))) + 1))

        for idx, k in enumerate(self.component_names):
            ax = axes[idx]
            color = colors[idx % len(colors)]
            data = np.array(self.ep_component_means[k])
            if len(data) >= self.log_every:
                smoothed = np.convolve(data, np.ones(self.log_every) / self.log_every, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color=color, linewidth=2)
            else:
                ax.plot(episodes, data, color=color, linewidth=2)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_title(k)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Cumulative per episode")
            ax.grid(True)

        for idx in range(len(self.component_names), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Reward Components — SAC+HER on UR10e", fontsize=14)
        plt.tight_layout()
        plt.savefig("training_reward_components.png", dpi=150)
        plt.show()

        
# ###############################
# 1. Create the environment
# ###############################



log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True) 
n_envs = 32 #4
env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv, monitor_dir=log_dir)

#env = gym.make("UR10E-v0", render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)#prvo igra 200 ucilne simulacije in potem 100 primerov
#env = gym.make("UR10E-v0", render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)#prvo igra 200 ucilne simulacije in potem 100 primerov

#max_episode_steps = 50
#env = gym.make("FetchReachDense-v4", render_mode="None", max_episode_steps=50)#prvo igra 200 ucilne simulacije in potem 100 primerov

# ###############################
# 2. Create SAC model (CPU)
# ###############################
#env = Monitor(env, log_dir)
model = SAC(
    "MultiInputPolicy", 
    env, 
    use_sde=False,  # Use State-Dependent Exploration for better smoothness
    sde_sample_freq=4,  # Sample a new noise value every 4 steps for smoother actions
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    learning_starts=32000,  # Warm-up steps before training starts #30 eps
    buffer_size=500000,  # Size of the replay buffer
    batch_size=512,  # Batch size for training (16x n_envs)
    gradient_steps=1,  # Update the model every step
    tau=0.005,  # Soft update coefficient
    device="cuda", 
    verbose=1,
    learning_rate=1e-3, # HER often benefits from a slightly higher learning rate
    gamma=0.99,          # Slightly lower gamma helps focus on reaching the goal
    ent_coef="auto"

)
"""model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,                 # Steps per environment per update
    batch_size=64,               # Batch size for gradient updates
    n_epochs=10,                 # Number of epochs per training update
    gamma=0.99,
    gae_lambda=0.95,             # Generalised Advantage Estimation lambda
    clip_range=0.2,              # PPO clipping parameter
    ent_coef=0.01,               # Entropy coefficient for exploration
    vf_coef=0.5,                 # Value function coefficient
    max_grad_norm=0.5,           # Gradient clipping
    device="cuda",
    verbose=1,
)"""
# ###############################
# 3. Train the model
# ###############################
# Training for demonstration purposes (you can increase timesteps for better performance)

callback = RewardLoggerCallback(log_every=32, checkpoint_every=100, n_envs=n_envs)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# Save the trained model
model.save("ur10e_stage_0")
callback.plot()
# ###############################
# 4. Run simulation with trained policy
# ###############################
print("Learning complete, continue to evaluation")
input("Press Enter to continue...")

env.close()