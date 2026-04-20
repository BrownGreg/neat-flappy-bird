import sys
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch
import optuna 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv


class FlappyBirdGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = FlappyBirdEnv()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done = self.env.step(action)

        if state[3] > 0 and state[4] > 0:
            reward += 1.0
            center = (state[3] + state[4]) / 2
            reward += center * 0.5

        if done:
            reward -= 500

        return np.array(state, dtype=np.float32), float(reward), done, False, {}


if __name__ == '__main__':

    env = Monitor(FlappyBirdGymEnv())
    eval_env = Monitor(FlappyBirdGymEnv())

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50000, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=5000,
        best_model_save_path='./ia/best_model_v2/',
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        n_steps=2048,
        batch_size=256,      
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log="./ia/ppo_tensorboard_v2/"
    )

    print("Début de l'entraînement...")
    print("Tensorboard : tensorboard --logdir ./ia/ppo_tensorboard_v2/")

    model.learn(total_timesteps=5_000_000, callback=eval_callback)

    model.save("ia/ppo_final_v2")
    print("Modèle sauvegardé dans ia/ppo_final_v2.zip")