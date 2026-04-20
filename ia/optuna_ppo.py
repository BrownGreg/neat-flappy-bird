import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv

N_ENVS = 4
TRIAL_TIMESTEPS = 500_000
FINAL_TIMESTEPS = 5_000_000
N_TRIALS = 50
N_JOBS = 4

ARCH_MAP = {
    "64x64":       [64, 64],
    "128x128":     [128, 128],
    "256x256":     [256, 256],
    "256x256x128": [256, 256, 128],
    "512x512x256": [512, 512, 256],
}


class FlappyBirdGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_coef_gap: float = 0.5, reward_penalty_death: float = 500.0):
        super().__init__()
        self.env = FlappyBirdEnv()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.reward_coef_gap = reward_coef_gap
        self.reward_penalty_death = reward_penalty_death

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done = self.env.step(action)
        if state[3] > 0 and state[4] > 0:
            reward += 1.0
            center = (state[3] + state[4]) / 2
            reward += center * self.reward_coef_gap
        if done:
            reward -= self.reward_penalty_death
        return np.array(state, dtype=np.float32), float(reward), done, False, {}


def _make_env(reward_coef_gap: float, reward_penalty_death: float) -> gym.Env:
    return Monitor(FlappyBirdGymEnv(reward_coef_gap, reward_penalty_death))


def objective(trial: optuna.Trial) -> float:
    lr              = trial.suggest_float("learning_rate",        1e-5,  1e-3,  log=True)
    n_steps         = trial.suggest_categorical("n_steps",        [1024, 2048, 4096, 8192])
    batch_size      = trial.suggest_categorical("batch_size",     [64, 128, 256, 512])
    n_epochs        = trial.suggest_int("n_epochs",               5, 30)
    gamma           = trial.suggest_float("gamma",                0.98,  0.999)
    gae_lambda      = trial.suggest_float("gae_lambda",           0.9,   0.99)
    clip_range      = trial.suggest_float("clip_range",           0.1,   0.4)
    ent_coef        = trial.suggest_float("ent_coef",             1e-4,  0.1,   log=True)
    net_arch        = trial.suggest_categorical("net_arch",       list(ARCH_MAP.keys()))
    reward_coef_gap        = trial.suggest_float("reward_coef_gap",        0.5,   3.0)
    reward_penalty_death   = trial.suggest_float("reward_penalty_death",   100.0, 1000.0)

    batch_size = min(batch_size, n_steps * N_ENVS)

    env_fns = [partial(_make_env, reward_coef_gap, reward_penalty_death) for _ in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns)
    eval_env  = DummyVecEnv([partial(_make_env, reward_coef_gap, reward_penalty_death)])

    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            device="cuda",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=ARCH_MAP[net_arch]),
            verbose=0,
        )
        model.learn(total_timesteps=TRIAL_TIMESTEPS)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    finally:
        train_env.close()
        eval_env.close()

    return float(mean_reward)


def train_final(best_params: dict) -> None:
    reward_coef_gap      = best_params["reward_coef_gap"]
    reward_penalty_death = best_params["reward_penalty_death"]
    batch_size           = min(best_params["batch_size"], best_params["n_steps"] * N_ENVS)

    env_fns   = [partial(_make_env, reward_coef_gap, reward_penalty_death) for _ in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns)
    eval_env  = DummyVecEnv([partial(_make_env, reward_coef_gap, reward_penalty_death)])

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000 // N_ENVS,
        best_model_save_path="./ia/best_model_optuna/",
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        device="cuda",
        n_steps=best_params["n_steps"],
        batch_size=batch_size,
        n_epochs=best_params["n_epochs"],
        learning_rate=best_params["learning_rate"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=ARCH_MAP[best_params["net_arch"]]),
        verbose=1,
        tensorboard_log="./ia/ppo_tensorboard_optuna/",
    )

    print("\nEntraînement final (5M timesteps)...")
    print("Tensorboard : tensorboard --logdir ./ia/ppo_tensorboard_optuna/")
    model.learn(total_timesteps=FINAL_TIMESTEPS, callback=eval_callback)

    model.save("ia/ppo_optuna_final")
    print("Modèle sauvegardé dans ia/ppo_optuna_final.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_results.db")
    storage = f"sqlite:///{DB_PATH}"

    study = optuna.create_study(
        study_name="flappy_ppo",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    print(f"Lancement de {N_TRIALS} trials Optuna (n_jobs={N_JOBS})...")
    print(f"Résultats stockés dans : {DB_PATH}")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True,
    )

    print("\n" + "=" * 50)
    print("MEILLEURS HYPERPARAMÈTRES")
    print("=" * 50)
    print(f"  Mean reward : {study.best_value:.2f}")
    for key, val in study.best_params.items():
        print(f"  {key:25s}: {val}")

    train_final(study.best_params)
