import sys
import os
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optuna_ppo import train_final

if __name__ == "__main__":
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_results.db")
    study = optuna.load_study(study_name="flappy_ppo", storage=f"sqlite:///{DB_PATH}")

    print(f"Best trial: {study.best_trial.number} | Score: {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k:25s}: {v}")

    train_final(study.best_params)
