import sys
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv
from main import draw_background, draw_ground, draw_pipe, draw_bird
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS


class FlappyBirdGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = FlappyBirdEnv()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done = self.env.step(action)
        return np.array(state, dtype=np.float32), float(reward), done, False, {}


if __name__ == '__main__':
    model = PPO.load("ia/best_model_optuna/best_model")
    env = FlappyBirdGymEnv()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - PPO")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)

    bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    draw_background(bg)
    draw_ground(bg)

    running = True
    while running:
        obs, _ = env.reset()
        done = False
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            screen.blit(bg, (0, 0))
            for pipe in env.env.pipes:
                draw_pipe(screen, pipe)
            draw_bird(screen, env.env.bird)

            surf = font.render(f"Score : {env.env.score}", True, (255, 255, 255))
            screen.blit(surf, (10, 10))

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()