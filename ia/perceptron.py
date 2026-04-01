import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

class Perceptron:
    def __init__(self, n_inputs=5):
        self.weights = np.random.uniform(-1, 1, n_inputs)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return np.tanh(z)

    def decide(self, x):
        return 1 if self.forward(x) > 0 else 0


def run(n_games=100):
    env = FlappyBirdEnv()
    scores = []
    net = Perceptron()
    best_score = -1
    best_weights = None
    best_bias = None

    for i in range(n_games):
        state = env.reset()
        done = False
        frames = 0
        while not done:
            action = net.decide(state)
            state, reward, done = env.step(action)
            frames += 1
        scores.append(env.score)
        print(f"Partie {i + 1} : score = {env.score}, frames = {frames}")
        if env.score > best_score:
            best_score = env.score
            best_weights = net.weights.copy()
            best_bias = net.bias
    print(f"\nScore moyen sur {n_games} parties : {sum(scores) / len(scores):.1f}")
    print(f"Meilleur score : {best_score}")
    print(f"Poids : {best_weights}")
    print(f"Biais : {best_bias}")


if __name__ == '__main__':
    run()