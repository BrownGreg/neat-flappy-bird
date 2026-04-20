import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    BIRD_X, BIRD_Y, BIRD_RADIUS,
    GRAVITY, JUMP_VELOCITY, MAX_VELOCITY,
    GROUND_HEIGHT, PIPE_WIDTH,
    DIFFICULTY_LEVELS,
)


def get_difficulty(score):
    """Retourne (pipe_speed, gap, spawn_interval_ms) selon le score."""
    level = DIFFICULTY_LEVELS[0]
    for entry in DIFFICULTY_LEVELS:
        if score >= entry[0]:
            level = entry
    return level[1], level[2], level[3]


class Bird:
    def __init__(self):
        self.x = float(BIRD_X)
        self.y = float(BIRD_Y)
        self.velocity = 0.0
        self.angle = 0.0

    def jump(self):
        self.velocity = JUMP_VELOCITY

    def update(self):
        self.velocity += GRAVITY
        if self.velocity > MAX_VELOCITY:
            self.velocity = MAX_VELOCITY
        self.y += self.velocity
        # Inclinaison visuelle selon la vélocité
        self.angle = max(-30.0, min(90.0, self.velocity * 6))

    def get_rect(self):
        """Retourne (x, y, w, h) de la boîte englobante."""
        return (
            self.x - BIRD_RADIUS,
            self.y - BIRD_RADIUS,
            BIRD_RADIUS * 2,
            BIRD_RADIUS * 2,
        )

    def is_out_of_bounds(self):
        if self.y + BIRD_RADIUS >= SCREEN_HEIGHT - GROUND_HEIGHT:
            return True
        if self.y - BIRD_RADIUS <= 0:
            return True
        return False


class Pipe:
    def __init__(self, x, gap_center_y, gap):
        self.x = float(x)
        self.gap_center_y = gap_center_y
        self.gap = gap
        self.passed = False

    @property
    def top_height(self):
        return self.gap_center_y - self.gap // 2

    @property
    def bottom_y(self):
        return self.gap_center_y + self.gap // 2

    def get_top_rect(self):
        return (self.x, 0, PIPE_WIDTH, self.top_height)

    def get_bottom_rect(self):
        bottom_height = SCREEN_HEIGHT - GROUND_HEIGHT - self.bottom_y
        return (self.x, self.bottom_y, PIPE_WIDTH, bottom_height)

    def update(self, speed):
        self.x -= speed

    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0

    def collides_with(self, bird):
        bx, by, bw, bh = bird.get_rect()
        # Marge de tolérance légère pour un ressenti plus juste
        margin = 3
        bx += margin
        by += margin
        bw -= margin * 2
        bh -= margin * 2

        tx, ty, tw, th = self.get_top_rect()
        if bx + bw > tx and bx < tx + tw and by < ty + th:
            return True

        rx, ry, rw, rh = self.get_bottom_rect()
        if bx + bw > rx and bx < rx + rw and by + bh > ry:
            return True

        return False


class FlappyBirdEnv:
    """
    Environnement Flappy Bird utilisable sans affichage (mode headless).

    Interface :
        reset()          -> état initial (list[float])
        step(action)     -> (état, récompense, done)
        get_state()      -> list[float] de 5 valeurs normalisées
    """

    def __init__(self, render=False, screen=None):
        self.bird = None
        self.pipes = []
        self.score = 0
        self.frames = 0
        self._ms_since_last_pipe = 0.0
        self._pipe_speed = 2.0
        self._gap = 180
        self._spawn_interval = 1800
        self._ms_per_frame = 1000.0 / FPS

    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frames = 0
        self._ms_since_last_pipe = 0.0
        self._pipe_speed, self._gap, self._spawn_interval = get_difficulty(0)
        self._spawn_pipe()
        return self.get_state()

    def _spawn_pipe(self):
        margin = 110
        gap_y = random.randint(margin, SCREEN_HEIGHT - GROUND_HEIGHT - margin)
        self.pipes.append(Pipe(SCREEN_WIDTH + 10, gap_y, self._gap))

    def step(self, action):
        """
        action : 0 = ne rien faire, 1 = sauter
        Retourne : (state, reward, done)
        """
        if action == 1:
            self.bird.jump()

        self.bird.update()

        self._ms_since_last_pipe += self._ms_per_frame
        if self._ms_since_last_pipe >= self._spawn_interval:
            self._ms_since_last_pipe = 0.0
            self._spawn_pipe()

        pipes_passed = 0
        for pipe in self.pipes:
            pipe.update(self._pipe_speed)
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
                pipes_passed += 1
                self._pipe_speed, self._gap, self._spawn_interval = get_difficulty(self.score)

        self.pipes = [p for p in self.pipes if not p.is_off_screen()]
        self.frames += 1

        done = self.bird.is_out_of_bounds()
        if not done:
            for pipe in self.pipes:
                if pipe.collides_with(self.bird):
                    done = True
                    break

        reward = 1 + pipes_passed * 500
        if done:
            reward = -100

        return self.get_state(), reward, done

    def get_state(self):
        """
        Vecteur d'état normalisé à 5 dimensions :
          [0] bird_y / SCREEN_HEIGHT
          [1] bird_velocity / MAX_VELOCITY
          [2] distance_au_prochain_tuyau / SCREEN_WIDTH
          [3] (bird_y - bord_haut_gap) / SCREEN_HEIGHT
          [4] (bord_bas_gap - bird_y) / SCREEN_HEIGHT
        """
        bird = self.bird

        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > bird.x:
                if next_pipe is None or pipe.x < next_pipe.x:
                    next_pipe = pipe

        if next_pipe is None:
            dist_norm = 1.0
            top_gap_norm = 0.5
            bottom_gap_norm = 0.5
        else:
            dist_norm = max(0.0, min(1.0, (next_pipe.x - bird.x) / SCREEN_WIDTH))
            top_gap_norm = (bird.y - next_pipe.top_height) / SCREEN_HEIGHT
            bottom_gap_norm = (next_pipe.bottom_y - bird.y) / SCREEN_HEIGHT

        return [
            bird.y / SCREEN_HEIGHT,
            bird.velocity / MAX_VELOCITY,
            dist_norm,
            top_gap_norm,
            bottom_gap_norm,
        ]
