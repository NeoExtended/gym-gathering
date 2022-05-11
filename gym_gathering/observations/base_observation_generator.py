from abc import ABC
from typing import Tuple, Optional

import cv2
import numpy as np

PARTICLE_MARKER = 255
GOAL_MARKER = 255


class ObservationGenerator(ABC):
    def __init__(
        self,
        random_goal: bool,
        goal_range: int,
        noise: float = 0.0,
        static_noise: float = 0.0,
    ):
        self.np_random = np.random.random.__self__
        self.observation_space = None
        self.random_goal = random_goal
        self.goal_range = goal_range
        self.noise = noise
        self.static_noise = static_noise

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        pass

    def render_particles(self, particles: np.ndarray, maze: np.ndarray, out=None):
        out = out if out is not None else np.zeros(maze.shape)
        out[particles[:, 0], particles[:, 1]] = PARTICLE_MARKER
        return out

    def generate_noise(self, image, maze: Optional[np.ndarray] = None):
        out = self._generate_noise(
            image, self.static_noise, noise_type="s&p", maze=maze
        )
        out = self._generate_noise(out, self.noise, noise_type="gauss", maze=maze)
        return out

    def _generate_noise(
        self,
        image: np.ndarray,
        strength: float,
        noise_type: str = "s&p",
        maze: Optional[np.ndarray] = None,
    ):
        out = image
        if strength > 0.0:
            if noise_type == "s&p":
                out = self.salt_and_pepper_noise(image, strength)
            elif noise_type == "gauss":
                out = self.gaussian_noise(image, strength)
            else:
                raise NotImplementedError(f"Unknown noise type {noise_type}")

            # Restrict noise to the maze area
            if maze is not None:
                out = image * (1 - maze)
        return out

    def gaussian_noise(self, image, strength):
        row, col = image.shape
        mean = 0
        var = strength
        sigma = var ** 0.5
        gauss = self.np_random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = np.clip(image + gauss * 255, 0, 255)
        return noisy

    def salt_and_pepper_noise(self, image, strength):
        n_salt = np.ceil(strength * image.size)
        coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in image.shape]
        image[tuple(coords)] = PARTICLE_MARKER

        coords = [self.np_random.randint(0, i - 1, int(n_salt)) for i in image.shape]
        image[tuple(coords)] = 0
        return image

    def render_maze(self, maze):
        return maze * 255

    def render_goal(self, maze: np.ndarray, goal: Tuple[int, int], out=None):
        out = out if out is not None else np.zeros(maze.shape)
        cv2.circle(out, tuple(goal), self.goal_range, GOAL_MARKER)
        out[goal[1] - 1 : goal[1] + 1, goal[0] - 1 : goal[0] + 1] = GOAL_MARKER
        return out

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random

    def reset(self):
        self.dirt = self.salt_and_pepper_noise(
            np.zeros(self.real_world_size), self.dirt_noise
        )

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random
