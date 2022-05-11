from typing import Tuple

import cv2
import gym
import numpy as np

from gym_gathering.observations.base_observation_generator import ObservationGenerator


class SingleChannelRealWorldObservationGenerator(ObservationGenerator):
    def __init__(
        self,
        maze: np.ndarray,
        random_goal: bool,
        goal_range: int,
        noise: float = 0.0,
        static_noise: float = 0.0,
        restrict_noise: bool = False,
        real_world_fac: float = 2,
        max_displacement: int = 5,
        max_crop: int = 5,
    ):
        super(SingleChannelRealWorldObservationGenerator, self).__init__(
            random_goal=random_goal,
            goal_range=goal_range,
            noise=noise,
            static_noise=static_noise,
            restrict_noise=restrict_noise,
        )
        self.real_world_fac = real_world_fac
        self.real_world_size = tuple([int(d * self.real_world_fac) for d in maze.shape])
        self.displacement = (0, 0)
        self.crop = (0, 0, 0, 0)
        self.dirt = np.ndarray([])
        self.max_displacement = max_displacement
        self.max_crop = max_crop
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, 1), dtype=np.uint8
        )

    def observation(
        self, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros(self.maze.shape)
        observation = self.render_particles(particles, out=observation)
        observation = self.distort(observation)

        if self.random_goal:
            observation = self.render_goal(goal, out=observation)

        return observation[:, :, np.newaxis]  # Convert to single channel image

    def distort(self, observation):
        output_shape = observation.shape

        # Scale up
        observation = cv2.resize(
            observation,
            (self.real_world_size[1], self.real_world_size[0]),
            interpolation=cv2.INTER_AREA,
        )

        # Add stationary dirt
        observation = np.clip(observation + self.dirt, 0, 255)

        # Threshold
        ret, particles = cv2.threshold(observation, 200, 255, cv2.THRESH_BINARY)

        # Random Crop
        y, x = particles.shape
        trim_left, trim_right, trim_top, trim_bot = self.crop
        particles = particles[trim_top : y - trim_bot, trim_left : x - trim_right]

        # Translate
        particles = self.shift(particles, self.displacement[0], self.displacement[1],)

        # Add Noise
        noisy = self._generate_noise(particles, strength=self.noise, noise_type="s&p", mask_noise=self.restrict_noise)
        noisy = self._generate_noise(noisy, strength=self.noise, noise_type="gauss", mask_noise=self.restrict_noise)

        # Downscale
        downscaled = cv2.resize(
            noisy, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA
        )

        # Threshold
        ret, out = cv2.threshold(downscaled, 80, 255, cv2.THRESH_BINARY)

        # Restrict noise to maze area + 2 pixels
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.dilate((1 - self.maze), kernel, iterations=2)
        out = out * opened

        return out

    def shift(self, image, tx, ty):
        # The number of pixels
        num_rows, num_cols = image.shape[:2]

        # Creating a translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Image translation
        return cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))

    def reset(self, maze: np.ndarray):
        self.maze = maze
        self.displacement = (
            self.np_random.randint(-self.max_displacement, self.max_displacement),
            self.np_random.randint(-self.max_displacement, self.max_displacement),
        )

        self.crop = [self.np_random.randint(0, self.max_crop) for _ in range(4)]

        self.dirt = self._generate_noise(
            np.zeros(self.real_world_size),
            self.static_noise,
            "s&p",
            mask_noise=self.restrict_noise,
        )
