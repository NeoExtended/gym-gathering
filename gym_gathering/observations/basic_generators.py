from typing import Tuple

import gym
import numpy as np

from gym_gathering.observations.base_observation_generator import (
    ObservationGenerator,
)


class SingleChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self, maze: np.ndarray, random_goal: bool, goal_range: int, noise: float = 0.0
    ):
        super(SingleChannelObservationGenerator, self).__init__(
            maze, random_goal, goal_range, noise
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, 1), dtype=np.uint8
        )

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros(maze.shape)
        observation = self.render_particles(particles, maze, out=observation)
        observation = self.generate_noise(observation, maze)

        if self.random_goal:
            observation = self.render_goal(maze, goal, out=observation)

        return observation[:, :, np.newaxis]  # Convert to single channel image


class MultiChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self, maze: np.ndarray, random_goal: bool, goal_range: int, noise: float = 0.0
    ):
        super(MultiChannelObservationGenerator, self).__init__(
            maze, random_goal, goal_range, noise
        )

        self.n_channels = 3 if random_goal else 2
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, self.n_channels), dtype=np.uint8
        )

    def observation(
        self, maze: np.ndarray, particles: np.ndarray, goal: Tuple[int, int]
    ):
        observation = np.zeros((*maze.shape, self.n_channels))
        observation[:, :, 0] = self.render_maze(maze)
        particle_image = self.render_particles(particles, maze)
        particle_image = self.generate_noise(particle_image, maze)
        observation[:, :, 1] = particle_image

        if self.random_goal:
            observation[:, :, 2] = self.render_goal(maze, goal)

        return observation