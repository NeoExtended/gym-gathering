from typing import Tuple

import gym
import numpy as np

from gym_gathering.observations.base_observation_generator import ObservationGenerator


class SingleChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self,
        maze: np.ndarray,
        random_goal: bool,
        goal_range: int,
        noise: float = 0.0,
        noise_type: str = "gauss",
        static_noise: float = 0.0,
        static_noise_type: str = "s&p",
        restrict_noise: bool = True,
    ):
        super(SingleChannelObservationGenerator, self).__init__(
            random_goal=random_goal,
            goal_range=goal_range,
            noise=noise,
            noise_type=noise_type,
            static_noise=static_noise,
            static_noise_type=static_noise_type,
            restrict_noise=restrict_noise,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, 1), dtype=np.uint8
        )

    def observation(self, particles: np.ndarray, goal: Tuple[int, int]):
        observation = np.zeros(self.maze.shape)
        observation = self.render_particles(particles, out=observation)
        observation = self.generate_noise(observation)

        if self.random_goal:
            observation = self.render_goal(goal, out=observation)

        return observation[:, :, np.newaxis]  # Convert to single channel image


class MultiChannelObservationGenerator(ObservationGenerator):
    def __init__(
        self,
        maze: np.ndarray,
        random_goal: bool,
        goal_range: int,
        noise: float = 0.0,
        noise_type: str = "gauss",
        static_noise: float = 0.0,
        static_noise_type: str = "s&p",
        restrict_noise: bool = True,
    ):
        super(MultiChannelObservationGenerator, self).__init__(
            random_goal=random_goal,
            goal_range=goal_range,
            noise=noise,
            noise_type=noise_type,
            static_noise=static_noise,
            static_noise_type=static_noise_type,
            restrict_noise=restrict_noise,
        )

        self.n_channels = 3 if random_goal else 2
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*maze.shape, self.n_channels), dtype=np.uint8
        )

    def observation(self, particles: np.ndarray, goal: Tuple[int, int]):
        observation = np.zeros((*self.maze.shape, self.n_channels))
        observation[:, :, 0] = self.render_maze()
        particle_image = self.render_particles(particles, self.maze)
        particle_image = self.generate_noise(particle_image)
        observation[:, :, 1] = particle_image

        if self.random_goal:
            observation[:, :, 2] = self.render_goal(goal)

        return observation
