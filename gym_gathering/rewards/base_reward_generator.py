import collections
from abc import ABC
from typing import Tuple, List, Dict

import numpy as np


class StepInformationProvider(ABC):
    """
    This class calculates certain values which are used frequently in reward generators.
    A single instance of this class can be shared between a set of (sub)generators
    to prevent multiple calculations of costly intermediate results.
    :param maze: (np.ndarray) two dimensional array defining the maze where 0 indicates passable terrain and 1 indicates an obstacle
    :param goal: (list) A point coordinate in form [x, y] ([column, row]) defining the goal.
    :param goal_range: (int) Range around the goal position which should be treated as 'goal reached'.
    :param n_particles: (int) Total number of robots.
    :param action_map: (dict) Map containing allowed actions.
    """

    def __init__(
        self,
        maze: np.ndarray,
        goal: Tuple[int, int],
        goal_range: int,
        n_particles: int,
        action_map: Dict[int, Tuple[int, int]],
        relative: bool = False,
    ):
        self.goal_range = goal_range
        self.n_particles = n_particles
        self.initial_robot_locations = None
        self.action_map = action_map
        self.maze = maze
        self.goal = goal
        self.relative = relative

        self.last_locations = None
        self.last_action = None

        self._cost = None
        self._max_start_cost = None
        self._max_cost = None
        self._particle_cost = None
        self._total_start_cost = None
        self._total_cost = None
        self._unique_particles = None
        self._done = False
        self._step_reward = 0.0
        self._mean_cost = None
        self._ep_len_estimate = None
        self._convex_corners = None

    def reset(self, locations):
        self.initial_robot_locations = np.copy(locations)
        self.last_locations = locations
        self._done = False

        if self.relative:
            self._ep_len_estimate = None
            self._total_start_cost = None
            self._max_start_cost = None

    def step(self, action, locations):
        self._step_reward = 0.0
        self.last_locations = locations
        self.last_action = action
        self._step_reset()

    def stepped_generator(self, done, reward):
        self._step_reward += reward
        if done:
            self._done = True

    def _step_reset(self):
        self._max_cost = None
        self._particle_cost = None
        self._total_cost = None
        self._unique_particles = None

    def _calculate_cost_map(self, maze, goal) -> np.ndarray:
        """
        Calculates the cost map based on a given goal position via bfs
        """
        queue = collections.deque([goal])  # [x, y] pairs in point notation order!
        seen = np.zeros(maze.shape, dtype=int)
        seen[goal[1], goal[0]] = 1
        cost = np.zeros(maze.shape, dtype=int)
        height, width = maze.shape

        while queue:
            x, y = queue.popleft()
            for action in self.action_map.values():
                x2, y2 = x + action[1], y + action[0]
                if (
                    0 <= x2 < width
                    and 0 <= y2 < height
                    and maze[y2, x2] != 1
                    and seen[y2, x2] != 1
                ):
                    queue.append([x2, y2])
                    seen[y2, x2] = 1
                    cost[y2, x2] = cost[y, x] + 1
        return cost

    def _count_convex_corners(self) -> Tuple[int, int, int, int]:
        """
        Calculates the number of convex corners
        :return: (Tuple[int, int, int, int]) Tuple containing the number of convex corners for nw, ne, sw and se convex corners.
        """
        nw = ne = sw = se = 0
        for ix, iy in np.ndindex(self.maze.shape):
            if self.maze[ix, iy] == 0:
                if self.maze[ix + 1, iy] == 1 and self.maze[ix, iy + 1] == 1:
                    sw += 1
                elif self.maze[ix + 1, iy] == 1 and self.maze[ix, iy - 1] == 1:
                    se += 1
                elif self.maze[ix - 1, iy] == 1 and self.maze[ix, iy + 1] == 1:
                    nw += 1
                elif self.maze[ix - 1, iy] == 1 and self.maze[ix, iy - 1] == 1:
                    ne += 1
        return (nw, ne, sw, se)

    def _count_freespace(self):
        free = 0
        idxes = np.argwhere(self.maze == 0)
        for iy, ix in idxes:
            if (self.maze[iy - 1 : iy + 1, ix - 1 : ix + 1] == 0).all():
                free += 1
        return free

    def set_particle_count(self, n_particles):
        self.n_particles = n_particles
        self._total_start_cost = None

    @property
    def convex_corners(self) -> Tuple[int, int, int, int]:
        if self._convex_corners is None:
            self._convex_corners = self._count_convex_corners()
        return self._convex_corners

    @property
    def costmap(self) -> np.ndarray:
        if self._cost is None:
            self._cost = self._calculate_cost_map(self.maze, self.goal)
        return self._cost

    @property
    def max_start_cost(self) -> float:
        if self._max_start_cost is None:
            if self.relative:
                self._max_start_cost = np.max(self.particle_cost)
            else:
                self._max_start_cost = np.max(self.costmap)
        return self._max_start_cost

    @property
    def particle_cost(self) -> np.ndarray:
        if self._particle_cost is None:
            self._particle_cost = self.costmap.ravel()[
                (
                    self.last_locations[:, 1]
                    + self.last_locations[:, 0] * self.costmap.shape[1]
                )
            ]
        return self._particle_cost

    @property
    def episode_length_estimate(self) -> int:
        if self._ep_len_estimate is None:
            points = np.argwhere(self.maze == 0)
            extremes = np.argmax(points, axis=0)
            if np.sum(points[extremes[0]]) > np.sum(points[extremes[1]]):
                extreme = points[extremes[0]]
            else:
                extreme = points[extremes[1]]
            costmap = self._calculate_cost_map(self.maze, (extreme[1], extreme[0]))
            self._ep_len_estimate = int(
                0.75
                * np.max(costmap)
                * np.log(self.mean_cost * np.min(self.convex_corners))
            )
        return self._ep_len_estimate

    @property
    def mean_cost(self) -> int:
        if self._mean_cost is None:
            self._mean_cost = np.ma.masked_equal(self.costmap, 0).mean()
        return self._mean_cost

    @property
    def total_start_cost(self) -> float:
        if self._total_start_cost is None:
            if self.relative:
                self._total_start_cost = np.sum(self.particle_cost)
            else:
                self._total_start_cost = self.mean_cost * self.n_particles
        return self._total_start_cost

    @property
    def total_cost(self) -> float:
        if self._total_cost is None:
            self._total_cost = np.sum(self.particle_cost)
        return self._total_cost

    @property
    def max_cost(self) -> float:
        if self._max_cost is None:
            self._max_cost = np.max(self.particle_cost)
        return self._max_cost

    @property
    def unique_particles(self) -> np.ndarray:
        if self._unique_particles is None:
            self._unique_particles = np.unique(self.last_locations, axis=0)
        return self._unique_particles

    @property
    def is_relative(self):
        return self.relative

    @property
    def is_done(self):
        return self._done

    @property
    def step_reward(self):
        return self._step_reward


class RewardGenerator(ABC):
    """
    Base Class for reward generators for the maze environments
    """

    def __init__(
        self, information_provider: StepInformationProvider = None, scale: float = 1.0
    ):
        self.calculator = None
        self.generators = []  # type: List[RewardGenerator]
        self.scale = scale

        if information_provider:
            self.set_information_provider(information_provider)

    def set_information_provider(self, calculator: StepInformationProvider):
        self.calculator = calculator
        for generator in self.generators:
            generator.set_information_provider(calculator)

    def set_particle_count(self, n_particles):
        self.calculator.set_particle_count(n_particles=n_particles)

    def add_sub_generator(self, generator):
        generator.set_information_provider(self.calculator)
        self.generators.append(generator)

    def reset(self, locations):
        self.calculator.reset(locations)
        self._reset(locations)
        self._reset_generators(locations)

    def _reset_generators(self, locations):
        for generator in self.generators:
            generator._reset(locations)

    def _reset(self, locations):
        pass

    def step(self, action, locations) -> Tuple[bool, float]:
        self.calculator.step(action, locations)
        self.calculator.stepped_generator(*self._step(action, locations))
        self._step_generators(action, locations)
        if self.calculator.is_done:
            end_reward = self._on_done()
            self.calculator.stepped_generator(False, end_reward)
            self._on_done_generators()

        return self.calculator.is_done, self.calculator.step_reward

    def _step(self, action, locations) -> Tuple[bool, float]:
        return False, 0.0

    def _on_done(self) -> float:
        return 0.0

    def _step_generators(self, action, locations) -> None:
        for generator in self.generators:
            self.calculator.stepped_generator(*generator._step(action, locations))

    def _on_done_generators(self) -> None:
        for generator in self.generators:
            end_reward = generator._on_done()
            self.calculator.stepped_generator(False, end_reward)
