from typing import Dict, Tuple

import numpy as np
from gym_gathering.steps.base_step_modifier import StepModifier


class SimpleMovementModifier(StepModifier):
    def _step(
        self, action: int, locations: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        dy, dx = self.action_map[action]
        return update + np.array([dy, dx])


class RandomMovementModifier(StepModifier):
    def __init__(
        self,
        action_map: Dict[int, Tuple[int, int]],
        random_move_chance: float = 0.25,
        random_move_distance: int = 1,
        **kwargs
    ):
        super(RandomMovementModifier, self).__init__(action_map, **kwargs)
        self.random_move_chance = random_move_chance
        self.random_move_distance = random_move_distance

    def _step(
        self, action: int, locations: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        if self.random_move_chance > 0.0:
            random_moves = self.np_random.randint(
                -self.random_move_distance,
                self.random_move_distance + 1,
                self.maze.shape + (2,),
            )
            # random_moves_y = self.np_random.randint(0, self.random_move_distance + 1, self.maze.shape)
            mask = self.np_random.choice(
                [0, 1],
                self.maze.shape + (1,),
                p=[1 - self.random_move_chance, self.random_move_chance],
            )
            random_moves = random_moves * mask

            return update + np.squeeze((np.array([random_moves[tuple(locations.T)]])))
        else:
            return update


class FuzzyMovementModifier(StepModifier):
    def __init__(
        self,
        action_map: Dict[int, Tuple[int, int]],
        fuzzy_action_probability: float = 0.25,
        **kwargs
    ):
        super(FuzzyMovementModifier, self).__init__(action_map, **kwargs)
        self.fuzzy_action_probability = fuzzy_action_probability

    def _step(
        self, action: int, locations: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        if self.fuzzy_action_probability == 0.0:
            return update
        else:
            global_mask = self.np_random.choice(
                [0, 1],
                self.maze.shape + (1,),
                p=[self.fuzzy_action_probability, 1 - self.fuzzy_action_probability],
            )
            particle_mask = global_mask[tuple(locations.T)]
            return np.where(particle_mask, update, [0, 0])
