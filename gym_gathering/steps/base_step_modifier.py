from abc import ABC
from typing import Dict, Tuple, Optional

import numpy as np


class StepModifier(ABC):
    def __init__(self, action_map: Dict[int, Tuple[int, int]], **kwargs):
        self.maze = None  # type: Optional[np.ndarray]
        self.freespace = None  # type: Optional[np.ndarray]
        self.action_map = action_map
        self.np_random = np.random.random.__self__
        self.sub_modifiers = []  # type: List[StepModifier]

    def add_sub_modifier(self, modifier):
        self.sub_modifiers.append(modifier)

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random
        for modifier in self.sub_modifiers:
            modifier.seed(np_random)

    def reset(self, locations: np.ndarray, maze: np.ndarray, freespace: np.ndarray):
        self.maze = maze
        self.freespace = freespace

        for modifier in self.sub_modifiers:
            modifier.reset(locations, maze, freespace)

    def _step(
        self, action: int, locations: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        pass

    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        update = self._step(action, locations, np.zeros_like(locations))
        for modifier in self.sub_modifiers:
            update = modifier.step(action, locations, update)
        return update

    def step_done(self, valid_locations: np.ndarray) -> None:
        for modifier in self.sub_modifiers:
            modifier.step_done(valid_locations)
