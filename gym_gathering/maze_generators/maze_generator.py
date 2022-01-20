from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pkg_resources
from gym.utils import seeding


class InstanceGenerator(ABC):
    """
    Base class for instance generators.
    :param width: (int) Width of the instance.
    :param height: (int) Height of the instance.
    """

    def __init__(self, width: int = 100, height: int = 100, seed: int = None) -> None:
        self.width = width
        self.height = height
        self.np_random = None
        self._last = None
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save(self, path) -> None:
        np.savetxt(path, self._last, fmt="%d")

    def last(self) -> np.ndarray:
        return self._last

    def has_next(self) -> bool:
        return True

    @abstractmethod
    def generate(self, success=True) -> np.ndarray:
        pass


class InstanceReader(InstanceGenerator):
    def __init__(self, path: str):
        super(InstanceReader, self).__init__(0, 0, None)
        self.path = path

    def generate(self, success=True) -> np.ndarray:
        if not self._last:
            path = pkg_resources.resource_filename("gym_gathering", self.path)
            self._last = np.loadtxt(path).astype(np.uint8)
            self.height, self.width = self._last.shape
        return self._last

    def has_next(self) -> bool:
        return False


class RandomInstanceReader(InstanceGenerator):
    def __init__(self, paths: List[str], seed: int = None):
        super(RandomInstanceReader, self).__init__(0, 0, seed)
        self.instances = []
        self._read_instances(paths)

    def generate(self, success=True) -> np.ndarray:
        self._last = self.instances[self.np_random.randint(len(self.instances))]
        self.height, self.width = self._last.shape
        return self._last

    def _read_instances(self, paths: List[str]):
        for path in paths:
            p = pkg_resources.resource_filename("gym_maze", path)
            self.instances.append(np.loadtxt(p))
