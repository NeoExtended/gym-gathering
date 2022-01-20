import math
from collections import deque
from typing import List

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from gym_gathering.maze_generators.maze_generator import InstanceGenerator


class Node(np.ndarray):
    """
    Helper class to store graph information for the RTTGenerator.
    """

    def __new__(cls, shape, dtype=int, buffer=None, offset=0, strides=None, order=None):
        obj = super(Node, cls).__new__(
            cls, shape, dtype, buffer, offset, strides, order
        )
        obj.prev = None
        obj.next = []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.prev = getattr(obj, "prev", np.array([]))
        self.next = getattr(obj, "next", [])


class RRTGenerator(InstanceGenerator):
    """
    This generator uses Rapidly-exploring random trees(RRT) to create random instances with blood vessel like shape.
    https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree

    :param width: (int) Instance width.
    :param height: (int) Instance height.
    :param n_nodes: (int) Number of nodes for the RTT (higher values will result in more vessels).
    :param max_length: (float) Maximum length for a new edge (controls how straight the vessels will be).
    :param n_loops: (int) Number of loops created at the end of the vessels.
    :param thickness: (float) Factor for overall vessel thickness.
    :param border: (bool) Weather or not to set all border pixels to blocked.
    """

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        n_nodes: int = 400,
        max_length: float = 5.0,
        n_loops: int = 10,
        thickness: float = 1.0,
        border: bool = True,
        seed=None,
    ) -> None:
        super(RRTGenerator, self).__init__(width, height, seed=seed)
        self.n_nodes = n_nodes
        self.max_length = max_length
        self.n_loops = n_loops
        self.thickness = thickness
        self.border = border

        if self.border:
            self.width -= 2
            self.height -= 2

    def generate(self, success=True) -> np.ndarray:
        nodes = self._generate_tree()
        self._calculate_flow(nodes)
        self._create_loops(nodes)
        maze = self._draw_graph(nodes)
        self._last = self._create_border(maze).astype(np.uint8)
        return self._last

    def _generate_tree(self) -> List[Node]:
        nodes = []
        nodes.append(
            np.array(
                [
                    self.np_random.randint(self.height),
                    self.np_random.randint(self.width),
                ]
            ).view(Node)
        )
        # nodes.append(np.array([imgy/2, imgx/2], dtype=int).view(Node))

        for i in range(self.n_nodes):
            rand = [
                self.np_random.randint(self.height),
                self.np_random.randint(self.width),
            ]
            dist = cdist(np.atleast_2d(rand), np.atleast_2d(nodes))
            min_dist = np.min(dist)
            min_node_index = np.argmin(dist)
            min_node = nodes[min_node_index]

            if min_dist < self.max_length:
                new_node = np.array(rand).view(Node)
            else:
                theta = math.atan2(rand[1] - min_node[1], rand[0] - min_node[0])
                new_node = np.array(
                    [
                        min_node[0] + self.max_length * math.cos(theta),
                        min_node[1] + self.max_length * math.sin(theta),
                    ],
                    dtype=int,
                ).view(Node)
            new_node.prev = min_node
            min_node.next.append(new_node)
            nodes.append(new_node)

        return nodes

    def _calculate_flow(self, nodes: List[Node]) -> None:
        end_nodes = []
        for node in nodes:
            if len(node.next) == 0:
                end_nodes.append(node)
            node.flow = 0

        current_nodes = deque()
        current_nodes.extend(end_nodes)
        while current_nodes:
            node = current_nodes.pop()
            if node.prev.size > 0:
                node.prev.flow += 1
                current_nodes.append(node.prev)

    def _create_loops(self, nodes: List[Node]) -> None:
        end_nodes = []
        for node in nodes:
            if len(node.next) == 0:
                end_nodes.append(node)

        for i in range(self.n_loops):
            rand = [
                self.np_random.randint(self.height),
                self.np_random.randint(self.width),
            ]
            dist = cdist(np.atleast_2d(rand), np.atleast_2d(end_nodes))
            first_min = end_nodes[np.argmin(dist)]
            second_min = end_nodes[np.argpartition(dist, 2)[0][2]]

            first_min.next.append(second_min)
            second_min.next.append(first_min)

    def _draw_graph(self, nodes: List[Node]) -> np.ndarray:
        maze = np.zeros((self.height, self.width))

        for node in nodes:
            for next_node in node.next:
                cv2.line(
                    maze,
                    (node[1], node[0]),
                    (next_node[1], next_node[0]),
                    (1),
                    int(max(2.0, np.sqrt(next_node.flow)) * self.thickness),
                )
        # structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # maze = cv2.morphologyEx(maze, cv2.MORPH_CLOSE, structure)
        return maze

    def _create_border(self, maze: np.ndarray) -> np.ndarray:
        if self.border:
            return np.pad(maze, pad_width=1, mode="constant", constant_values=0)
        else:
            return maze


class BufferedRRTGenerator(RRTGenerator):
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        n_nodes: int = 400,
        max_length: float = 5.0,
        n_loops: int = 10,
        thickness: float = 1.0,
        border: bool = True,
        buffer_size: int = 100,
        generation_chance: float = 0.05,
        pre_generate: int = 8,
        seed=None,
    ):
        super(BufferedRRTGenerator, self).__init__(
            width, height, n_nodes, max_length, n_loops, thickness, border, seed=seed
        )

        self.buffer = deque(maxlen=buffer_size)
        self.generation_chance = generation_chance

        for i in range(pre_generate):
            self._generate()

    def generate(self, success=True) -> np.ndarray:
        if self.np_random.rand() < self.generation_chance:
            self._generate()
        return self.buffer[self.np_random.randint(len(self.buffer))]

    def _generate(self):
        last = self._last
        self.buffer.append(super().generate())
        self._last = last


class StagesRRTGenerator(RRTGenerator):
    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        n_nodes: int = 400,
        max_length: float = 5.0,
        n_loops: int = 10,
        thickness: float = 1.0,
        border: bool = True,
        max_stages: int = 100,
        seed=None,
    ):
        super(StagesRRTGenerator, self).__init__(
            width, height, n_nodes, max_length, n_loops, thickness, border, seed=seed
        )
        self.max_stages = max_stages
        self.stages = []
        self.current_stage = 0
        self._generate()

    def generate(self, success=True) -> np.ndarray:
        if success:
            self.current_stage += 1
        else:
            self.current_stage = 0
        if self.current_stage >= len(self.stages):
            self._generate()

        return self.stages[self.current_stage]

    def _generate(self):
        self.stages.append(super().generate())
