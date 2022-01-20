import numpy as np
from typing import Dict, Tuple

from gym_gathering.steps.base_step_modifier import StepModifier


class PhysicalMovementModifier(StepModifier):
    def __init__(
        self,
        action_map: Dict[int, Tuple[int, int]],
        random_weights: bool = True,
        force: float = 0.05,
        drag: float = 1.025,
        particle_self_collision: bool = True,
        max_particles_per_cell: int = 3,
        collision_range: int = 3,
        initially_frozen: float = 0.0,
        frozen_state_change_prob: Tuple[float, float] = (0.0, 0.0),
        **kwargs
    ):
        super(PhysicalMovementModifier, self).__init__(action_map, **kwargs)

        self.random_particle_weights = random_weights
        self.particle_speed = None
        self.exact_locations = None
        self.force = force
        self.drag = drag
        self.acceleration = None

        self.max_particles_per_cell = max_particles_per_cell
        self.use_self_collision = particle_self_collision
        self.collisions = np.ndarray([])
        self.collision_range = collision_range

        self.initially_frozen = initially_frozen
        self.frozen_state_change_prob = frozen_state_change_prob
        self.frozen = np.ndarray([])

    def reset(self, locations: np.ndarray, maze: np.ndarray, freespace: np.ndarray):
        super(PhysicalMovementModifier, self).reset(locations, maze, freespace)
        self.exact_locations = np.copy(locations)

        n_particles = len(locations)
        self.frozen = self.np_random.uniform(size=n_particles) <= self.initially_frozen

        self.particle_speed = np.zeros((n_particles, 2))
        self.exact_locations = np.zeros((n_particles, 2))
        if self.random_particle_weights:
            self.particle_weight = self.np_random.randint(1, 2, size=(n_particles,))
        else:
            self.particle_weight = np.full((n_particles,), 1)

        # self.acceleration = 0.25
        self.acceleration = self.force / self.particle_weight

    def _step(
        self, action: int, locations: np.ndarray, update: np.ndarray
    ) -> np.ndarray:
        dy, dx = self.action_map[action]

        self.particle_speed = self.particle_speed + (
            np.stack([dy * self.acceleration, dx * self.acceleration], axis=1)
        )

        if self.frozen_state_change_prob[0] > 0 or self.frozen_state_change_prob[1] > 0:
            new_frozen = (
                self.np_random.uniform(size=len(locations))
                <= self.frozen_state_change_prob[0]
            )
            unfrozen = (
                self.np_random.uniform(size=len(locations))
                <= self.frozen_state_change_prob[1]
            )

            currently_frozen = self.frozen | new_frozen
            self.frozen = np.logical_xor(
                currently_frozen, unfrozen, out=currently_frozen, where=currently_frozen
            )

        # Stop the movement of frozen particles
        self.particle_speed *= ~self.frozen[:, np.newaxis]

        # Calculate integer update
        rounded_update = np.rint(self.particle_speed).astype(int)

        if self.use_self_collision:
            valid_locations = self.self_collision(locations)
            rounded_update = np.where(
                valid_locations, rounded_update, np.zeros_like(rounded_update)
            )

        return rounded_update + update

    def check_collision(self, free: np.ndarray, locations: np.ndarray):
        valid_locations = free.ravel()[
            (locations[:, 1] + locations[:, 0] * free.shape[1])
        ]
        return valid_locations[:, np.newaxis]

    def self_collision(self, locations: np.ndarray):
        # Calculate normed particle direction vectors
        norm = np.linalg.norm(self.particle_speed, axis=1)
        # Prevent division by zero on zero length vectors
        norm[norm == 0.0] = 1.0

        directions = self.particle_speed / norm[:, np.newaxis]

        # Calculate maze positions that are blocked by walls or other particles
        unique, counts = np.unique(locations, return_counts=True, axis=0)
        current_distribution = np.copy(self.maze) * self.max_particles_per_cell
        current_distribution[unique[:, 0], unique[:, 1]] = counts
        current_distribution[current_distribution < self.max_particles_per_cell] = 1
        current_distribution[current_distribution >= self.max_particles_per_cell] = 0

        # Calculate collisions. Future collisions may depend on the future movement of other particles.
        # Calculate self.collision_range steps ahead
        collisions = [
            self.check_collision(
                current_distribution, locations + np.rint(directions * i).astype(int)
            )
            for i in range(1, self.collision_range)
        ]

        self.collisions = np.max(np.concatenate(collisions, axis=1), axis=1,)[
            :, np.newaxis
        ]
        return self.collisions

    def step_done(self, valid_locations):
        super(PhysicalMovementModifier, self).step_done(valid_locations)
        if self.use_self_collision:

            valid_locations = np.min(
                np.concatenate([valid_locations, self.collisions], axis=1), axis=1
            )[:, np.newaxis]

        self.exact_locations = np.where(
            valid_locations,
            self.exact_locations + self.particle_speed,
            self.exact_locations,
        )
        self.particle_speed = self.particle_speed / self.drag
        self.particle_speed = np.where(
            valid_locations, self.particle_speed, self.particle_speed / 2
        )  # collision
        self.particle_speed = np.where(
            np.abs(self.particle_speed) > 0.01, self.particle_speed, 0
        )  # stop really slow particles.
