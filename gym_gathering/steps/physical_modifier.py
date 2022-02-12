import logging
from typing import Dict, Tuple

import numpy as np

from gym_gathering.steps.base_step_modifier import StepModifier


class PhysicalMovementModifier(StepModifier):
    def __init__(
        self,
        action_map: Dict[int, Tuple[int, int]],
        random_particle_weights: bool = True,
        max_particle_weight: float = 1.2,
        force: float = 0.05,
        drag: float = 1.025,
        particle_self_collision: bool = True,
        max_particles_per_cell: int = 3,
        initially_frozen: float = 0.0,
        frozen_state_change_prob: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super(PhysicalMovementModifier, self).__init__(action_map, **kwargs)

        self.random_particle_weights = random_particle_weights
        self.max_particle_weight = max_particle_weight
        self.particle_speed = None
        self.exact_locations = None
        self.force = force
        self.drag = drag
        self.acceleration = None

        self.max_particles_per_cell = max_particles_per_cell
        self.use_self_collision = particle_self_collision
        self.collisions = np.ndarray([])

        self.initially_frozen = initially_frozen
        self.frozen_state_change_prob = frozen_state_change_prob
        self.frozen = np.ndarray([])

    def reset(self, locations: np.ndarray, maze: np.ndarray, freespace: np.ndarray):
        super(PhysicalMovementModifier, self).reset(locations, maze, freespace)

        n_particles = len(locations)
        self.frozen = self.np_random.uniform(size=n_particles) <= self.initially_frozen

        self.particle_speed = np.zeros((n_particles, 2))
        self.exact_locations = np.copy(locations)
        if self.random_particle_weights:
            self.particle_weight = (
                self.max_particle_weight - 1
            ) * self.np_random.random(size=(n_particles,)) + 1
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

        # Calculate exact positions
        exact_locations = self.exact_locations + self.particle_speed
        rounded_locations = np.rint(exact_locations).astype(int)

        # Particles might move faster than a single pixel per step so clip the position to the border.
        # Every maze is required to have at least one non-free pixel of border - clipping results in a collision.
        rounded_locations[:, 0] = np.clip(
            rounded_locations[:, 0], 0, self.freespace.shape[0] - 1
        )
        rounded_locations[:, 1] = np.clip(
            rounded_locations[:, 1], 0, self.freespace.shape[1] - 1
        )

        # Calculate integer update
        rounded_update = rounded_locations - locations

        if self.use_self_collision:
            valid_locations = self.self_collision(locations, rounded_locations)
            rounded_update = np.where(
                valid_locations, rounded_update, np.zeros_like(rounded_update)
            )

        return rounded_update + update

    def check_collision(self, free: np.ndarray, locations: np.ndarray):
        valid_locations = free.ravel()[
            (locations[:, 1] + locations[:, 0] * free.shape[1])
        ]
        return valid_locations[:, np.newaxis]

    def self_collision(self, locations: np.ndarray, future_locations: np.ndarray):
        # Initial collisions are wall collisions
        collisions = self.check_collision(self.freespace, future_locations)
        current_collisions = len(collisions) - collisions.sum()
        prev_collisions = -1

        # Calculate collisions by iteratively resetting particles which cannot
        # enter their target pixel to their original position.
        while current_collisions > prev_collisions:
            prev_collisions = current_collisions

            real_positions = np.where(collisions, future_locations, locations)
            unique, counts = np.unique(real_positions, return_counts=True, axis=0)

            current_distribution = np.copy(self.maze) * (
                self.max_particles_per_cell + 1
            )
            current_distribution[unique[:, 0], unique[:, 1]] = counts
            current_distribution[
                current_distribution <= self.max_particles_per_cell
            ] = 1
            current_distribution[current_distribution > self.max_particles_per_cell] = 0

            new_collisions = self.check_collision(current_distribution, real_positions)

            collisions = np.min(
                np.concatenate([collisions, new_collisions], axis=1,), axis=1,
            )[:, np.newaxis]
            current_collisions = len(collisions) - collisions.sum()

        # Particles which do not change their pixel should not have a collision.
        # This prevents particles which enter a pixel and cause a collision
        # from stopping particles (by marking them as having a collision) which
        # have speed that is not sufficient to change the pixel
        change = (
            np.rint(self.exact_locations + self.particle_speed).astype(int) - locations
        )
        collisions = np.where(
            np.sum(np.abs(change), axis=1)[:, np.newaxis] > 0, collisions, 1
        )

        self.collisions = collisions
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
        # self.particle_speed = np.clip(self.particle_speed, 0.0, 1.0)
        self.particle_speed = np.where(
            valid_locations, self.particle_speed, self.particle_speed / 4
        )  # collision

        # stop really slow particles.
        fast_particles = np.abs(self.particle_speed) > 0.01
        self.particle_speed = np.where(fast_particles, self.particle_speed, 0)
        self.exact_locations = np.where(
            fast_particles, self.exact_locations, np.rint(self.exact_locations)
        )
