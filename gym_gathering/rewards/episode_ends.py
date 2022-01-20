from abc import ABC
from typing import Tuple

import numpy as np

from gym_gathering.rewards.base_reward_generator import (
    RewardGenerator,
    StepInformationProvider,
)


class EpisodeEndRewardGenerator(RewardGenerator, ABC):
    def __init__(
        self,
        information_provider: StepInformationProvider = None,
        scale: float = 1.0,
        end_reward: float = 0.0,
    ):
        super().__init__(information_provider, scale)

        self.end_reward = end_reward


class GatheringEpisodeEnd(EpisodeEndRewardGenerator):
    def _step(self, action, locations) -> Tuple[bool, float]:
        if len(self.calculator.unique_particles) == 1:
            return True, self.end_reward
        return False, 0.0


class DynamicEpisodeEnd(EpisodeEndRewardGenerator):
    def __init__(
        self,
        information_provider: StepInformationProvider = None,
        scale: float = 1.0,
        end_reward: float = 0.0,
        normalize=False,
        continuous=False,
    ):
        super().__init__(information_provider, scale, end_reward)
        self.continuous = continuous
        self.tier = 0
        self.dynamic_moves = None
        self.moves_left = 0
        self.normalize = normalize
        self.normalization = 1.0

    def _reset(self, locations):
        if self.continuous:
            n_subgoals = self.calculator.episode_length_estimate
        else:
            n_subgoals = int(self.calculator.max_start_cost / 2)
        avg_moves_additional_moves = int(
            self.calculator.episode_length_estimate * 2.0 / n_subgoals
        )
        self.dynamic_moves = np.flip(
            np.rint(
                np.linspace(1, avg_moves_additional_moves * 2 - 1, n_subgoals * 2 + 1)
            )
        )
        self.moves_left = self.dynamic_moves[0]
        self.tier = 1

        if self.normalize:
            self.normalization = np.sum(self.dynamic_moves)

    def _step(self, action, locations) -> Tuple[bool, float]:
        if self.calculator.step_reward > 0 and self.tier < len(self.dynamic_moves):
            self.moves_left += self.dynamic_moves[self.tier]
            self.tier += 1

        self.moves_left -= 1

        if self.moves_left <= 0:
            return True, 0.0
        return False, 0.0

    def _on_done(self) -> float:
        reward = float(self.moves_left)

        if self.calculator.max_cost > self.calculator.goal_range:
            reward -= self.dynamic_moves[self.tier + 1 :].sum()

        return reward / self.normalization * self.scale


class GoalReachedEpisodeEnd(EpisodeEndRewardGenerator):
    def _step(self, action, locations) -> Tuple[bool, float]:
        if self.calculator.max_cost <= self.calculator.goal_range:
            return True, 0.0
        return False, 0.0
