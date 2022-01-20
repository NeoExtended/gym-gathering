import logging

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
    VecEnv,
)


class VecHardGoalSampleWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, alpha: float = 0.65):
        super(VecHardGoalSampleWrapper, self).__init__(venv)

        self.episode_returns = None
        self.episode_lengths = None
        self.goal_locations = venv.get_attr("locations", 0)[0]
        maze = venv.get_attr("maze", 0)[0]
        self.avg_rewards = np.zeros(maze.shape)
        self.goal_sampled = np.zeros(maze.shape)
        self.avg_rewards[self.goal_locations[:, 0], self.goal_locations[:, 1]] = np.inf
        # self.goal_sampled[self.goal_locations[:, 0], self.goal_locations[:, 1]] = 0

        self.alpha = alpha
        self.all_goals_sampled = False

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                goal = self.venv.get_attr("goal", i)[0]
                if self.avg_rewards[goal[1], goal[0]] != np.inf:
                    self.avg_rewards[goal[1], goal[0]] = (
                        self.alpha * self.avg_rewards[goal[1], goal[0]]
                        + (1 - self.alpha) * episode_return
                    )
                else:
                    self.avg_rewards[goal[1], goal[0]] = episode_return

                if not self.all_goals_sampled:
                    self.goal_sampled[goal[1], goal[0]] = 1
                    logging.info(np.sum(self.goal_sampled))
                    if np.sum(self.goal_sampled) == len(self.goal_locations):
                        logging.info("All goals have been sampled at least once.")
                        self.all_goals_sampled = True
                        self.update_probs_from_reward()
                    else:
                        self.update_probs_from_unvisited()
                else:
                    self.update_probs_from_reward()

                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return obs, rewards, dones, new_infos

    def update_probs_from_unvisited(self):
        not_visited = np.zeros(self.avg_rewards.shape)
        not_visited[self.goal_locations[:, 0], self.goal_locations[:, 1]] = 1
        not_visited = not_visited - self.goal_sampled

        prob_positions = not_visited[
            self.goal_locations[:, 0], self.goal_locations[:, 1]
        ]
        probs = prob_positions / np.sum(prob_positions)
        self.venv.env_method("update_goal_probs", probs)

    def update_probs_from_reward(self):
        rew_per_goal = self.avg_rewards[
            self.goal_locations[:, 0], self.goal_locations[:, 1]
        ]
        rew_per_goal = rew_per_goal * -1
        rew_per_goal = rew_per_goal + np.min(rew_per_goal) * -1 + 0.01
        probs = rew_per_goal / np.sum(rew_per_goal)
        self.venv.env_method("update_goal_probs", probs)

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
