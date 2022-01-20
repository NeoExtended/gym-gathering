import numpy as np

from gym_gathering.rewards.base_reward_generator import (
    StepInformationProvider,
)
from gym_gathering.rewards.episode_ends import (
    DynamicEpisodeEnd,
    GoalReachedEpisodeEnd,
    GatheringEpisodeEnd,
)
from gym_gathering.rewards.reward_generators import (
    ContinuousTotalCostReward,
    TimePenaltyReward,
    DiscreteTotalCostReward,
    DiscreteMaxCostReward,
    GatheringReward,
    ContinuousMaxCostReward,
)


def ContinuousRewardGenerator(
    maze,
    goal,
    goal_range,
    n_particles,
    action_map,
    relative=False,
    positive_only=False,
    time_penalty=True,
    dynamic_episode_length=False,
    normalize=True,
    gathering_reward=0.0,
    max_cost_reward=True,
    total_cost_reward=True,
):
    assert (
        max_cost_reward or total_cost_reward
    ), "One of the main reward components must be enabled!"
    information = StepInformationProvider(
        maze, goal, goal_range, n_particles, action_map, relative
    )
    if max_cost_reward and total_cost_reward:
        generator = ContinuousTotalCostReward(
            information, positive_only=positive_only, normalize=normalize
        )
        generator.add_sub_generator(
            ContinuousMaxCostReward(
                information, positive_only=positive_only, normalize=normalize
            )
        )
    elif max_cost_reward and not total_cost_reward:
        generator = ContinuousMaxCostReward(
            information, positive_only=positive_only, normalize=normalize
        )
    elif total_cost_reward and not max_cost_reward:
        generator = ContinuousTotalCostReward(
            information, positive_only=positive_only, normalize=normalize
        )

    if gathering_reward > 0.0:
        generator.add_sub_generator(
            GatheringReward(scale=gathering_reward, normalize=normalize)
        )

    if time_penalty:
        if normalize:
            generator.add_sub_generator(TimePenaltyReward())
        else:
            generator.add_sub_generator(
                TimePenaltyReward(
                    scale=np.ma.masked_equal(information.costmap, 0).mean()
                )
            )

    if dynamic_episode_length:
        generator.add_sub_generator(
            DynamicEpisodeEnd(normalize=normalize, continuous=True)
        )

    generator.add_sub_generator(GoalReachedEpisodeEnd())

    return generator


def GoalRewardGenerator(
    maze,
    goal,
    goal_range,
    n_particles,
    action_map,
    relative=False,
    n_subgoals=None,
    final_reward=100,
    min_reward=2,
    max_reward=4,
    time_penalty=True,
    dynamic_episode_length=False,
    max_cost_reward=True,
    total_cost_reward=True,
    gathering_reward=0.0,
):
    assert (
        max_cost_reward or total_cost_reward
    ), "One of the main reward components must be enabled!"
    information = StepInformationProvider(
        maze, goal, goal_range, n_particles, action_map, relative
    )
    test = information.convex_corners

    if max_cost_reward and total_cost_reward:
        generator = DiscreteTotalCostReward(
            information,
            n_subgoals=n_subgoals,
            min_reward=min_reward,
            max_reward=max_reward,
        )
        generator.add_sub_generator(
            DiscreteMaxCostReward(
                n_subgoals=n_subgoals, min_reward=min_reward, max_reward=max_reward
            )
        )
    elif max_cost_reward and not total_cost_reward:
        generator = DiscreteMaxCostReward(
            information,
            n_subgoals=n_subgoals,
            min_reward=min_reward,
            max_reward=max_reward,
        )
    elif total_cost_reward and not max_cost_reward:
        generator = DiscreteTotalCostReward(
            information,
            n_subgoals=n_subgoals,
            min_reward=min_reward,
            max_reward=max_reward,
        )

    if gathering_reward > 0.0:
        generator.add_sub_generator(
            GatheringReward(scale=gathering_reward, normalize=False)
        )

    if time_penalty:
        generator.add_sub_generator(
            TimePenaltyReward(scale=2 * np.sum(generator.reward_scale))
        )

    if dynamic_episode_length:
        generator.add_sub_generator(DynamicEpisodeEnd())

    generator.add_sub_generator(GoalReachedEpisodeEnd(end_reward=final_reward))

    return generator


def GatheringRewardGenerator(
    maze,
    goal,
    goal_range,
    n_particles,
    action_map,
    relative=False,
    time_penalty=True,
    normalize=True,
):
    information = StepInformationProvider(
        maze, goal, goal_range, n_particles, action_map, relative
    )
    generator = GatheringReward(information, normalize=normalize)

    if time_penalty:
        generator.add_sub_generator(TimePenaltyReward())

    generator.add_sub_generator(GatheringEpisodeEnd())
    return generator


REWARD_GENERATORS = {
    "goal": GoalRewardGenerator,
    "continuous": ContinuousRewardGenerator,
    "gathering": GatheringRewardGenerator,
}
