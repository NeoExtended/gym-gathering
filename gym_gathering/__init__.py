import itertools

from gym.envs.registration import register

from gym_gathering.maze_generators import (
    BufferedRRTGenerator,
    StagesRRTGenerator,
)

mazes = {
    "Corridor": "mapdata/map0318.csv",
    "Capillary": "mapdata/map0518.csv",
    "Brain": "mapdata/map0122.csv",
    "Vessel": "mapdata/small_vessel.csv",
    "Pulmonary": "mapdata/pulmonary.csv",
    "RandomRRT": BufferedRRTGenerator,
    "StagesRRT": StagesRRTGenerator,
}


default_goals = {
    "Corridor": [82, 80],
    "Capillary": [61, 130],
    "Brain": [96, 204],
    "Vessel": [85, 66],
}

goal_range = 10
time_limit = 2000
entry_point = "gym_gathering.envs:MazeBase"

# reward_types = {"Discrete": "goal", "Continuous": "continuous"}
physics_types = {"Algorithmic": "simple", "Fuzzy": "fuzzy", "Physical": "real-world"}
particle_counts = {"FixedPC": 256, "RandomPC": "random", "FilledPC": "filled"}
goal_types = {"FixedGoal": "default", "RandomGoal": "randomgoal"}

for physics_type, particle_count, goal_type in itertools.product(
    physics_types, particle_counts, goal_types
):
    for maze_type in mazes:
        observation_type = "multichannel"
        if goal_type == "FixedGoal" and maze_type in default_goals:
            observation_type = "simple"

        name = f"{maze_type}{physics_type}{particle_count}{goal_type}-v0"
        args = {
            "instance": mazes[maze_type],
            "goal": default_goals.get(maze_type, None)
            if goal_type == "FixedGoal"
            else None,
            "goal_range": goal_range,
            "n_particles": particle_counts[particle_count],
            "step_type": physics_types[physics_type],
            "reward_generator": "continuous",
            "observation_type": observation_type,
        }

        register(
            id=name, entry_point=entry_point, max_episode_steps=time_limit, kwargs=args,
        )


KEYMAP = {
    "up": 6,
    "down": 2,
    "left": 4,
    "right": 0,
}
