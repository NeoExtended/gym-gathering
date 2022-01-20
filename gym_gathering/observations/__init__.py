from gym_gathering.observations.base_observation_generator import (
    ObservationGenerator,
)
from gym_gathering.observations.real_world import (
    SingleChannelRealWorldObservationGenerator,
)
from gym_gathering.observations.basic_generators import (
    SingleChannelObservationGenerator,
    MultiChannelObservationGenerator,
)


OBS_GENERATORS = {
    "simple": SingleChannelObservationGenerator,
    "multichannel": MultiChannelObservationGenerator,
    "real-world": SingleChannelRealWorldObservationGenerator,
}
