from gym_gathering.observations.base_observation_generator import (
    ObservationGenerator,
)
from gym_gathering.observations.basic_generators import (
    SingleChannelObservationGenerator,
    MultiChannelObservationGenerator,
)
from gym_gathering.observations.real_world import (
    SingleChannelRealWorldObservationGenerator,
)

OBS_GENERATORS = {
    "simple": SingleChannelObservationGenerator,
    "multichannel": MultiChannelObservationGenerator,
    "real-world": SingleChannelRealWorldObservationGenerator,
}
