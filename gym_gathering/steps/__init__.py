from gym_gathering.steps.base_step_modifier import StepModifier
from gym_gathering.steps.basic_modifiers import (
    SimpleMovementModifier,
    FuzzyMovementModifier,
    RandomMovementModifier,
)
from gym_gathering.steps.physical_modifier import PhysicalMovementModifier


def ActionErrorMovement(action_map, **kwargs):
    base = SimpleMovementModifier(action_map=action_map, **kwargs)
    base.add_sub_modifier(RandomMovementModifier(action_map=action_map, **kwargs))
    base.add_sub_modifier(FuzzyMovementModifier(action_map=action_map, **kwargs))
    return base


STEP_MODIFIERS = {
    "simple": SimpleMovementModifier,
    "fuzzy": ActionErrorMovement,
    "real-world": PhysicalMovementModifier,
}
