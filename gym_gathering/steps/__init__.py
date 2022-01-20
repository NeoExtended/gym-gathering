from gym_gathering.steps.base_step_modifier import StepModifier
from gym_gathering.steps.basic_modifiers import (
    SimpleMovementModifier,
    FuzzyMovementModifier,
    RandomMovementModifier,
)
from gym_gathering.steps.physical_modifier import PhysicalMovementModifier


def ActionErrorMovement(**kwargs):
    base = SimpleMovementModifier(**kwargs)
    base.add_sub_modifier(RandomMovementModifier(**kwargs))
    base.add_sub_modifier(FuzzyMovementModifier(**kwargs))
    return base


STEP_MODIFIERS = {
    "simple": SimpleMovementModifier,
    "fuzzy": ActionErrorMovement,
    "real-world": PhysicalMovementModifier,
}
