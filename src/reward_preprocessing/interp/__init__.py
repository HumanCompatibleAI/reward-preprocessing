"""Package containing Sacred Ingredients for interpreting reward models.
Each Ingredient should contain at least one capture function which receives
a RewardModel (and if necessary other input, such as episode rollouts or single
transitions) and produces some artifacts or returns values to be used for
interpretation. This can include visualizations, but also returning an
equivalent but easier to interpret reward model.

The reason to implement these as Sacred ingredients rather than just simple
functions is that each method may have parameters (and we want Sacred's mechanisms
to deal with those) and that they may produce artifacts (which should happen
directly here, rather than in the main script.

Each Ingredient should have a boolean 'enabled' config option.
If this is set to False, then the ingredient should have no
effect. This means that all ingredients can be included in
the interpret.py script and the user can decide which ones to
use via CLI arguments.
"""

from .noise import add_noise_potential, noise_ingredient
from .visualize_rollout import rollout_ingredient, visualize_rollout

__all__ = [
    "rollout_ingredient",
    "visualize_rollout",
    "noise_ingredient",
    "add_noise_potential",
]
