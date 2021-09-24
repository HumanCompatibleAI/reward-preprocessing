"""Package containing Sacred Ingredients for interpreting reward models.
Each Ingredient should contain at least one capture function which receives
a RewardNet (and if necessary other input, such as episode rollouts or single
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

from .fixed_processor import add_fixed_potential, fixed_ingredient
from .noise import add_noise_potential, noise_ingredient
from .optimize import optimize, optimize_ingredient
from .optimize_tabular import optimize_tabular, optimize_tabular_ex
from .plot_heatmaps import heatmap_ingredient, plot_heatmaps
from .plot_rewards import plot_rewards, reward_ingredient
from .value_net_shaping import value_net_ingredient, value_net_potential

__all__ = [
    "optimize_ingredient",
    "optimize",
    "optimize_tabular_ingredient",
    "optimize_tabular",
    "noise_ingredient",
    "add_noise_potential",
    "fixed_ingredient",
    "add_fixed_potential",
    "reward_ingredient",
    "plot_rewards",
    "heatmap_ingredient",
    "plot_heatmaps",
    "value_net_ingredient",
    "value_net_potential",
]
