"""Package containing Sacred Ingredients for interpreting reward models.
Each Ingredient should contain at least one capture function which receives
a RewardModel (and if necessary other input, such as episode rollouts or single
transitions) and produces some artifacts or returns values to be used for
interpretation.

The reason to implement these as Sacred ingredients rather than just simple
functions is that each method may have parameters (and we want Sacred's mechanisms
to deal with those) and that they may produce artifacts (which should happen
directly here, rather than in the main script.
"""
