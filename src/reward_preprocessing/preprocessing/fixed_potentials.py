"""Collection of hand-designed potentials for specific environments."""

from imitation.rewards.reward_nets import RewardNet
from stable_baselines3.common.vec_env import VecNormalize

from .potential_shaping import PotentialShaping


class SparseHalfCheetah(PotentialShaping):
    """Potential that is proportional to the negative distance from the origin.

    Here's the reasoning: the HalfCheetah reward is
        r = (x' - x)/dt - action_penalty
    where x' is the next position and x the current one.
    dt is timestep * frameskip, which for HalfCheetah is
        dt = 0.01 * 5 = 0.05
    (you can also check this by just printing the `dt` attribute
    of the gym environment).

    The action penalty is hard to get rid of with an (s, s') model
    like potential shaping gives. But we can mostly get rid of
    the velocity part of the reward, by setting
        Phi(x) = -x/dt

    Then the shaped reward is
        r = (x' - x)/dt - action_penalty + gamma * Phi(x') - Phi(x)
          = (1 - gamma) * x'/dt - action_penalty
    which for gamma close to 1 should be much lower.
    """

    def __init__(self, env: VecNormalize, model: RewardNet, gamma: float):
        self.dt = 0.05  # see docstring for explanation

        def potential(obs):
            # observations should be 19-dimensional; if not,
            # we might have excluded the COM or we're using
            # the wrong environment.
            assert obs.size(1) == 19
            # unnormalize the observation
            # TODO: this can't undo the clipping, which might be
            # a serious issue in HalfCheetah specifically?
            obs = env.unnormalize_obs(obs)
            # The first axis is the batch dimension.
            # The second axis enumerates different positions
            # and velocities. We want the center of mass,
            # which is the first element.
            return -obs[:, 0] / self.dt

        super().__init__(model=model, gamma=gamma, potential=potential)
