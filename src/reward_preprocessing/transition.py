class Transition:
    """A transition in an MDP, consisting of a state, an action and a following state
    (the reward is not part of this class).

    This class is mostly agnostic to the type of the state, action and next state.
    However, it does provide .to(device) as a convenience method that makes
    working with Transitions consisting of tensors easier, and similar methods may
    be added in the future.
    """

    def __init__(self, state=None, action=None, next_state=None):
        self.state = state
        self.action = action
        self.next_state = next_state

    def apply(self, fn):
        state = None if self.state is None else fn(self.state)
        action = None if self.action is None else fn(self.action)
        next_state = None if self.next_state is None else fn(self.next_state)
        return Transition(state, action, next_state)

    def to(self, device):
        return self.apply(lambda x: x.to(device))
