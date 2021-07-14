import itertools
from typing import Iterator

from reward_preprocessing.transition import Transition, get_transitions


def has_len(iterator: Iterator, target_length: int):
    """Check whether the given iterator has the desired length.
    Uses up the iterator!

    The advantage of this function over more naive approaches
    such as `sum(1 for _ iterator) == target_length` is that it
    always terminates in O(target_length) runtime, even if the
    iterator is infinite.
    """
    for _ in range(target_length):
        try:
            _ = next(iterator)
        except StopIteration:
            # there are to few elements in the iterator
            # (since we got a StopIteration before getting target_length elements)
            return False
    try:
        next(iterator)
        # There are too many elements in the iterator: we already got out
        # target_length before but there was still another element left
        return False
    except StopIteration:
        # We got exactly target_length elements, followed by a StopIteration,
        # so the iterator had the right length
        return True


def test_apply():
    transition = Transition("a", None, "b")
    new = transition.apply(lambda x: x.upper())
    assert new.state == "A"
    assert new.action is None
    assert new.next_state == "B"


def test_num_transitions_single_env(env, model):
    assert has_len(get_transitions(env, model, num=200), 200)
    assert has_len(get_transitions(env, model, num=0), 0)
    assert has_len(get_transitions(env, model, num=1), 1)
    # By default (with num=None), the iterator should never terminate.
    # We check that at least it generates 200 transitions.
    assert has_len(itertools.islice(get_transitions(env, model), 200), 200)


def test_policy_can_be_none(env):
    # if no policy is passed, it should still work (with random actions)
    assert has_len(get_transitions(env, num=10), 10)


def test_multiple_envs(venv, model):
    assert has_len(get_transitions(venv, model, num=17), 17)


def test_generated_transitions(mock_env):
    """Test whether the generated transitions are correct on a simple example.

    The main purpose is to check the behavior when an episode ends:
    the next_state of the last transition should be the terminal state,
    while the state of the next transition should be the initial state
    of the new episode.
    """

    def policy(states):
        # always walk to the left (towards 0)
        return [0] * len(states)

    data = list(get_transitions(mock_env, policy, num=6))
    expected_states = [5, 4, 3, 2, 1, 5]
    expected_next_states = [4, 3, 2, 1, 0, 4]
    assert len(data) == len(expected_states) == len(expected_next_states)
    for x, exp_state, exp_next_state in zip(
        data, expected_states, expected_next_states
    ):
        transition, reward = x
        assert transition.state == exp_state
        assert transition.next_state == exp_next_state


def test_generated_transitions_multiple_envs(mock_venv):
    """Test whether transitions are generated correctly if the venv contains
    multiple environments.

    We probabldy don't cover all important cases here but at least check
    the basics: we have two environments, one of which doesn't terminate
    during the test, while the other one has an episode end before the end
    of the test. We then check that all the transitions that are produced
    are correct (in particular that one environment resetting doesn't also
    reset the other one or interfere with it in any other way).
    """

    def policy(states):
        assert len(states) == 2
        # In the first env, we walk to the left, in the second we stay near the middle
        # (so that it never terminates)
        if states[1] > 5:
            return [0, 0]
        else:
            return [0, 1]

    data = list(get_transitions(mock_venv, policy, num=12))
    # the 1st, 3rd, ... state belong to the first env,
    # the other ones to the second env
    expected_states = [5, 5, 4, 6, 3, 5, 2, 6, 1, 5, 5, 6]
    expected_next_states = [4, 6, 3, 5, 2, 6, 1, 5, 0, 6, 4, 5]
    assert len(data) == len(expected_states) == len(expected_next_states)
    for x, exp_state, exp_next_state in zip(
        data, expected_states, expected_next_states
    ):
        transition, reward = x
        assert transition.state == exp_state
        assert transition.next_state == exp_next_state
