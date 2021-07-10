from reward_preprocessing.transition import Transition


def test_apply():
    transition = Transition("a", None, "b")
    new = transition.apply(lambda x: x.upper())
    assert new.state == "A"
    assert new.action is None
    assert new.next_state == "B"
