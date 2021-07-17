import torch

from reward_preprocessing.env.maze import get_agent_positions, is_terminal


def test_get_agent_positions():
    # create a batch of 6 5x5 mazes
    n = 5
    b = 6
    obs = torch.zeros((b, n, n))
    agent_positions = torch.tensor([3, 6, 10, 14, 18, 22])
    goal_positions = torch.tensor([2, 6, 11, 14, 19, 21])

    # add agents
    obs[
        range(b),
        torch.div(agent_positions, n, rounding_mode="floor"),
        agent_positions % n,
    ] = 2
    # add goals
    obs[
        range(b),
        torch.div(goal_positions, n, rounding_mode="floor"),
        goal_positions % n,
    ] = 3
    print(obs)

    assert torch.all(get_agent_positions(obs) == agent_positions), get_agent_positions(
        obs
    )


def test_is_terminal():
    # create a batch of 6 5x5 mazes
    n = 5
    b = 6
    obs = torch.zeros((b, n, n))
    agent_positions = torch.tensor([3, 6, 10, 14, 18, 22])
    goal_positions = torch.tensor([2, 6, 11, 14, 19, 21])

    # add agents
    obs[
        range(b),
        torch.div(agent_positions, n, rounding_mode="floor"),
        agent_positions % n,
    ] = 2
    # add goals
    obs[
        range(b),
        torch.div(goal_positions, n, rounding_mode="floor"),
        goal_positions % n,
    ] = 3
    print(obs)

    # second and fourth state are terminal because agent is on
    # the goal
    assert torch.all(
        is_terminal(obs) == torch.tensor([False, True, False, True, False, False])
    )
